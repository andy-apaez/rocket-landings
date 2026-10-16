"""Interactive mode: launch rockets from the 3D viewer and train the policy launch by launch.

    python play.py                     # then open http://localhost:8000/viewer/
    python play.py --fresh             # discard the saved interactive models and start over

Each launch from the viewer:
  1. trains the PPO policy for the chosen number of steps on the batched sim (same body as the launch),
  2. flies one rocket under the launch conditions you set, with the updated policy (or the PD autopilot),
  3. sends the flight back to the viewer and adds it to the launch history.

Orbital missions (the default view): lift off from a launch site into orbit, pick a landing site on the globe,
then the lander deorbits and the policy trains on that site's terrain and wind before flying the landing
(see rocket_sim/orbit.py and viewer/mission.js).

Each task (land on legs / tower catch) has its own model and launch history in runs/interactive/<task>/,
so a later session continues where you left off.
"""

import argparse
import functools
import json
import shutil
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from rocket_sim import EnvConfig
from rocket_sim import orbit as orb
from rocket_sim.replay import launch_setup, record, save
from train import make_ppo, make_training_env

ROOT = Path(__file__).resolve().parent
BODIES = ("earth", "mars", "moon")
TASKS = ("land", "catch")
MAX_TRAIN_STEPS = 2_000_000

# Slider ranges match the training reset distribution (SimConfig); outside it the policy has never practiced.
LIMITS = {
    "altitude": (120.0, 250.0), "offset_x": (-80.0, 80.0), "offset_z": (-80.0, 80.0),
    "descent": (5.0, 30.0), "drift_x": (-10.0, 10.0), "drift_z": (-10.0, 10.0),
    "tilt_deg": (0.0, 14.0), "fuel": (0.6, 1.0),
}


class Session:
    def __init__(self, out: Path, seed: int, fresh: bool, task: str = "land"):
        self.task = task
        self.out = out
        self.replay_dir = out / "replays"
        if fresh and out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.lock = threading.Lock()  # one launch/training at a time
        self.status = {"phase": "idle", "done": 0, "total": 0}
        self._build(load=not fresh)

    # Model
    def _build(self, load: bool):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        self.cfg = EnvConfig(task=self.task)
        venv, self.raw = make_training_env(self.cfg, seed=self.seed)
        model_path, stats_path = self.out / "ppo_rocket.zip", self.out / "vecnormalize.pkl"
        if load and model_path.exists():
            if stats_path.exists():
                venv = VecNormalize.load(str(stats_path), venv.venv)
            self.model = PPO.load(model_path, env=venv, device="cpu")
            print(f"loaded {self.task} model {model_path} ({self.model.num_timesteps:,} steps trained)", flush=True)
        else:
            self.model = make_ppo(venv, self.seed, verbose=0)
        self.venv = venv
        self.launches = len(self._manifest())

    def _save_model(self):
        self.model.save(self.out / "ppo_rocket")
        self.venv.save(str(self.out / "vecnormalize.pkl"))

    def _manifest(self):
        p = self.replay_dir / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else []

    def reset(self):
        with self.lock:
            shutil.rmtree(self.out, ignore_errors=True)
            self.out.mkdir(parents=True)
            self._build(load=False)

    def train(self, steps: int):
        from stable_baselines3.common.callbacks import BaseCallback

        session = self

        class Progress(BaseCallback):
            def _on_training_start(self):
                self.start = self.num_timesteps

            def _on_step(self):
                session.status["done"] = self.num_timesteps - self.start
                return True

        self.status.update(phase="training", done=0, total=steps)
        self.model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=Progress())

    # Launch
    def launch(self, req: dict) -> dict:
        body = req.get("body", "earth")
        if body not in BODIES:
            raise ValueError(f"unknown body {body!r}")
        pilot = req.get("pilot", "ppo")
        train_steps = int(np.clip(int(req.get("train_steps", 0)), 0, MAX_TRAIN_STEPS))
        cond = {k: float(np.clip(float(req[k]), *LIMITS[k])) for k in LIMITS if req.get(k) is not None}
        site = req.get("site")          # from orbit.site_conditions, for a landing picked on the globe
        with self.lock:
            t0 = time.time()
            # Train: on the body (and landing site) being launched, so the policy practices what you ask it to do
            self.raw.sim.cfg.sim.bodies = (body,)
            self.cfg.sim.bodies = (body,)
            orb.apply_site(self.raw.sim.cfg, site)
            orb.apply_site(self.cfg, site)
            if pilot == "ppo" and train_steps > 0:
                self.train(train_steps)
            self.status.update(phase="flying", done=0, total=0)

            # Fly
            cfg = EnvConfig(task=self.task)
            cfg.sim.bodies = (body,)
            orb.apply_site(cfg, site)
            if pilot == "pd":
                from rocket_sim.controller import PDController
                ctl = PDController()
                policy = lambda env, obs: ctl(env)  # noqa: E731
            else:
                model = self.model
                policy = lambda env, obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
            seed = (int(req["seed"]) if req.get("seed") is not None else site["seed"] if site
                    else int(np.random.default_rng().integers(1 << 30)))
            replay = record(cfg, policy, num_rockets=1, seed=seed, setup=launch_setup(cond))

            # Save
            self.launches += 1
            steps = int(self.model.num_timesteps)
            rk = replay["rockets"][0]
            task = " · catch" if self.task == "catch" else ""
            where = f" · {site['lat']:.1f}°, {site['lon']:.1f}°" if site else ""
            label = f"#{self.launches} · {'PPO' if pilot == 'ppo' else 'autopilot'} · {body}{task}{where} · {steps / 1e6:.2f}M"
            name = f"launch_{self.launches:05d}"
            save(replay, name, self.replay_dir, label=label, step=steps,
                 extra={"launch": self.launches, "pilot": pilot, "outcome": rk["outcome"], "body": body,
                        "trained": train_steps if pilot == "ppo" else 0, "mission": bool(site)})
            if pilot == "ppo" and train_steps > 0:
                self._save_model()
            self.status.update(phase="idle")
            replay["meta"]["label"] = label
            replay["meta"]["file"] = f"{name}.json"
            replay["meta"]["seconds"] = round(time.time() - t0, 1)
            if site:
                replay["meta"]["site"] = site
            return replay

    # Mission
    def mission_land(self, req: dict) -> dict:
        """Deorbit to the site picked on the globe, then train and fly the landing under its conditions."""
        lat, lon = float(req["lat"]), float(req["lon"])
        site = orb.site_conditions(lat, lon, float(req.get("ruggedness", 0.3)), req.get("surface", "land"),
                                   req.get("color"))
        plan = mission_plan(req)
        if not plan["reachable"]:
            raise ValueError(f"site out of reach: needs {plan['dv']:.0f} m/s, the stage has {plan['budget_dv']:.0f}")
        cond = orb.landing_conditions(plan, site)
        replay = self.launch({"pilot": req.get("pilot", "ppo"), "body": "earth",
                              "train_steps": req.get("train_steps", 0), "site": site, **cond})
        return {"plan": plan, "site": site, "conditions": cond, "replay": replay}

    def state(self) -> dict:
        return {
            "task": self.task,
            "tasks": TASKS,
            "status": self.status,
            "model_steps": int(self.model.num_timesteps),
            "launches": self.launches,
            "replay_dir": "/" + str(self.replay_dir.relative_to(ROOT)),
            "limits": LIMITS,
            "bodies": BODIES,
            "max_train_steps": MAX_TRAIN_STEPS,
            "rollout_steps": self.model.n_steps * self.model.n_envs,
            "launch_sites": {k: {"name": n, "lat": la, "lon": lo} for k, (n, la, lo) in orb.LAUNCH_SITES.items()},
            "vehicle": orb.vehicle_stats(orb.LaunchVehicle()),
            "orbit_limits": {"altitude": [orb.MIN_ORBIT_ALT / 1e3, orb.MAX_ORBIT_ALT / 1e3], "inclination": [0.0, 120.0]},
        }


# Routes
def mission_ascent(req: dict) -> dict:
    """Lift-off: fly the launcher from a launch site into the requested orbit."""
    key = req.get("site", "cape")
    if key not in orb.LAUNCH_SITES:
        raise ValueError(f"unknown launch site {key!r}")
    name, lat, lon = orb.LAUNCH_SITES[key]
    out = orb.ascent(lat, lon, float(req.get("altitude", 300)) * 1e3, float(req.get("inclination", 53.0)))
    out["site"].update(key=key, name=name)
    return out


def mission_plan(req: dict) -> dict:
    """First affordable deorbit opportunity from the current orbit to (lat, lon)."""
    plan = orb.plan_landing(req["orbit"], float(req["lat"]), float(req["lon"]), float(req.get("t_now", 0.0)),
                            float(req["budget_dv"]))
    plan["reachable"] = bool(plan["reachable"])
    return plan


# Server
class Sessions:
    """One Session per task, built the first time that task is used."""

    def __init__(self, out: Path, seed: int, fresh: bool):
        self.out, self.seed, self.fresh = out, seed, fresh
        self.lock = threading.Lock()
        self.by_task: dict[str, Session] = {}
        if fresh and out.exists():
            shutil.rmtree(out)
        self._migrate_legacy()

    def _migrate_legacy(self):
        """Older versions kept a single (land) model directly in `out`; move it into out/land/."""
        legacy, land = self.out / "ppo_rocket.zip", self.out / "land"
        if not legacy.exists() or (land / "ppo_rocket.zip").exists():
            return
        land.mkdir(parents=True, exist_ok=True)
        for name in ("ppo_rocket.zip", "vecnormalize.pkl", "replays"):
            src, dst = self.out / name, land / name
            if not src.exists():
                continue
            if dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()                     # empty folder left by a newer server run
            if not dst.exists():
                shutil.move(str(src), str(dst))
        print(f"moved the existing land model and launch history into {land}", flush=True)

    def get(self, task: str | None) -> Session:
        task = task or "land"
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}")
        with self.lock:
            if task not in self.by_task:
                self.by_task[task] = Session(self.out / task, self.seed, fresh=False, task=task)
            return self.by_task[task]


class Handler(SimpleHTTPRequestHandler):
    sessions: Sessions = None

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _task(self):
        return parse_qs(urlparse(self.path).query).get("task", [None])[0]

    def do_GET(self):
        if self.path.split("?")[0] == "/api/state":
            try:
                return self._json(self.sessions.get(self._task()).state())
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        session = None
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            route = self.path.split("?")[0]
            session = self.sessions.get(req.get("task") or self._task())
            if route == "/api/launch":
                return self._json(session.launch(req))
            if route == "/api/mission/ascent":
                return self._json(mission_ascent(req))
            if route == "/api/mission/plan":
                return self._json(mission_plan(req))
            if route == "/api/mission/land":
                return self._json(session.mission_land(req))
            if route == "/api/reset":
                session.reset()
                return self._json(session.state())
            self._json({"error": "not found"}, 404)
        except Exception as e:  # report to the viewer instead of dropping the connection
            if session is not None:
                session.status.update(phase="idle")
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", default="runs/interactive")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="start a new model, deleting the saved one")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    Handler.sessions = Sessions(ROOT / args.out, args.seed, args.fresh)
    Handler.sessions.get("land")  # build the default model before the first request
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(ROOT)))
    url = f"http://localhost:{args.port}/viewer/"
    print(f"Rocket launch control at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
