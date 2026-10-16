"""Record landing attempts to JSON for the 3D viewer (viewer/index.html).

    python -m rocket_sim.replay --pd                                   # scripted controller
    python -m rocket_sim.replay --model runs/ppo_rocket.zip --body mars
    python -m http.server 8000      # then open http://localhost:8000/viewer/

All rockets in one replay share a single terrain so they can be shown together.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import REASONS, VecRocketEnv
from .physics import quat_from_axis_angle, quat_mul

REPLAY_DIR = Path("runs/replays")


def record(cfg: EnvConfig, policy, num_rockets: int = 16, seed: int = 0, setup=None) -> dict:
    """Run one batch to completion. policy(env, obs) -> actions. Returns a JSON-ready dict.

    setup(env), if given, runs after the reset to override the starting state (see launch_setup).
    """
    env = VecRocketEnv(cfg, num_envs=num_rockets, seed=seed, autoreset=False)
    env.reset()
    _share_terrain(env)
    if setup is not None:
        setup(env)
        env.potential[:] = env._potential()
    obs = env.observe()

    frames = []  # per decision step: (N, 12) = x y z qw qx qy qz throttle gx gz fuel com_y
    end_step = np.full(num_rockets, -1)
    while not env.done.all():
        frames.append(_snapshot(env))
        obs, *_ = env.step(policy(env, obs))
        end_step[(end_step < 0) & env.done] = len(frames)
    frames.append(_snapshot(env))

    data = np.stack(frames, 1)  # (N, T, 12)
    t = env.terrain
    spec = cfg.rocket
    return {
        "meta": {
            "dt": env.decision_dt,
            "body": list(cfg.sim.bodies),
            "rocket": {"length": spec.length, "radius": spec.radius, "leg_radius": spec.leg_radius,
                       "leg_drop": spec.leg_drop, "max_gimbal": spec.max_gimbal},
            "pad": t.pad[0].round(3).tolist(),
            "pad_radius": cfg.criteria.max_pad_dist,
            "flat_radius": cfg.terrain.flatten_radius,
            "task": cfg.task,
            "tower": {
                "base": [float(env.tower_xz[0, 0]), float(env.tower_base[0]), float(env.tower_xz[0, 1])],
                "direction": list(cfg.catch.tower_direction),
                "width": cfg.catch.tower_width, "height": cfg.catch.tower_height,
                "catch_target": env.catch_target()[0].round(3).tolist(),
                "catch_radius": cfg.catch.catch_radius,
            },
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "terrain": {"grid": t.cfg.grid, "size": t.size, "heights": t.heights[0].round(2).tolist()},
        "rockets": [
            {
                "outcome": REASONS[env.reason[i]],
                "end": int(end_step[i]),
                "return": round(float(env.ep_return[i]), 2),
                "frames": data[i, : end_step[i] + 1].round(3).tolist(),
            }
            for i in range(num_rockets)
        ],
    }


def launch_setup(cond: dict):
    """Build a setup(env) that places every rocket per user launch conditions.

    Keys (all optional; missing ones keep the random reset value): altitude (m above ground),
    offset_x / offset_z (m from the pad), descent (m/s, positive = down), drift_x / drift_z (m/s),
    tilt_deg, fuel (fraction 0..1).
    """
    def setup(env: VecRocketEnv):
        rk, t, n = env.rocket, env.terrain, env.n
        pad = t.pad
        if "offset_x" in cond or "offset_z" in cond or "altitude" in cond:
            x = pad[:, 0] + cond.get("offset_x", rk.pos[:, 0] - pad[:, 0])
            z = pad[:, 2] + cond.get("offset_z", rk.pos[:, 2] - pad[:, 2])
            ground = t.height_at(np.stack([x, z], -1)[:, None, :])[:, 0]
            alt = cond.get("altitude", rk.pos[:, 1] - ground)
            rk.pos[:] = np.stack([x, ground + alt, z], -1)
        if "descent" in cond:
            rk.vel[:, 1] = -abs(cond["descent"])
        if "drift_x" in cond:
            rk.vel[:, 0] = cond["drift_x"]
        if "drift_z" in cond:
            rk.vel[:, 2] = cond["drift_z"]
        if "tilt_deg" in cond:
            az = env.rng.uniform(0, 2 * np.pi, n)
            axis = np.stack([np.cos(az), np.zeros(n), np.sin(az)], -1)
            q_tilt = quat_from_axis_angle(axis, np.full(n, np.radians(cond["tilt_deg"])))
            q_roll = quat_from_axis_angle(np.tile([0.0, 1.0, 0.0], (n, 1)), env.rng.uniform(-np.pi, np.pi, n))
            rk.quat[:] = quat_mul(q_tilt, q_roll)
        if "fuel" in cond:
            rk.fuel[:] = np.clip(cond["fuel"], 0.0, 1.0) * env.cfg.rocket.fuel_capacity
    return setup


def _share_terrain(env: VecRocketEnv):
    """Copy rocket 0's terrain to every rocket, keeping each start position relative to its own pad."""
    t, rk = env.terrain, env.rocket
    offset = t.pad[0] - t.pad
    rk.pos += offset
    t.heights[:] = t.heights[0]
    t.pad[:] = t.pad[0]
    env.tower_xz[:] = env.tower_xz[0]
    env.tower_base[:] = env.tower_base[0]
    env.potential[:] = env._potential()


def _snapshot(env: VecRocketEnv) -> np.ndarray:
    rk = env.rocket
    return np.concatenate([
        rk.pos, rk.quat, rk.throttle[:, None], rk.gimbal,
        (rk.fuel / env.cfg.rocket.fuel_capacity)[:, None], rk.derived()["com_y"][:, None],
    ], axis=1)


def save(replay: dict, name: str, out_dir: Path = REPLAY_DIR, label: str | None = None, step: int | None = None,
         extra: dict | None = None):
    """Write a replay and add it to the manifest the viewer polls."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(replay, separators=(",", ":")))
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [m for m in manifest if m["file"] != f"{name}.json"]
    outcomes = [r["outcome"] for r in replay["rockets"]]
    manifest.append({
        "file": f"{name}.json",
        "label": label or name,
        "step": step,
        "landed": sum(o in ("landed", "caught") for o in outcomes),
        "total": len(outcomes),
        "mean_return": round(float(np.mean([r["return"] for r in replay["rockets"]])), 1),
        "created": replay["meta"]["created"],
        **(extra or {}),
    })
    manifest.sort(key=lambda m: (m["step"] is None, m["step"] or 0, m["created"]))
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(manifest_path)  # atomic, so the viewer never reads a half-written file
    return out_dir / f"{name}.json"


def make_replay_callback(cfg: EnvConfig, every: int, num_rockets: int = 16, seed: int = 123,
                         out_dir: Path = REPLAY_DIR):
    """SB3 callback: every `every` timesteps, record the current policy on a fixed seed."""
    from stable_baselines3.common.callbacks import BaseCallback

    class ReplayCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_at = 0

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_at:
                self.next_at = self.num_timesteps + every
                model = self.model
                replay = record(cfg, lambda env, obs: model.predict(obs, deterministic=True)[0],
                                num_rockets, seed)
                save(replay, f"step_{self.num_timesteps:09d}", out_dir,
                     label=f"step {self.num_timesteps / 1e6:.2f}M", step=self.num_timesteps)
            return True

    return ReplayCallback()


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pd", action="store_true", help="use the scripted PD controller")
    src.add_argument("--model", help="PPO .zip to record")
    p.add_argument("--body", default="earth")
    p.add_argument("--task", choices=["land", "catch"], default="land")
    p.add_argument("--rockets", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", help="replay file name (without .json)")
    args = p.parse_args()

    cfg = EnvConfig(task=args.task)
    cfg.sim.bodies = (args.body,)
    if args.pd:
        from .controller import PDController
        ctl = PDController()
        policy, label = (lambda env, obs: ctl(env)), f"PD controller · {args.body}"
    else:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        policy, label = (lambda env, obs: model.predict(obs, deterministic=True)[0]), f"{Path(args.model).stem} · {args.body}"

    replay = record(cfg, policy, args.rockets, args.seed)
    task = "" if args.task == "land" else "_catch"
    name = args.name or ("pd" if args.pd else Path(args.model).stem) + f"{task}_{args.body}_s{args.seed}"
    path = save(replay, name, label=label + (" · catch" if task else ""))
    ok = sum(r["outcome"] in ("landed", "caught") for r in replay["rockets"])
    print(f"{ok}/{args.rockets} {'caught' if task else 'landed'} · saved {path}")


if __name__ == "__main__":
    main()
