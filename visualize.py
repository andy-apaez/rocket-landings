"""Plot one landing attempt: 3D trajectory over the terrain plus telemetry.

    python visualize.py                       # PD controller
    python visualize.py --model runs/ppo_rocket.zip --body mars --seed 4
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rocket_sim import REASONS, EnvConfig, VecRocketEnv
from rocket_sim.physics import tilt_angle, up_vector


def record(env: VecRocketEnv, policy):
    obs = env.reset()
    log = {k: [] for k in ("t", "pos", "up", "speed", "throttle", "tilt", "fuel", "clear")}
    while not env.done[0]:
        rk = env.rocket
        log["t"].append(env.time[0])
        log["pos"].append(rk.pos[0].copy())
        log["up"].append(up_vector(rk.quat)[0])
        log["speed"].append(np.linalg.norm(rk.vel[0]))
        log["throttle"].append(rk.throttle[0])
        log["tilt"].append(np.degrees(tilt_angle(rk.quat)[0]))
        log["fuel"].append(rk.fuel[0] / env.cfg.rocket.fuel_capacity)
        log["clear"].append(env._foot_clearance()[0])
        obs, *_ = env.step(policy(env, obs))
    return {k: np.array(v) for k, v in log.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="PPO .zip; omit to use the PD controller")
    p.add_argument("--body", default="earth")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="landing.png")
    args = p.parse_args()

    cfg = EnvConfig()
    cfg.sim.bodies = (args.body,)
    env = VecRocketEnv(cfg, num_envs=1, seed=args.seed, autoreset=False)
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        policy, name = (lambda e, o: model.predict(o, deterministic=True)[0]), "PPO"
    else:
        from rocket_sim.controller import PDController
        ctl = PDController()
        policy, name = (lambda e, o: ctl(e)), "PD controller"

    log = record(env, policy)
    outcome = REASONS[env.reason[0]]

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    t = env.terrain
    axis = np.linspace(-t.size / 2, t.size / 2, t.cfg.grid)
    X, Z = np.meshgrid(axis, axis, indexing="ij")
    ax.plot_surface(X, Z, t.heights[0], cmap="terrain", alpha=0.5, linewidth=0)
    pos = log["pos"]
    ax.plot(pos[:, 0], pos[:, 2], pos[:, 1], color="crimson", lw=2, label="trajectory")
    every = max(len(pos) // 25, 1)
    L = cfg.rocket.length
    for p_, u in zip(pos[::every], log["up"][::every]):
        a, b = p_ - u * L / 2, p_ + u * L / 2
        ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], color="k", lw=1)
    pad = t.pad[0]
    ang = np.linspace(0, 2 * np.pi, 50)
    r = cfg.criteria.max_pad_dist
    ax.plot(pad[0] + r * np.cos(ang), pad[2] + r * np.sin(ang), np.full(50, pad[1] + 0.2), color="blue", lw=2,
            label="pad")
    lo = np.minimum(pos[:, [0, 2]].min(0), pad[[0, 2]]) - 30
    hi = np.maximum(pos[:, [0, 2]].max(0), pad[[0, 2]]) + 30
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("altitude (m)")
    ax.set_title(f"{name} on {args.body}: {outcome}")
    ax.legend(loc="upper left")

    series = [("clear", "leg clearance (m)"), ("speed", "speed (m/s)"), ("tilt", "tilt (°)"),
              ("throttle", "throttle"), ("fuel", "fuel fraction")]
    for i, (key, label) in enumerate(series):
        a = fig.add_subplot(len(series), 2, 2 * i + 2)
        a.plot(log["t"], log[key], color="tab:blue")
        a.set_ylabel(label, fontsize=8)
        a.grid(alpha=0.3)
        if i < len(series) - 1:
            a.tick_params(labelbottom=False)
    a.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"{outcome} after {log['t'][-1]:.1f}s, saved {args.out}")


if __name__ == "__main__":
    main()
