"""Train or evaluate a PPO policy on the batched rocket landing env.

    python train.py train --steps 5_000_000 --bodies earth mars moon
    python train.py eval  --model runs/ppo_rocket.zip
    python train.py baseline              # scripted PD controller for comparison

Training writes replays to runs/replays/ for the 3D viewer: python -m http.server, open /viewer/.
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from rocket_sim import REASONS, EnvConfig, VecRocketEnv


def make_cfg(args) -> EnvConfig:
    cfg = EnvConfig(task=args.task)
    cfg.sim.bodies = tuple(args.bodies)
    cfg.sim.num_envs = args.num_envs
    if args.flat:
        cfg.terrain.randomize = False
    return cfg


def summarize(reasons: np.ndarray, returns: np.ndarray, label: str):
    counts = collections.Counter(REASONS[r] for r in reasons)
    ok = np.isin(reasons, [REASONS.index("landed"), REASONS.index("caught")])
    print(f"{label}: success {ok.mean():.1%} | "
          f"mean return {returns.mean():.1f} | " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))


def rollout(cfg, policy, episodes, seed):
    """Run `episodes` episodes in one batch. policy(env, obs) -> actions."""
    env = VecRocketEnv(cfg, num_envs=episodes, seed=seed, autoreset=False)
    obs = env.reset()
    while not env.done.all():
        obs, *_ = env.step(policy(env, obs))
    return env.reason.copy(), env.ep_return.copy()


def make_training_env(cfg: EnvConfig, seed: int | None = None):
    """Batched SB3 env with episode stats and reward scaling. Returns (venv, raw RocketVecEnv)."""
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    from rocket_sim.gym_env import make_sb3_vec_env

    raw = make_sb3_vec_env(cfg, seed=seed)
    venv = VecMonitor(raw, info_keywords=("is_success",))
    # Observations are already normalized by the env; only rewards are scaled for PPO.
    return VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995), raw


def make_ppo(venv, seed: int = 0, tensorboard_log: str | None = None, verbose: int = 1):
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy", venv,
        n_steps=256, batch_size=4096, n_epochs=5, learning_rate=3e-4,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
        tensorboard_log=tensorboard_log, seed=seed, verbose=verbose,
    )


def cmd_train(args):
    from stable_baselines3.common.callbacks import CheckpointCallback

    cfg = make_cfg(args)
    venv, _ = make_training_env(cfg, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = make_ppo(venv, args.seed, str(out / "tb") if args.tensorboard else None)
    ckpt = CheckpointCallback(save_freq=max(200_000 // cfg.sim.num_envs, 1), save_path=str(out / "ckpt"),
                              name_prefix="ppo_rocket")
    callbacks = [ckpt]
    if args.replay_every > 0:
        from rocket_sim.replay import make_replay_callback
        callbacks.append(make_replay_callback(cfg, args.replay_every, out_dir=out / "replays"))
    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(out / "ppo_rocket")
    print(f"saved {out / 'ppo_rocket.zip'}")


def cmd_eval(args):
    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    cfg = make_cfg(args)
    reasons, returns = rollout(cfg, lambda env, obs: model.predict(obs, deterministic=True)[0],
                               args.episodes, args.seed)
    summarize(reasons, returns, "PPO")


def cmd_baseline(args):
    from rocket_sim.controller import PDController

    ctl = PDController()
    cfg = make_cfg(args)
    for body in args.bodies:
        cfg.sim.bodies = (body,)
        reasons, returns = rollout(cfg, lambda env, obs: ctl(env), args.episodes, args.seed)
        summarize(reasons, returns, f"PD [{body}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "baseline"])
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--bodies", nargs="+", default=["earth"])
    p.add_argument("--task", choices=["land", "catch"], default="land",
                   help="land on legs, or get caught by the tower arms")
    p.add_argument("--flat", action="store_true", help="flat terrain (easier curriculum start)")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--model", default="runs/ppo_rocket.zip")
    p.add_argument("--out", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--replay-every", type=int, default=250_000,
                   help="record a 3D-viewer replay every N timesteps (0 disables)")
    args = p.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "baseline": cmd_baseline}[args.mode](args)


if __name__ == "__main__":
    main()
