"""Gymnasium and Stable-Baselines3 adapters around VecRocketEnv."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .env import ACT_DIM, OBS_DIM, REASONS, VecRocketEnv

OBS_SPACE = spaces.Box(-10.0, 10.0, (OBS_DIM,), np.float32)
ACT_SPACE = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)


class RocketLandingEnv(gym.Env):
    """Single-rocket Gymnasium env. For training prefer make_sb3_vec_env (batched)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None):
        self.sim = VecRocketEnv(cfg, num_envs=1, autoreset=False)
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset(seed=seed if seed is not None else int(self.np_random.integers(2**31)))
        return obs[0], {}

    def step(self, action):
        obs, rew, term, trunc, info = self.sim.step(np.asarray(action)[None])
        return obs[0], float(rew[0]), bool(term[0]), bool(trunc[0]), {
            "reason": REASONS[info["reason"][0]], "landed": bool(info["landed"][0])}


gym.register("RocketLanding-v0", entry_point=RocketLandingEnv)


def make_sb3_vec_env(cfg: EnvConfig | None = None, num_envs: int | None = None, seed: int | None = None):
    """Wrap the batched sim as an SB3 VecEnv so all N rockets step in one numpy call."""
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class RocketVecEnv(VecEnv):
        def __init__(self):
            self.sim = VecRocketEnv(cfg, num_envs=num_envs, seed=seed, autoreset=True)
            super().__init__(self.sim.n, OBS_SPACE, ACT_SPACE)
            self._actions = None

        def reset(self):
            return self.sim.reset()

        def step_async(self, actions):
            self._actions = actions

        def step_wait(self):
            obs, rew, term, trunc, info = self.sim.step(self._actions)
            dones = info["done"]
            infos = [{} for _ in range(self.num_envs)]
            for i in np.flatnonzero(dones):
                infos[i] = {
                    "terminal_observation": info["final_obs"][i],
                    "TimeLimit.truncated": bool(trunc[i] and not term[i]),
                    "episode": {"r": float(info["episode_return"][i]), "l": int(info["episode_length"][i])},
                    "is_success": bool(info["landed"][i]),
                    "reason": REASONS[info["reason"][i]],
                }
            return obs, rew, dones.copy(), infos

        def close(self):
            pass

        def seed(self, seed=None):
            self.sim.rng = np.random.default_rng(seed)
            return [seed] * self.num_envs

        def get_attr(self, attr_name, indices=None):
            return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

        def set_attr(self, attr_name, value, indices=None):
            setattr(self, attr_name, value)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return [None] * len(self._get_indices(indices))

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._get_indices(indices))

        def _get_indices(self, indices):
            if indices is None:
                return range(self.num_envs)
            return [indices] if isinstance(indices, int) else indices

    return RocketVecEnv()
