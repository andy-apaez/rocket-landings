import numpy as np
import pytest

from rocket_sim import REASONS, EnvConfig, VecRocketEnv
from rocket_sim.controller import PDController


def catch_env(n=2, **catch):
    cfg = EnvConfig(task="catch")
    cfg.terrain.randomize = False
    for k, v in catch.items():
        setattr(cfg.catch, k, v)
    env = VecRocketEnv(cfg, num_envs=n, seed=0, autoreset=False)
    env.reset()
    return env


def put_catch_point_at(env, point):
    """Upright, at rest, with the catch point at `point` (N,3)."""
    rk = env.rocket
    rk.quat[:] = [1, 0, 0, 0]
    rk.vel[:] = 0
    rk.omega[:] = 0
    rk.pos += point - env.catch_point()
    env.potential[:] = env._potential()


def hover(env):
    """Throttle that holds the current altitude, as an action array."""
    mass = env.rocket.derived()["mass"]
    thr = mass * env.gravity / env.cfg.rocket.t_max
    a = np.zeros((env.n, 4))
    a[:, 0] = 2 * thr - 1
    env.rocket.throttle[:] = thr   # skip the throttle ramp-up
    return a


def test_holding_in_the_zone_is_a_catch_and_needs_the_full_hold_time():
    env = catch_env(2)
    target = env.catch_target()
    put_catch_point_at(env, target)
    env.rocket.pos[1, 0] += 3.0          # outside the catch radius
    a = hover(env)
    steps_needed = int(np.ceil(env.cfg.catch.hold_time / env.decision_dt))
    for _ in range(steps_needed - 1):
        env.step(a)
    assert not env.done[0]                # still holding
    _, rew, term, _, info = env.step(a)
    assert term[0] and REASONS[info["reason"][0]] == "caught" and info["landed"][0]
    assert rew[0] > 50
    assert not env.done[1]


def test_tower_is_solid():
    env = catch_env(1)
    tower_top = np.array([[env.tower_xz[0, 0], env.tower_base[0] + 20.0, env.tower_xz[0, 1]]])
    put_catch_point_at(env, tower_top)
    _, rew, term, _, info = env.step(hover(env))
    assert term[0] and REASONS[info["reason"][0]] == "hit_tower"
    assert rew[0] < -50


def test_any_ground_contact_is_a_crash_without_legs():
    env = catch_env(1)
    rk = env.rocket
    rk.quat[:] = [1, 0, 0, 0]
    rk.vel[:] = [0, -0.5, 0]   # would be a perfect landing in the land task
    rk.pos[:] = env.terrain.pad
    rk.pos[:, 1] += env.cfg.rocket.length / 2 + 0.01 - rk.derived()["com_y"]
    while not env.done[0]:
        env.step(np.array([[-1.0, 0, 0, 0]]))
    assert REASONS[env.reason[0]] == "body_contact"


def test_observation_is_catch_relative():
    env = catch_env(1)
    put_catch_point_at(env, env.catch_target() + np.array([[2.0, 3.0, -1.0]]))
    obs = env.observe()[0]
    assert np.allclose(obs[:3], [0.02, 0.03, -0.01], atol=1e-5)
    assert obs[19] == pytest.approx(0.3, abs=1e-5)                 # vertical catch error / 10
    assert np.allclose(obs[20:22], [0.2, -0.1], atol=1e-5)         # horizontal catch error / 10


def test_pd_controller_catches_on_earth():
    cfg = EnvConfig(task="catch")
    env = VecRocketEnv(cfg, num_envs=64, seed=7, autoreset=False)
    env.reset()
    ctl = PDController()
    while not env.done.all():
        env.step(ctl(env))
    assert np.mean(env.reason == REASONS.index("caught")) > 0.9


def test_bad_task_rejected():
    with pytest.raises(ValueError):
        EnvConfig(task="hover")
