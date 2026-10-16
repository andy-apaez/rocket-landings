import numpy as np
import pytest

from rocket_sim import OBS_DIM, REASONS, EnvConfig, VecRocketEnv
from rocket_sim.config import G0
from rocket_sim.controller import PDController
from rocket_sim.physics import RocketBatch, quat_from_axis_angle, tilt_angle, up_vector

LANDED = REASONS.index("landed")


def flat_env(n=4, **sim):
    cfg = EnvConfig()
    cfg.terrain.randomize = False
    for k, v in sim.items():
        setattr(cfg.sim, k, v)
    env = VecRocketEnv(cfg, num_envs=n, seed=0, autoreset=False)
    env.reset()
    return env


def place(env, alt, vel=(0, 0, 0), quat=(1, 0, 0, 0)):
    """Put every rocket `alt` meters (leg clearance) above the pad, at rest unless vel is given."""
    rk = env.rocket
    rk.quat[:] = quat
    rk.omega[:] = 0
    rk.vel[:] = vel
    rk.pos[:] = env.terrain.pad
    rk.pos[:, 1] += alt - env._foot_clearance()
    env.potential[:] = env._potential()


def test_free_fall_matches_gravity():
    env = flat_env(1, start_vvel=(0.0, 0.0))
    env.density[:] = 0.0
    place(env, alt=200.0)
    y0 = env.rocket.pos[0, 1]
    for _ in range(20):  # 2 s with engine off
        env.step(np.array([[-1.0, 0, 0, 0]]))
    t = env.time[0]
    assert env.rocket.vel[0, 1] == pytest.approx(-env.gravity[0] * t, rel=1e-6)
    assert y0 - env.rocket.pos[0, 1] == pytest.approx(0.5 * env.gravity[0] * t ** 2, rel=0.01)


def test_fuel_burn_follows_rocket_equation_rate():
    spec = EnvConfig().rocket
    rk = RocketBatch(spec, 1)
    rk.throttle[:] = 1.0
    dt = 0.01
    burned = sum(rk.step(dt, np.ones(1), np.zeros((1, 2)), np.zeros(1), np.full(1, 9.81), np.zeros(1),
                         np.ones(1), np.zeros((1, 3)))[0] for _ in range(100))
    assert burned == pytest.approx(spec.t_max / (spec.isp * G0) * 1.0, rel=1e-6)


def test_engine_off_below_min_throttle_and_without_fuel():
    spec = EnvConfig().rocket
    rk = RocketBatch(spec, 2)
    rk.throttle[:] = [spec.min_throttle * 0.9, 1.0]
    rk.fuel[:] = [spec.fuel_capacity, 0.0]
    burned = rk.step(0.01, rk.throttle.copy(), np.zeros((2, 2)), np.zeros(2), np.full(2, 9.81), np.zeros(2),
                     np.ones(2), np.zeros((2, 3)))
    assert np.all(burned == 0)
    assert np.allclose(rk.vel[:, 1], -9.81 * 0.01)


def test_gimbal_produces_torque_and_quaternion_stays_unit():
    spec = EnvConfig().rocket
    rk = RocketBatch(spec, 1)
    rk.throttle[:] = 0.8
    for _ in range(200):
        rk.step(0.01, np.full(1, 0.8), np.array([[spec.max_gimbal, 0.0]]), np.zeros(1), np.full(1, 9.81),
                np.zeros(1), np.ones(1), np.zeros((1, 3)))
    assert rk.omega[0, 0] < -0.1          # +δx pitches about -x
    assert abs(rk.omega[0, 2]) < 1e-9
    assert np.linalg.norm(rk.quat[0]) == pytest.approx(1.0)


def test_derived_mass_properties():
    spec = EnvConfig().rocket
    rk = RocketBatch(spec, 2)
    rk.fuel[:] = [spec.fuel_capacity, 0.0]
    d = rk.derived()
    assert d["mass"][0] == spec.dry_mass + spec.fuel_capacity
    assert d["com_y"][0] < 0 and d["com_y"][1] == 0     # fuel pulls the COM toward the engine
    assert d["lever_arm"][0] < d["lever_arm"][1]
    assert d["inertia"][0, 0] > d["inertia"][1, 0]


def test_terrain_is_flat_at_pad():
    cfg = EnvConfig()
    env = VecRocketEnv(cfg, num_envs=16, seed=1)
    env.reset()
    t = env.terrain
    offsets = np.array([[0, 0], [3, 0], [0, -3], [-2, 2]], float)
    h, nrm = t.height_and_normal(t.pad[:, None, [0, 2]] + offsets[None])
    assert np.allclose(h, t.pad[:, 1:2], atol=1e-6)
    assert np.allclose(nrm[..., 1], 1.0)


def test_gentle_touchdown_lands_and_fast_one_crashes():
    env = flat_env(3)
    place(env, alt=0.05, vel=(0, -1.0, 0))
    env.rocket.vel[1] = [0, -5.0, 0]   # too fast vertically
    env.rocket.vel[2] = [2.0, -1.0, 0]  # too fast horizontally
    _, rew, term, _, info = env.step(np.zeros((3, 4)))
    assert term.all()
    assert [REASONS[r] for r in info["reason"]] == ["landed", "crash_vspeed", "crash_hspeed"]
    assert rew[0] > 50 > -50 > rew[1]


def test_tilted_touchdown_and_off_pad():
    env = flat_env(2)
    q = quat_from_axis_angle(np.array([[1.0, 0, 0]]), np.array([0.4]))[0]
    place(env, alt=0.05, vel=(0, -0.5, 0))
    env.rocket.quat[0] = q
    env.rocket.pos[1, 0] += 20.0
    while not env.done.all():
        env.step(np.zeros((2, 4)) - np.array([1, 0, 0, 0]))
    assert REASONS[env.reason[0]] == "crash_tilt"
    assert REASONS[env.reason[1]] == "off_pad"


def test_autoreset_returns_final_obs():
    cfg = EnvConfig()
    env = VecRocketEnv(cfg, num_envs=8, seed=0, autoreset=True)
    obs = env.reset()
    assert obs.shape == (8, OBS_DIM) and obs.dtype == np.float32
    for _ in range(2000):
        obs, rew, term, trunc, info = env.step(np.tile([-1.0, 0, 0, 0], (8, 1)))  # engine off: falls
        if info["done"].any():
            break
    assert "final_obs" in info
    assert not env.done.any()  # done envs were reset


def test_pd_controller_lands_on_earth():
    cfg = EnvConfig()
    env = VecRocketEnv(cfg, num_envs=64, seed=7, autoreset=False)
    env.reset()
    ctl = PDController()
    while not env.done.all():
        env.step(ctl(env))
    assert np.mean(env.reason == LANDED) > 0.9


def test_gymnasium_env_passes_checker():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from rocket_sim.gym_env import RocketLandingEnv

    env = RocketLandingEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_sb3_vec_env_steps():
    pytest.importorskip("stable_baselines3")
    from rocket_sim.gym_env import make_sb3_vec_env

    venv = make_sb3_vec_env(num_envs=4, seed=0)
    obs = venv.reset()
    assert obs.shape == (4, OBS_DIM)
    for _ in range(500):
        obs, rew, dones, infos = venv.step(np.tile([-1.0, 0, 0, 0], (4, 1)).astype(np.float32))
        if dones.any():
            i = int(np.flatnonzero(dones)[0])
            assert infos[i]["terminal_observation"].shape == (OBS_DIM,)
            assert "episode" in infos[i]
            return
    pytest.fail("no episode finished")
