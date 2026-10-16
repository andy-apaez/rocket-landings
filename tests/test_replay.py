import json

import numpy as np
import pytest

from rocket_sim import EnvConfig
from rocket_sim.controller import PDController
from rocket_sim.physics import tilt_angle
from rocket_sim.replay import launch_setup, record, save


def test_launch_setup_places_rocket_relative_to_pad():
    from rocket_sim import VecRocketEnv

    env = VecRocketEnv(EnvConfig(), num_envs=3, seed=0, autoreset=False)
    env.reset()
    cond = {"altitude": 150.0, "offset_x": 20.0, "offset_z": -10.0, "descent": 12.0, "drift_x": 3.0,
            "drift_z": -2.0, "tilt_deg": 8.0, "fuel": 0.7}
    launch_setup(cond)(env)
    rk, pad = env.rocket, env.terrain.pad
    assert np.allclose(rk.pos[:, 0] - pad[:, 0], 20.0)
    assert np.allclose(rk.pos[:, 2] - pad[:, 2], -10.0)
    ground = env.terrain.height_at(rk.pos[:, None, [0, 2]])[:, 0]
    assert np.allclose(rk.pos[:, 1] - ground, 150.0)
    assert np.allclose(rk.vel, [3.0, -12.0, -2.0])
    assert np.allclose(tilt_angle(rk.quat), np.radians(8.0))
    assert np.allclose(rk.fuel, 0.7 * env.cfg.rocket.fuel_capacity)


def test_record_single_launch_and_manifest(tmp_path):
    ctl = PDController()
    replay = record(EnvConfig(), lambda env, obs: ctl(env), num_rockets=1, seed=3,
                    setup=launch_setup({"altitude": 140.0, "offset_x": 10.0, "offset_z": 5.0}))
    rk = replay["rockets"][0]
    assert rk["outcome"] == "landed"
    first = rk["frames"][0]
    assert first[0] - replay["meta"]["pad"][0] == pytest.approx(10.0, abs=1e-3)
    save(replay, "a", tmp_path, step=5, extra={"launch": 1, "outcome": rk["outcome"]})
    save(replay, "b", tmp_path, step=5, extra={"launch": 2, "outcome": rk["outcome"]})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [m["launch"] for m in manifest] == [1, 2]


def test_play_session_launch(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    import play

    monkeypatch.setattr(play, "ROOT", tmp_path)
    session = play.Session(tmp_path / "interactive", seed=0, fresh=True)
    replay = session.launch({"pilot": "pd", "body": "mars", "altitude": 9999, "offset_x": 0, "offset_z": 0})
    assert replay["meta"]["body"] == ["mars"]
    assert session.state()["launches"] == 1
    # out-of-range conditions are clamped to what training covers
    start_alt = replay["rockets"][0]["frames"][0][1] - replay["meta"]["pad"][1]
    assert start_alt == pytest.approx(play.LIMITS["altitude"][1], abs=1.0)
    with pytest.raises(ValueError):
        session.launch({"body": "pluto"})


def test_play_migrates_legacy_layout(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    import play

    out = tmp_path / "interactive"
    (out / "replays").mkdir(parents=True)
    (out / "ppo_rocket.zip").write_bytes(b"model")
    (out / "vecnormalize.pkl").write_bytes(b"stats")
    (out / "replays" / "manifest.json").write_text("[]")
    play.Sessions(out, seed=0, fresh=False)
    assert (out / "land" / "ppo_rocket.zip").read_bytes() == b"model"
    assert (out / "land" / "replays" / "manifest.json").exists()
    assert not (out / "ppo_rocket.zip").exists()
