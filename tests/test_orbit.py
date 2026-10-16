import math

import numpy as np
import pytest

from rocket_sim import EnvConfig, VecRocketEnv
from rocket_sim.orbit import (EARTH, LAUNCH_SITES, apply_site, ascent, descent_profile, elements, landing_conditions,
                              plan_landing, propagate, site_conditions)

CAPE = LAUNCH_SITES["cape"][1:]


@pytest.fixture(scope="module")
def cape53():
    return ascent(*CAPE, altitude=300e3, inclination=53.0)


def test_ascent_reaches_requested_circular_orbit(cape53):
    o = cape53["orbit"]
    assert abs(o["apoapsis"] - 300e3) < 20e3 and abs(o["periapsis"] - 300e3) < 20e3
    assert o["inclination"] == pytest.approx(53.0, abs=0.5)
    names = [e["name"] for e in cape53["events"]]
    assert names == ["liftoff", "max_q", "meco", "stage_sep", "ses", "seco1", "circ_burn", "seco2"]
    assert cape53["budget_dv"] > 500   # the orbital stage keeps propellant for the deorbit burn
    F = {f: i for i, f in enumerate(cape53["fields"])}
    peak_q = max(f[F["q"]] for f in cape53["frames"])
    assert 15e3 < peak_q < 35e3        # throttle bucket holds max-Q near the limit


def test_ascent_climbs_vertically_off_the_pad(cape53):
    F = {f: i for i, f in enumerate(cape53["fields"])}
    f = next(f for f in cape53["frames"] if f[F["t"]] >= 10)
    # downrange is measured over the (spinning) ground, so the pad's own 400 m/s doesn't count
    assert f[F["downrange"]] < 0.01 and f[F["alt"]] > 200


def test_inclination_below_launch_latitude_is_raised():
    o = ascent(*CAPE, altitude=300e3, inclination=10.0)["orbit"]
    assert o["inclination"] == pytest.approx(CAPE[0], abs=0.5)


def test_kepler_propagation_round_trip():
    r0 = np.array([7.0e6, 0.0, 0.0])
    v0 = np.array([0.0, 7.2e3, 1.5e3])
    o = elements(r0, v0, 100.0)
    r, v = propagate(o, np.array([100.0, 100.0 + o["period"]]))
    assert np.allclose(r[0], r0, atol=1.0) and np.allclose(v[0], v0, atol=1e-3)
    assert np.allclose(r[1], r0, atol=1.0)          # one period later, back where it started
    energy = lambda r, v: v @ v / 2 - EARTH.mu / np.linalg.norm(r)  # noqa: E731
    r2, v2 = propagate(o, 2500.0)
    assert energy(r2, v2) == pytest.approx(energy(r0, v0), rel=1e-9)


def test_descent_reaches_gate_inside_training_envelope():
    prof = descent_profile(EARTH.radius + 300e3)
    g = prof["gate"]
    assert g["alt"] == pytest.approx(230.0, abs=5.0)
    assert 5.0 <= g["v_down"] <= 30.0           # play.LIMITS["descent"]
    assert 0.6 <= g["fuel"] <= 1.0              # play.LIMITS["fuel"]
    assert 50 < prof["dv_retro"] < 150 and prof["peak_g"] < 12
    names = [e["name"] for e in prof["events"]]
    assert names.index("entry_interface") < names.index("braking_burn") < names.index("gate")


def test_plan_lands_on_the_site(cape53):
    lat, lon = 34.5, -106.0
    p = plan_landing(cape53["orbit"], lat, lon, cape53["duration"], cape53["budget_dv"])
    assert p["reachable"] and p["dv"] <= p["budget_dv"]
    assert p["t_burn"] > cape53["duration"] and p["t_land"] == pytest.approx(p["t_burn"] + p["tof"])
    end = p["track"][-1]
    assert end[1] == pytest.approx(lat, abs=0.05) and math.cos(math.radians(end[2] - lon)) > math.cos(math.radians(0.05))
    assert end[3] == pytest.approx(p["gate"]["alt"], abs=1.0)


def test_far_site_needs_a_plane_change_it_cannot_afford(cape53):
    p = plan_landing(cape53["orbit"], -80.0, 0.0, cape53["duration"], cape53["budget_dv"])
    assert not p["reachable"] and p["dv"] > cape53["budget_dv"]
    # with an unlimited budget the first pass is taken
    q = plan_landing(cape53["orbit"], -80.0, 0.0, cape53["duration"], 1e9)
    assert q["reachable"] and q["t_burn"] <= p["t_burn"] + 1e-6


def test_site_conditions():
    with pytest.raises(ValueError):
        site_conditions(0.0, -30.0, surface="ocean")
    flat, rough = site_conditions(40.0, -100.0, 0.0), site_conditions(40.0, -100.0, 1.0)
    assert rough["amplitude"] > flat["amplitude"] and rough["wind_gust"] > flat["wind_gust"]
    assert site_conditions(40.0, -100.0)["seed"] == site_conditions(40.0, -100.0)["seed"]
    assert site_conditions(45.0, 0.0)["wind"][0] > 0      # mid-latitude westerlies blow toward the east
    assert site_conditions(15.0, 0.0)["wind"][0] < 0      # trade winds blow toward the west


def test_site_sets_terrain_and_wind_in_the_landing_sim():
    site = {**site_conditions(45.0, 10.0, 0.8), "wind_gust": 0.0}
    cfg = apply_site(EnvConfig(), site)
    env = VecRocketEnv(cfg, num_envs=4, seed=0, autoreset=False)
    env.reset()
    assert np.allclose(env.wind, site["wind"])
    h = env.terrain.heights
    assert (h.max(axis=(1, 2)) - h.min(axis=(1, 2))).mean() > 10.0    # rugged site, rugged ground
    apply_site(cfg, None)
    assert cfg.sim.wind is None and cfg.terrain.amplitude == EnvConfig().terrain.amplitude


def test_landing_conditions_start_the_landing_sim_at_the_gate(cape53):
    site = site_conditions(34.5, -106.0, 0.5)
    p = plan_landing(cape53["orbit"], 34.5, -106.0, cape53["duration"], cape53["budget_dv"])
    cond = landing_conditions(p, site)
    assert cond["altitude"] == pytest.approx(p["gate"]["alt"])
    assert cond["descent"] == pytest.approx(p["gate"]["v_down"])
    assert abs(cond["offset_x"]) < 150 and abs(cond["offset_z"]) < 150
    assert cond == landing_conditions(p, site)     # deterministic per site


def test_play_mission_land(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    import play

    monkeypatch.setattr(play, "ROOT", tmp_path)
    asc = play.mission_ascent({"site": "cape", "altitude": 300, "inclination": 53})
    session = play.Session(tmp_path / "interactive", seed=0, fresh=True)
    out = session.mission_land({"orbit": asc["orbit"], "budget_dv": asc["budget_dv"], "t_now": asc["duration"],
                                "lat": 34.5, "lon": -106.0, "ruggedness": 0.3, "pilot": "pd"})
    meta = out["replay"]["meta"]
    assert meta["site"]["lat"] == 34.5 and out["replay"]["rockets"][0]["outcome"] == "landed"
    # the training env goes back to default terrain and wind for a plain launch
    session.launch({"pilot": "pd", "body": "earth"})
    assert session.raw.sim.cfg.sim.wind is None
    with pytest.raises(ValueError):
        session.mission_land({"orbit": asc["orbit"], "budget_dv": asc["budget_dv"], "t_now": asc["duration"],
                              "lat": 0.0, "lon": -30.0, "surface": "ocean"})
