"""Orbital mission around the landing sim: lift-off to orbit, picking a landing site, and the descent to the
hand-off gate where the 6-DOF landing sim (env.py) takes over.

    from rocket_sim.orbit import ascent, plan_landing, site_conditions
    asc = ascent(28.56, -80.58, altitude=300e3, inclination=53.0)
    plan = plan_landing(asc["orbit"], lat=35.0, lon=-106.0, t_now=asc["duration"], budget_dv=asc["budget_dv"])

Frames
------
ECI (planet-centered inertial): z is the spin axis, x points at longitude 0 at t = 0 (liftoff). The planet
spins at ω, so a point at longitude λ sits at ECI angle λ + ω·t. Latitudes are geocentric.

Physics is point-mass: inverse-square gravity, an exponential atmosphere, drag, and staged thrust with
Tsiolkovsky mass flow. On ascent the atmosphere co-rotates with the planet. The descent profile is planar
and ignores planet rotation (the lander ends up almost hovering, so the error is a few km of ground track).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .config import G0, RocketSpec, TerrainConfig


@dataclass
class Planet:
    name: str = "earth"
    radius: float = 6_371_000.0       # m
    mu: float = 3.986004418e14        # m³/s²
    omega: float = 7.2921159e-5       # rad/s, sidereal spin
    rho0: float = 1.225               # kg/m³ at the surface
    scale_height: float = 8_500.0     # m


EARTH = Planet()


@dataclass
class Stage:
    name: str
    dry_mass: float                   # kg
    prop_mass: float                  # kg
    thrust_sl: float                  # N
    thrust_vac: float                 # N
    isp_sl: float                     # s
    isp_vac: float                    # s


@dataclass
class LaunchVehicle:
    """Two-stage launcher with the lander (RocketSpec, full tanks) as its payload."""
    stages: tuple[Stage, ...] = (
        Stage("Booster", 25_600.0, 395_700.0, 7_607_000.0, 8_227_000.0, 282.0, 311.0),
        Stage("Orbital stage", 4_000.0, 70_000.0, 981_000.0, 981_000.0, 348.0, 348.0),
    )
    payload: RocketSpec = field(default_factory=RocketSpec)
    diameter: float = 3.7             # m
    cd: float = 0.45
    max_q: float = 30_000.0           # Pa, the booster throttles down to stay near this
    min_throttle: float = 0.6
    kick_speed: float = 60.0          # m/s, relative speed at the pitch kick
    kick_angle: float = 0.09          # rad from vertical, then a gravity turn
    insertion_tau: float = 60.0       # s, time constant of the orbital stage's altitude hold
    insertion_alt: float = 190e3      # m, altitude held while burning to raise apoapsis to the target

    @property
    def payload_mass(self) -> float:
        return self.payload.dry_mass + self.payload.fuel_capacity


LAUNCH_SITES = {
    "cape": ("Cape Canaveral, USA", 28.56, -80.58),
    "vandenberg": ("Vandenberg, USA", 34.63, -120.61),
    "starbase": ("Starbase, Texas", 25.99, -97.15),
    "kourou": ("Kourou, French Guiana", 5.24, -52.77),
    "baikonur": ("Baikonur, Kazakhstan", 45.96, 63.31),
    "tanegashima": ("Tanegashima, Japan", 30.40, 130.97),
    "mahia": ("Mahia, New Zealand", -39.26, 177.86),
}

MIN_ORBIT_ALT, MAX_ORBIT_ALT = 240e3, 1200e3   # m

ASCENT_FIELDS = ["t", "lat", "lon", "alt", "speed", "v_rel", "fpa", "q", "accel_g", "mass", "throttle", "stage",
                 "downrange", "pitch"]
DESCENT_FIELDS = ["t", "lat", "lon", "alt", "speed", "fpa", "q", "accel_g", "heat", "fuel", "phase"]
DESCENT_PHASES = ["coast", "entry", "braking"]


# Frames
def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _unit(v):
    return v / math.sqrt(float(v @ v))


def lat_lon_to_unit(lat, lon):
    """Geocentric lat/lon (radians, arrays OK) to a unit vector in the planet-fixed frame."""
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], -1)


def site_eci(lat_deg, lon_deg, t, planet: Planet = EARTH):
    """Unit ECI direction (..., 3) of a surface site at times t."""
    t = np.asarray(t, float)
    return lat_lon_to_unit(np.radians(lat_deg) + 0 * t, np.radians(lon_deg) + planet.omega * t)


def eci_to_lat_lon(r, t, planet: Planet = EARTH):
    """ECI positions (..., 3) at times t to planet-fixed (lat, lon) in degrees, lon wrapped to [-180, 180)."""
    r = np.asarray(r, float)
    rn = np.linalg.norm(r, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(r[..., 2] / rn, -1, 1)))
    lon = np.degrees(np.arctan2(r[..., 1], r[..., 0]) - planet.omega * np.asarray(t, float))
    return lat, (lon + 180.0) % 360.0 - 180.0


# Kepler
def elements(r, v, t, planet: Planet = EARTH) -> dict:
    """Classical elements of the orbit through state (r, v) at time t, as a JSON-ready dict."""
    mu, R = planet.mu, planet.radius
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn = math.sqrt(r @ r)
    h = _cross(r, v)
    e_vec = _cross(v, h) / mu - r / rn
    e = math.sqrt(e_vec @ e_vec)
    a = 1.0 / (2.0 / rn - (v @ v) / mu)
    h_hat = _unit(h)
    p_hat = e_vec / e if e > 1e-9 else r / rn
    q_hat = _cross(h_hat, p_hat)
    nu = math.atan2(r @ q_hat, r @ p_hat)
    E = 2.0 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    n = math.sqrt(mu / a ** 3)
    return {
        "a": a, "e": e, "n": n, "M0": E - e * math.sin(E), "epoch": float(t),
        "p": p_hat.tolist(), "q": q_hat.tolist(), "h": h_hat.tolist(),
        "inclination": math.degrees(math.acos(np.clip(h_hat[2], -1, 1))),
        "period": 2 * math.pi / n,
        "apoapsis": a * (1 + e) - R, "periapsis": a * (1 - e) - R,
    }


def propagate(orbit: dict, t):
    """Two-body positions and velocities (..., 3) at times t."""
    t = np.asarray(t, float)
    a, e, n = orbit["a"], orbit["e"], orbit["n"]
    M = orbit["M0"] + n * (t - orbit["epoch"])
    E = M.copy()
    for _ in range(10):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    cE, sE, b = np.cos(E), np.sin(E), math.sqrt(1 - e * e)
    P, Q = np.asarray(orbit["p"]), np.asarray(orbit["q"])
    Edot = n / (1 - e * cE)
    r = (a * (cE - e))[..., None] * P + (a * b * sE)[..., None] * Q
    v = (-a * sE * Edot)[..., None] * P + (a * b * cE * Edot)[..., None] * Q
    return r, v


# Ascent
def ascent(lat: float, lon: float, altitude: float = 300e3, inclination: float | None = None,
           vehicle: LaunchVehicle | None = None, planet: Planet = EARTH, dt: float = 0.2,
           max_time: float = 1500.0) -> dict:
    """Fly the launcher from (lat, lon) into an orbit `altitude` m up, with the given inclination (deg).

    The booster flies a gravity turn after a pitch kick, throttling down through max-Q. After staging,
    the orbital stage holds a low insertion altitude (radial PD) while building horizontal speed in the
    target plane, cuts off once apoapsis reaches the target (SECO-1), coasts to apoapsis and circularizes.
    Inclinations below |lat| can't be flown directly, so they are raised to |lat|, and the altitude is
    clamped to [MIN_ORBIT_ALT, MAX_ORBIT_ALT].
    """
    veh, P = vehicle or LaunchVehicle(), planet
    R, mu, w = P.radius, P.mu, np.array([0.0, 0.0, P.omega])
    altitude = float(np.clip(altitude, MIN_ORBIT_ALT, MAX_ORBIT_ALT))
    la, lo = math.radians(lat), math.radians(lon)
    inc = math.radians(abs(lat) if inclination is None else inclination)
    inc = min(max(inc, abs(la) + 1e-6), math.pi - abs(la) - 1e-6)
    r_hat = lat_lon_to_unit(la, lo)
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    az = math.asin(np.clip(math.cos(inc) / math.cos(la), -1.0, 1.0))      # inertial launch azimuth
    n_hat = _cross(r_hat, math.cos(az) * north + math.sin(az) * east)     # target orbit normal
    r = R * r_hat
    v = _cross(w, r)                                                     # sitting on the spinning pad
    target_r = R + altitude
    insert_r = R + min(veh.insertion_alt, altitude - 60e3)              # perigee of the transfer orbit
    area = math.pi * (veh.diameter / 2) ** 2
    stages = veh.stages
    prop = [s.prop_mass for s in stages]
    k = 0                      # active stage
    phase = "vertical"         # vertical -> kick -> turn -> coast (staging) -> insertion -> circularize
    coast_until = 0.0
    events = [{"name": "liftoff", "t": 0.0}]
    frames, t, next_frame, q_peak, coast_time = [], 0.0, 0.0, (0.0, 0.0), 0.0
    r0_hat = r_hat.copy()

    def yaw():
        return _unit(300.0 * e_h - ((v @ n_hat) + (r @ n_hat) / 60.0) * n_hat)

    def mass():
        return veh.payload_mass + sum(s.dry_mass for s in stages[k:]) + sum(prop[k:])

    while True:
        rn = math.sqrt(r @ r)
        rh = r / rn
        h = rn - R
        e_h = _unit(_cross(n_hat, rh))                                   # downrange, horizontal
        v_rel = v - _cross(w, r)
        vrel = math.sqrt(v_rel @ v_rel)
        rho = P.rho0 * math.exp(-max(h, 0.0) / P.scale_height)
        q = 0.5 * rho * vrel * vrel
        m = mass()
        st = stages[k]
        burning = phase not in ("coast", "orbit") and prop[k] > 0

        # Throttle: the booster rides the max-Q bucket, the orbital stage runs flat out
        thr = 0.0
        if burning:
            thr = 1.0 if k > 0 else float(np.clip(1.0 - (q - 0.85 * veh.max_q) / (0.5 * veh.max_q),
                                                   veh.min_throttle, 1.0))
        pr = rho / P.rho0
        thrust = thr * (st.thrust_vac - (st.thrust_vac - st.thrust_sl) * pr)
        isp = st.isp_vac - (st.isp_vac - st.isp_sl) * pr

        # Steering
        if phase == "vertical":
            d = rh
            if vrel >= veh.kick_speed:
                phase = "kick"
        if phase == "kick":
            d = math.cos(veh.kick_angle) * rh + math.sin(veh.kick_angle) * e_h
            if math.atan2(v_rel @ e_h, v_rel @ rh) >= veh.kick_angle:
                phase = "turn"
        if phase == "turn":   # gravity turn: thrust along the in-plane relative velocity
            th = math.atan2(v_rel @ e_h, v_rel @ rh)
            d = math.cos(th) * rh + math.sin(th) * e_h
        if phase == "insertion":
            v_r = v @ rh
            v_hor = v - v_r * rh
            tau = veh.insertion_tau
            acc_r = (insert_r - rn) / tau ** 2 - 2.0 * v_r / tau + mu / rn ** 2 - (v_hor @ v_hor) / rn
            a_t = thrust / m
            s = float(np.clip(acc_r / max(a_t, 1e-6), -0.6, 0.9))
            # horizontal: downrange in the target plane, yawing to null out-of-plane speed and position
            d = s * rh + math.sqrt(1 - s * s) * yaw()
        if phase == "circularize":   # prograde along the horizon at apoapsis
            d = yaw()
        if phase in ("coast", "orbit"):
            d = rh

        # Forces
        acc = -mu / rn ** 3 * r
        if vrel > 1e-6:
            acc = acc - (0.5 * rho * veh.cd * area * vrel / m) * v_rel
        if thrust > 0:
            acc = acc + (thrust / m) * d
        accel_g = math.sqrt(float((acc + mu / rn ** 3 * r) @ (acc + mu / rn ** 3 * r))) / G0

        # Record
        if t >= next_frame - 1e-9:
            frames.append(_ascent_frame(t, r, v, v_rel, rh, e_h, q, accel_g, m, thr, k, r0_hat, R, d, P))
            next_frame += 1.0
        if q > q_peak[0]:
            q_peak = (q, t)

        # Integrate: semi-implicit Euler, with fine steps close to each cutoff
        step = dt
        apo, peri = _apsides(r, v, mu) if phase == "insertion" else (0.0, 0.0)
        if (phase == "insertion" and apo > target_r - 40e3 and peri > insert_r - 80e3) or phase == "circularize":
            step = dt / 20
        v = v + acc * step
        r = r + v * step
        if thrust > 0:
            prop[k] = max(prop[k] - thrust / (isp * G0) * step, 0.0)
        t += step

        # Staging
        if phase in ("kick", "turn") and prop[k] <= 0:
            events.append({"name": "meco", "t": t})
            events.append({"name": "stage_sep", "t": t + 1.0})
            k += 1
            phase, coast_until = "coast", t + 3.0
        elif phase == "coast" and t >= coast_until:
            events.append({"name": "ses", "t": t})
            phase = "insertion"
        elif phase == "insertion" and ((apo >= target_r and peri >= insert_r - 50e3) or prop[k] <= 0):
            # SECO-1 (only once actually in orbit: a lofted booster arc can already reach past the target
            # apoapsis on a suborbital path), then coast, stepped analytically, up to apoapsis
            events.append({"name": "seco1", "t": t})
            el = elements(r, v, t, P)
            t_apo = t + ((math.pi - el["M0"]) % (2 * math.pi)) / el["n"]
            for tc in np.arange(math.ceil(t / 10.0) * 10.0, t_apo, 10.0):
                rc, vc = propagate(el, tc)
                rch = _unit(rc)
                frames.append(_ascent_frame(tc, rc, vc, vc - _cross(w, rc), rch, _unit(_cross(n_hat, rch)), 0.0, 0.0,
                                            m, 0.0, k, r0_hat, R, rch, P))
            coast_time += t_apo - t
            r, v = propagate(el, t_apo)
            t, next_frame = t_apo, t_apo
            events.append({"name": "circ_burn", "t": t})
            phase = "circularize"
        elif phase == "circularize":
            rn = math.sqrt(r @ r)
            v_r = v @ (r / rn)
            if (v @ v) - v_r * v_r >= mu / rn or prop[k] <= 0:
                events.append({"name": "seco2", "t": t})
                break
        if h < -1.0 or t - coast_time > max_time:
            raise RuntimeError(f"ascent failed in phase {phase} at t={t:.0f}s, h={h / 1e3:.1f} km")

    # Orbit
    events.append({"name": "max_q", "t": q_peak[1]})
    events.sort(key=lambda e: e["t"])
    rh = r / math.sqrt(r @ r)
    frames.append(_ascent_frame(t, r, v, v - _cross(w, r), rh, _unit(_cross(n_hat, rh)), 0.0, 0.0, mass(), 0.0,
                                k, r0_hat, R, rh, P))
    orbit = elements(r, v, t, P)
    if orbit["periapsis"] < 120e3:
        raise RuntimeError(f"orbit periapsis too low ({orbit['periapsis'] / 1e3:.0f} km)")
    # Budget: what the orbital stage has left, with the lander still attached, is the deorbit Δv
    st = stages[-1]
    m_end = mass()
    budget = st.isp_vac * G0 * math.log(m_end / (m_end - prop[-1])) if prop[-1] > 0 else 0.0
    return {
        "site": {"lat": lat, "lon": lon},
        "fields": ASCENT_FIELDS,
        "frames": [[round(x, 4) for x in f] for f in frames],
        "events": events,
        "orbit": orbit,
        "duration": t,
        "budget_dv": budget,
        "prop_left": prop[-1],
        "vehicle": vehicle_stats(veh),
        "requested": {"altitude": altitude, "inclination": None if inclination is None else float(inclination)},
    }


def _apsides(r, v, mu):
    """(apoapsis, periapsis) radii of the orbit through (r, v)."""
    rn = math.sqrt(r @ r)
    a = 1.0 / (2.0 / rn - (v @ v) / mu)
    h = _cross(r, v)
    e = math.sqrt(max(1.0 - (h @ h) / (mu * a), 0.0))
    return a * (1 + e), a * (1 - e)


def _ascent_frame(t, r, v, v_rel, rh, e_h, q, accel_g, m, thr, k, r0_hat, R, d, planet):
    lat, lon = eci_to_lat_lon(r, t, planet)
    speed = math.sqrt(v @ v)
    vr = math.sqrt(v_rel @ v_rel)
    fpa = math.degrees(math.asin(np.clip((v_rel @ rh) / vr, -1, 1))) if vr > 1e-6 else 90.0
    # ground distance from the pad, so in the planet-fixed frame (undo the spin since liftoff)
    c, s_ = math.cos(planet.omega * t), math.sin(planet.omega * t)
    fixed = np.array([c * rh[0] + s_ * rh[1], -s_ * rh[0] + c * rh[1], rh[2]])
    downrange = R * math.acos(np.clip(fixed @ r0_hat, -1, 1)) / 1e3
    pitch = math.degrees(math.atan2(d @ rh, d @ e_h))        # thrust elevation above the horizon
    return [t, float(lat), float(lon), math.sqrt(r @ r) - R, speed, vr, fpa, q, accel_g, m, thr, k, downrange, pitch]


def vehicle_stats(veh: LaunchVehicle) -> dict:
    total = veh.payload_mass + sum(s.dry_mass + s.prop_mass for s in veh.stages)
    stages, above = [], veh.payload_mass
    for s in reversed(veh.stages):
        m0 = above + s.dry_mass + s.prop_mass
        stages.append({"name": s.name, "dry_mass": s.dry_mass, "prop_mass": s.prop_mass, "thrust_vac": s.thrust_vac,
                       "thrust_sl": s.thrust_sl, "isp_vac": s.isp_vac, "isp_sl": s.isp_sl,
                       "dv": s.isp_vac * G0 * math.log(m0 / (m0 - s.prop_mass))})
        above = m0
    p = veh.payload
    return {"liftoff_mass": total, "liftoff_twr": veh.stages[0].thrust_sl / (total * G0), "stages": stages[::-1],
            "lander": {"dry_mass": p.dry_mass, "fuel": p.fuel_capacity, "thrust": p.t_max, "isp": p.isp,
                       "length": p.length, "radius": p.radius, "leg_radius": p.leg_radius, "leg_drop": p.leg_drop,
                       "max_gimbal": p.max_gimbal}}


# Descent
def descent_profile(r0: float, spec: RocketSpec | None = None, planet: Planet = EARTH, perigee_alt: float = 30e3,
                    gate_alt: float = 230.0, gate_speed: float = 15.0, ignite: float = 0.4) -> dict:
    """Planar deorbit, coast, ballistic entry and braking burn from a circular orbit of radius r0.

    The orbital stage lowers periapsis to perigee_alt and separates. The lander falls tail-first (drag area
    cd·ref_area), then lights its own engine late enough to reach `gate_speed` at `gate_alt`, the start of
    the 6-DOF landing sim; it lights once the burn needs `ignite` × full thrust beyond what drag provides.
    Returns sampled arrays plus the deorbit Δv, time of flight and downrange angle.
    """
    s, P = spec or RocketSpec(), planet
    R, mu = P.radius, P.mu
    v_c = math.sqrt(mu / r0)
    rp = R + perigee_alt
    v1 = math.sqrt(2 * mu * rp / (r0 * (r0 + rp)))
    r, th, vr, vt = r0, 0.0, 0.0, v1
    fuel = s.fuel_capacity
    cda = s.cd * s.ref_area
    t, phase, burning = 0.0, 0, False
    out = {k: [] for k in ("t", "theta", "alt", "speed", "fpa", "q", "accel_g", "heat", "fuel", "phase")}
    events = [{"name": "deorbit", "t": 0.0}]
    peak_heat = peak_g = (0.0, 0.0)
    next_rec = 0.0
    while True:
        h = r - R
        rho = P.rho0 * math.exp(-max(h, 0.0) / P.scale_height) if h < 200e3 else 0.0
        sp = math.hypot(vr, vt)
        m = s.dry_mass + fuel
        q = 0.5 * rho * sp * sp
        g = mu / (r * r)
        a_drag = q * cda / m
        heat = 1.7415e-4 * math.sqrt(rho / 1.0) * sp ** 3       # Sutton-Graves, 1 m nose radius, W/m²
        if phase == 0 and h < 120e3:
            phase = 1
            events.append({"name": "entry_interface", "t": t})
        # Braking: deceleration needed to hit the gate (mostly vertical fall) beyond what drag gives
        thrust = 0.0
        if phase >= 1 and h < 20e3 and fuel > 0:
            a_req = (sp * sp - gate_speed ** 2) / (2 * max(h - gate_alt, 1.0)) + g - a_drag
            a_max = s.t_max / m
            if not burning and a_req >= ignite * a_max:
                burning, phase = True, 2
                events.append({"name": "braking_burn", "t": t})
            if burning:
                thrust = s.t_max * float(np.clip(m * a_req / s.t_max, s.min_throttle, 1.0))
        a_thr = thrust / m
        # drag and retro thrust both act along -velocity
        ar = -(a_drag + a_thr) * vr / max(sp, 1e-9)
        at = -(a_drag + a_thr) * vt / max(sp, 1e-9)
        # Record
        if t >= next_rec - 1e-9:
            for k, val in zip(out, (t, th, h, sp, math.degrees(math.atan2(vr, vt)), q, (a_drag + a_thr) / G0,
                                    heat, fuel / s.fuel_capacity, phase)):
                out[k].append(val)
            next_rec = t + (10.0 if phase == 0 else 1.0 if h > 3e3 else 0.25)
        if heat > peak_heat[0]:
            peak_heat = (heat, t)
        if (a_drag + a_thr) > peak_g[0]:
            peak_g = (a_drag + a_thr, t)
        if burning and h <= gate_alt:
            break
        if h < 0:
            raise RuntimeError("descent hit the ground before reaching the gate")

        dt = 2.0 if phase == 0 else 0.25 if h > 5e3 else 0.02
        # Integrate: polar equations of motion, semi-implicit
        vr += (vt * vt / r - g + ar) * dt
        vt += (-vr * vt / r + at) * dt
        r += vr * dt
        th += vt / r * dt
        if thrust > 0:
            fuel = max(fuel - thrust / (s.isp * G0) * dt, 0.0)
        t += dt

    for k, val in zip(out, (t, th, r - R, math.hypot(vr, vt), math.degrees(math.atan2(vr, vt)), q, 0.0, 0.0,
                            fuel / s.fuel_capacity, 2)):
        out[k].append(val)
    events.append({"name": "max_heating", "t": peak_heat[1]})
    events.append({"name": "peak_decel", "t": peak_g[1]})
    events.append({"name": "gate", "t": t})
    events.sort(key=lambda e: e["t"])
    return {
        "arrays": {k: np.asarray(v) for k, v in out.items()},
        "events": events,
        "dv_retro": v_c - v1, "v_after": v1, "tof": t, "dtheta": th,
        "gate": {"alt": r - R, "v_down": -vr, "v_hor": abs(vt), "fuel": fuel / s.fuel_capacity},
        "peak_heat": peak_heat[0], "peak_g": peak_g[0] / G0,
    }


_PROFILE_CACHE: dict = {}


def _profile(r0, spec, planet):
    key = (round(r0 / 100.0), id(planet), spec.dry_mass, spec.fuel_capacity, spec.t_max, spec.cd, spec.isp)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = descent_profile(r0, spec, planet)
    return _PROFILE_CACHE[key]


def _travel_angle(rh, s, h_hat):
    """Angle travelled from rh to s along the great circle through both, in the direction of motion h_hat.

    Also returns the new orbit normal for that great circle. Works on (..., 3) arrays.
    """
    c = np.cross(rh, s)
    cn = np.linalg.norm(c, axis=-1, keepdims=True)
    n_new = c / np.maximum(cn, 1e-12)
    sign = np.where(np.sum(n_new * h_hat, -1, keepdims=True) >= 0, 1.0, -1.0)
    n_new = n_new * sign
    ang = np.arctan2(np.sum(c * n_new, -1), np.sum(rh * s, -1)) % (2 * np.pi)
    return ang, n_new


def plan_landing(orbit: dict, lat: float, lon: float, t_now: float, budget_dv: float, spec: RocketSpec | None = None,
                 planet: Planet = EARTH, horizon: float = 2 * 86400.0, earliest: float = 60.0) -> dict:
    """Find the first deorbit burn that brings the lander down on (lat, lon) within the Δv budget.

    The descent covers a fixed downrange angle Δθ in time T (descent_profile). A burn at time t works if the
    site, where it will be at t + T, lies Δθ ahead along a great circle through the burn point. Any
    cross-track offset is fixed by turning the orbit plane about the burn point in the same burn, which is
    what makes far-off sites expensive. Candidates are scanned every 20 s and refined by bisection.
    """
    spec, P = spec or RocketSpec(), planet
    prof = _profile(orbit["a"], spec, P)
    tof, dth, v1 = prof["tof"], prof["dtheta"], prof["v_after"]
    h_hat = np.asarray(orbit["h"])

    def miss(t):
        r, _ = propagate(orbit, t)
        rh = r / np.linalg.norm(r, axis=-1, keepdims=True)
        ang, _ = _travel_angle(rh, site_eci(lat, lon, t + tof, P), h_hat)
        return (ang - dth + np.pi) % (2 * np.pi) - np.pi

    def evaluate(t):
        r, v = propagate(orbit, t)
        rh = _unit(r)
        s = site_eci(lat, lon, t + tof, P)
        _, n_new = _travel_angle(rh, s, h_hat)
        v_new = v1 * _cross(n_new, rh)
        dv = float(np.linalg.norm(v_new - v))
        plane = math.degrees(math.acos(np.clip(n_new @ h_hat, -1, 1)))
        cross_track = math.asin(np.clip(s @ h_hat, -1, 1)) * P.radius
        return {"t_burn": float(t), "dv": dv, "plane_change": plane, "cross_track": cross_track,
                "r": r, "n_new": n_new}

    # Search
    ts = t_now + earliest + np.arange(0.0, horizon, 20.0)
    f = miss(ts)
    # the site comes into the right downrange slot when the miss angle passes from + to - (smoothly)
    idx = np.flatnonzero((f[:-1] > 0) & (f[1:] <= 0) & (f[:-1] - f[1:] < 0.5))
    best = None
    for i in idx:
        lo_t, hi_t = ts[i], ts[i + 1]
        for _ in range(40):
            mid = 0.5 * (lo_t + hi_t)
            if miss(np.array([mid]))[0] > 0:
                lo_t = mid
            else:
                hi_t = mid
        cand = evaluate(0.5 * (lo_t + hi_t))
        if best is None or cand["dv"] < best["dv"]:
            best = cand
        if cand["dv"] <= budget_dv:
            best = cand
            break
    if best is None:
        raise RuntimeError("no deorbit opportunity found")
    # Plan
    reachable = best["dv"] <= budget_dv
    plan = {
        "lat": lat, "lon": lon, "reachable": reachable,
        "t_now": float(t_now), "t_burn": best["t_burn"], "t_land": best["t_burn"] + tof,
        "wait": best["t_burn"] - t_now, "revs": (best["t_burn"] - t_now) / orbit["period"],
        "dv": best["dv"], "dv_retro": prof["dv_retro"], "budget_dv": budget_dv,
        "plane_change": best["plane_change"], "cross_track": best["cross_track"],
        "tof": tof, "peak_g": prof["peak_g"], "peak_heat": prof["peak_heat"],
        "gate": prof["gate"], "events": [{**e, "t": e["t"] + best["t_burn"]} for e in prof["events"]],
    }
    plan["fields"] = DESCENT_FIELDS
    plan["track"] = _descent_track(prof, best["t_burn"], _unit(best["r"]), best["n_new"], P)
    return plan


def _descent_track(prof, t_burn, rh, n_new, planet, max_points=500):
    """Planet-fixed frames of the descent, from the burn to the gate."""
    A = prof["arrays"]
    k = np.unique(np.linspace(0, len(A["t"]) - 1, min(max_points, len(A["t"]))).round().astype(int))
    th, alt = A["theta"][k], A["alt"][k]
    tdir = _cross(n_new, rh)
    pos = (np.cos(th)[:, None] * rh + np.sin(th)[:, None] * tdir) * (planet.radius + alt)[:, None]
    t = t_burn + A["t"][k]
    lat, lon = eci_to_lat_lon(pos, t, planet)
    cols = [t, lat, lon, alt, A["speed"][k], A["fpa"][k], A["q"][k], A["accel_g"][k], A["heat"][k], A["fuel"][k],
            A["phase"][k]]
    return np.round(np.stack(cols, -1), 4).tolist()


# Site
# Mean surface wind by |latitude|: trade easterlies, mid-latitude westerlies, polar easterlies (m/s).
_WIND_LAT = [0, 15, 30, 45, 60, 75, 90]
_WIND_EAST = [-3.0, -4.5, 0.0, 5.5, 3.0, -2.5, -1.0]
_WIND_POLEWARD = [0.0, -1.5, 0.0, 1.0, 0.0, -1.0, 0.0]


def site_conditions(lat: float, lon: float, ruggedness: float = 0.3, surface: str = "land",
                    color: list | None = None) -> dict:
    """Landing-world settings for a site picked on the globe.

    ruggedness (0 flat .. 1 mountains) sets the terrain amplitude, latitude sets the prevailing wind, and the
    site itself seeds the terrain so the same spot always gets the same ground. Landing-world axes:
    +x east, +y up, +z south.
    """
    if surface == "ocean":
        raise ValueError("that site is open ocean; pick a spot on land")
    rug = float(np.clip(ruggedness, 0.0, 1.0))
    a = abs(lat)
    east = float(np.interp(a, _WIND_LAT, _WIND_EAST))
    north = float(np.interp(a, _WIND_LAT, _WIND_POLEWARD)) * (1.0 if lat >= 0 else -1.0)
    seed = (int(round((lat + 90) * 100)) * 36_001 + int(round((lon + 180) * 100))) % (2 ** 31)
    return {
        "lat": lat, "lon": lon, "surface": surface, "ruggedness": rug, "color": color,
        "amplitude": 2.0 + 12.0 * rug,
        "wind": [east, 0.0, -north],
        "wind_gust": 0.5 + 2.0 * rug,
        "seed": seed,
    }


def apply_site(cfg, site: dict | None):
    """Set terrain and wind in an EnvConfig for a landing site (or back to the defaults for None)."""
    cfg.terrain.amplitude = site["amplitude"] if site else TerrainConfig().amplitude
    cfg.sim.wind = tuple(site["wind"]) if site else None
    cfg.sim.wind_gust = site["wind_gust"] if site else None
    return cfg


def landing_conditions(plan: dict, site: dict) -> dict:
    """Starting conditions for the 6-DOF landing (keys of replay.launch_setup) from the descent gate.

    The gate is reached right above the site; navigation dispersion (seeded by the site) puts the pad a few
    tens of meters off, and the leftover horizontal speed drifts along the prevailing wind.
    """
    g = plan["gate"]
    rng = np.random.default_rng(site["seed"])
    off = rng.normal(0.0, 25.0, 2)
    wind = np.asarray(site["wind"])[[0, 2]]
    wn = np.linalg.norm(wind)
    drift = (wind / wn if wn > 1e-6 else np.zeros(2)) * g["v_hor"] + rng.normal(0.0, 1.0, 2)
    return {
        "altitude": g["alt"], "descent": g["v_down"],
        "offset_x": float(off[0]), "offset_z": float(off[1]),
        "drift_x": float(drift[0]), "drift_z": float(drift[1]),
        "tilt_deg": float(min(math.degrees(math.atan2(g["v_hor"], max(g["v_down"], 1e-6))) + 2.0, 12.0)),
        "fuel": g["fuel"],
    }
