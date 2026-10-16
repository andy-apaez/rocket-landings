"""Vectorized rocket landing environment (pure numpy, N rockets in parallel)."""

import numpy as np

from .config import BODIES, EnvConfig
from .physics import RocketBatch, quat_from_axis_angle, quat_mul, quat_rotate, tilt_angle, up_vector
from .terrain import Terrain

# Episode outcome codes (info["reason"]).
REASONS = [
    "running", "landed", "crash_vspeed", "crash_hspeed", "crash_tilt", "crash_angvel",
    "off_pad", "steep_slope", "uneven_legs", "body_contact", "flight_tilt", "out_of_bounds",
    "fuel_empty", "timeout", "caught", "hit_tower",
]
R = {name: i for i, name in enumerate(REASONS)}

N_RAYS = 8
RAY_RADIUS = 15.0
OBS_DIM = 32
ACT_DIM = 4  # throttle, gimbal δx, gimbal δz, roll; all in [-1, 1]


class VecRocketEnv:
    def __init__(self, cfg: EnvConfig | None = None, num_envs: int | None = None, seed: int | None = None,
                 autoreset: bool = True):
        self.cfg = cfg or EnvConfig()
        self.n = num_envs or self.cfg.sim.num_envs
        self.autoreset = autoreset
        self.rng = np.random.default_rng(seed)
        self.rocket = RocketBatch(self.cfg.rocket, self.n)
        self.terrain = Terrain(self.cfg.terrain, self.n)
        n = self.n
        self.gravity = np.zeros(n)
        self.density = np.zeros(n)
        self.isp_scale = np.ones(n)
        self.wind = np.zeros((n, 3))
        self.time = np.zeros(n)
        self.done = np.zeros(n, bool)
        self.reason = np.zeros(n, int)
        self.potential = np.zeros(n)
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, int)
        self.catch_timer = np.zeros(n)      # seconds continuously inside the catch zone
        self.tower_xz = np.zeros((n, 2))
        self.tower_base = np.zeros(n)
        angles = np.arange(N_RAYS) * 2 * np.pi / N_RAYS
        self.ray_offsets = RAY_RADIUS * np.stack([np.cos(angles), np.sin(angles)], -1)  # (K, 2) in xz
        self.decision_dt = self.cfg.sim.dt * self.cfg.sim.substeps

    observation_dim = OBS_DIM
    action_dim = ACT_DIM

    # Reset
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_idx(np.arange(self.n))
        return self.observe()

    def reset_idx(self, idx: np.ndarray):
        idx = np.asarray(idx)
        m = len(idx)
        if m == 0:
            return
        rng, sim, rk = self.rng, self.cfg.sim, self.rocket

        # Body
        names = rng.choice(sim.bodies, size=m)
        for i, name in zip(idx, names):
            b = BODIES[name]
            self.gravity[i], self.density[i], self.isp_scale[i] = b.gravity, b.air_density, b.isp_scale
            wind = b.wind if sim.wind is None else sim.wind
            gust = b.wind_gust if sim.wind_gust is None else sim.wind_gust
            self.wind[i] = np.asarray(wind) + rng.normal(0, gust, 3) * np.array([1, 0.2, 1])

        # Terrain
        self.terrain.regenerate(idx, rng)
        pad = self.terrain.pad[idx]
        cc = self.cfg.catch
        self.tower_xz[idx] = pad[:, [0, 2]] + cc.tower_distance * np.asarray(cc.tower_direction)
        self.tower_base[idx] = self.terrain.height_at(self.tower_xz[idx][:, None, :], idx)[:, 0]

        # Start
        half = self.cfg.terrain.world_size / 2 - 20.0
        xz = np.clip(pad[:, [0, 2]] + rng.uniform(-sim.start_offset, sim.start_offset, (m, 2)), -half, half)
        ground = self.terrain.height_at(xz[:, None, :], idx)[:, 0]
        alt = rng.uniform(*sim.start_altitude, m)
        rk.pos[idx] = np.stack([xz[:, 0], ground + alt, xz[:, 1]], -1)
        rk.vel[idx] = np.stack([rng.uniform(-sim.start_hvel, sim.start_hvel, m),
                                rng.uniform(*sim.start_vvel, m),
                                rng.uniform(-sim.start_hvel, sim.start_hvel, m)], -1)

        # Attitude: random tilt about a horizontal axis, composed with a random roll about body y
        az = rng.uniform(0, 2 * np.pi, m)
        tilt_axis = np.stack([np.cos(az), np.zeros(m), np.sin(az)], -1)
        q_tilt = quat_from_axis_angle(tilt_axis, rng.uniform(0, sim.start_tilt, m))
        q_roll = quat_from_axis_angle(np.tile([0.0, 1.0, 0.0], (m, 1)), rng.uniform(-np.pi, np.pi, m))
        rk.quat[idx] = quat_mul(q_tilt, q_roll)
        rk.omega[idx] = rng.uniform(-sim.start_angvel, sim.start_angvel, (m, 3))
        rk.fuel[idx] = rng.uniform(*sim.start_fuel, m) * self.cfg.rocket.fuel_capacity
        rk.throttle[idx] = 0.0
        rk.gimbal[idx] = 0.0

        # Episode
        self.time[idx] = 0.0
        self.done[idx] = False
        self.reason[idx] = R["running"]
        self.ep_return[idx] = 0.0
        self.ep_len[idx] = 0
        self.catch_timer[idx] = 0.0
        self.potential[idx] = self._potential()[idx]

    # Helpers
    def _foot_clearance(self):
        """Height of the leg plane above the ground directly below the COM, (N,)."""
        rk = self.rocket
        ground = self.terrain.height_at(rk.pos[:, None, [0, 2]])[:, 0]
        foot_below_com = self.cfg.rocket.length / 2 + self.cfg.rocket.leg_drop + rk.derived()["com_y"]
        return rk.pos[:, 1] - foot_below_com - ground

    # Catch
    def catch_target(self):
        """World position (N,3) where the booster's catch point sits when held by the arms."""
        cc, L = self.cfg.catch, self.cfg.rocket.length
        t = self.terrain.pad.copy()
        t[:, 1] += cc.base_clearance + L - cc.catch_below_top
        return t

    def catch_point(self):
        """World position (N,3) of the catch point on the booster (just under the interstage)."""
        y = self.cfg.rocket.length / 2 - self.cfg.catch.catch_below_top
        return self.rocket.world_points(np.array([[0.0, y, 0.0]]))[:, 0]

    def _hit_tower(self):
        """Does the booster body intersect the tower column? (N,) bool."""
        cc, s, rk = self.cfg.catch, self.cfg.rocket, self.rocket
        ys = np.linspace(-s.length / 2, s.length / 2, 7)
        pts = rk.world_points(np.stack([np.zeros(7), ys, np.zeros(7)], -1))       # (N, 7, 3) along the axis
        rel = pts[..., [0, 2]] - self.tower_xz[:, None, :]
        d = np.asarray(cc.tower_direction)
        lx, lz = rel @ d, rel @ np.array([-d[1], d[0]])
        half = cc.tower_width / 2 + s.radius
        y = pts[..., 1] - self.tower_base[:, None]
        return np.any((np.abs(lx) <= half) & (np.abs(lz) <= half) & (y >= 0) & (y <= cc.tower_height), axis=1)

    def _catch_error(self):
        """(horizontal, vertical) offset of the catch point from the catch target."""
        e = self.catch_point() - self.catch_target()
        return np.linalg.norm(e[:, [0, 2]], axis=-1), e[:, 1]

    # Observation
    def _potential(self):
        rk = self.rocket
        if self.cfg.task == "catch":
            h, dy = self._catch_error()
            speed = np.linalg.norm(rk.vel, axis=-1)
            # Horizontal precision matters most: the arms leave only catch_radius of slack
            return -(h / 30.0 + np.abs(dy) / 20.0 + speed / 30.0 + tilt_angle(rk.quat))
        hdist = np.linalg.norm(rk.pos[:, [0, 2]] - self.terrain.pad[:, [0, 2]], axis=-1)
        clear = np.maximum(self._foot_clearance(), 0.0)
        speed = np.linalg.norm(rk.vel, axis=-1)
        # Clearance gets its own, stronger term so hovering above the pad isn't nearly as good as landing
        return -(hdist / 100.0 + clear / 20.0 + speed / 30.0 + tilt_angle(rk.quat))

    def observe(self) -> np.ndarray:
        rk, s = self.rocket, self.cfg.rocket
        pad = self.terrain.pad
        xz = rk.pos[:, [0, 2]]
        ground, grad = self.terrain._bilinear(xz[:, None, :], None)
        ground, grad = ground[:, 0], grad[:, 0]
        rays = self.terrain.height_at(xz[:, None, :] + self.ray_offsets[None])
        body_x = quat_rotate(rk.quat, np.tile([1.0, 0.0, 0.0], (self.n, 1)))
        rel = rk.pos - pad
        clearance = self._foot_clearance() / 100.0
        if self.cfg.task == "catch":
            # Same layout, catch-relative meaning: the target is the catch point, and the clearance and
            # slope slots carry fine-grained catch error (the booster never touches the ground on purpose).
            e = self.catch_point() - self.catch_target()
            rel, clearance, grad = e, e[:, 1] / 10.0, e[:, [0, 2]] / 10.0
        obs = np.concatenate([
            rel / 100.0,
            rk.vel / 30.0,
            up_vector(rk.quat),
            body_x,
            rk.omega,
            (rk.fuel / s.fuel_capacity)[:, None],
            rk.throttle[:, None],
            rk.gimbal / s.max_gimbal,
            clearance[:, None],
            grad,
            (rays - ground[:, None]) / 10.0,
            (self.gravity / 9.81)[:, None],
            (self.density / 1.225)[:, None],
        ], axis=-1)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # Contact
    def _check_contact(self, alive: np.ndarray):
        """Detect ground contact for alive envs. Returns (contact, landed, soft) bool arrays (N,)."""
        rk, c = self.rocket, self.cfg.criteria
        none = np.zeros(self.n, bool)
        if self.cfg.task == "catch":
            # Legs stay stowed for a catch: any ground contact is a crash
            body = rk.world_points(rk.body_points)
            contact = np.any(body[..., 1] <= self.terrain.height_at(body[..., [0, 2]]), axis=1) & alive
            self.reason = np.where(contact, R["body_contact"], self.reason)
            self._landing_slope = np.zeros(self.n)
            return contact, none, none
        legs = rk.world_points(rk.leg_points)        # (N, 4, 3)
        body = rk.world_points(rk.body_points)       # (N, 5, 3)
        leg_ground = self.terrain.height_at(legs[..., [0, 2]])
        body_ground = self.terrain.height_at(body[..., [0, 2]])
        leg_hit = np.any(legs[..., 1] <= leg_ground, axis=1)
        body_hit = np.any(body[..., 1] <= body_ground, axis=1)
        contact = (leg_hit | body_hit) & alive
        if not contact.any():
            return contact, np.zeros(self.n, bool), np.zeros(self.n, bool)

        _, normal = self.terrain.height_and_normal(rk.pos[:, None, [0, 2]])
        normal = normal[:, 0]
        v_n = np.sum(rk.vel * normal, -1)
        v_t = np.linalg.norm(rk.vel - v_n[:, None] * normal, axis=-1)
        tilt = tilt_angle(rk.quat)
        angvel = np.linalg.norm(rk.omega, axis=-1)
        pad_dist = np.linalg.norm(rk.pos[:, [0, 2]] - self.terrain.pad[:, [0, 2]], axis=-1)
        slope = np.arccos(np.clip(normal[:, 1], -1, 1))
        leg_diff = leg_ground.max(1) - leg_ground.min(1)

        # Criteria: ordered checks, the first failure is the reported reason
        checks = [
            (body_hit, "body_contact"),
            (np.abs(v_n) > c.max_vspeed, "crash_vspeed"),
            (v_t > c.max_hspeed, "crash_hspeed"),
            (tilt > c.max_tilt, "crash_tilt"),
            (angvel > c.max_angvel, "crash_angvel"),
            (slope > c.max_slope, "steep_slope"),
            (leg_diff > c.max_leg_height_diff, "uneven_legs"),
            (pad_dist > c.max_pad_dist, "off_pad"),
        ]
        reason = np.full(self.n, R["landed"])
        for fail, name in reversed(checks):
            reason = np.where(fail, R[name], reason)
        landed = contact & (reason == R["landed"])
        # Soft touchdown that only failed on placement (off pad / slope / uneven ground)
        soft = contact & np.isin(reason, [R["off_pad"], R["steep_slope"], R["uneven_legs"]])
        self.reason = np.where(contact, reason, self.reason)
        self._landing_slope = slope
        return contact, landed, soft

    def _check_catch(self, alive: np.ndarray, dt: float):
        """Catch task: tower collisions, and the hold timer in the catch zone. Returns (caught, crashed)."""
        cc = self.cfg.catch
        rk = self.rocket
        crashed = self._hit_tower() & alive if cc.collide else np.zeros(self.n, bool)
        self.reason[crashed] = R["hit_tower"]
        h, dy = self._catch_error()
        v_h = np.linalg.norm(rk.vel[:, [0, 2]], axis=-1)
        in_zone = ((h <= cc.catch_radius) & (np.abs(dy) <= cc.height_tolerance)
                   & (np.abs(rk.vel[:, 1]) <= cc.max_vspeed) & (v_h <= cc.max_hspeed)
                   & (tilt_angle(rk.quat) <= cc.max_tilt) & (np.linalg.norm(rk.omega, axis=-1) <= cc.max_angvel)
                   & alive & ~crashed)
        self.catch_timer = np.where(in_zone, self.catch_timer + dt, 0.0)
        caught = in_zone & (self.catch_timer >= cc.hold_time - 1e-9)
        self.reason[caught] = R["caught"]
        return caught, crashed

    # Step
    def step(self, actions: np.ndarray):
        """actions: (N, 4) in [-1, 1]. Returns obs, reward, terminated, truncated, info."""
        cfg, rk = self.cfg, self.rocket
        rw, sim, crit = cfg.reward, cfg.sim, cfg.criteria
        # Actions
        a = np.clip(np.asarray(actions, dtype=float).reshape(self.n, ACT_DIM), -1.0, 1.0)
        target_thr = (a[:, 0] + 1.0) / 2.0
        target_gimbal = a[:, 1:3] * cfg.rocket.max_gimbal
        roll = a[:, 3]

        alive = ~self.done
        start_alive = alive.copy()
        reward = np.zeros(self.n)
        terminated = np.zeros(self.n, bool)
        burned = np.zeros(self.n)
        landed_any = np.zeros(self.n, bool)
        impact_speed = np.zeros(self.n)

        # Physics
        for _ in range(sim.substeps):
            burned += rk.step(sim.dt, target_thr, target_gimbal, roll, self.gravity, self.density,
                              self.isp_scale, self.wind, active=alive)
            self.time += sim.dt * alive
            contact, landed, soft = self._check_contact(alive)
            if contact.any():
                speed = np.linalg.norm(rk.vel, axis=-1)
                impact_speed = np.where(contact, speed, impact_speed)
                fuel_frac = rk.fuel / cfg.rocket.fuel_capacity
                bonus = rw.landing_bonus + rw.fuel_bonus * fuel_frac - rw.slope_penalty * self._landing_slope
                crash = -rw.crash_penalty - rw.impact_penalty * speed
                reward += np.where(landed, bonus, np.where(soft, -0.5 * rw.crash_penalty, crash)) * contact
                landed_any |= landed
                terminated |= contact
                alive &= ~contact
            if cfg.task == "catch":
                landed, crashed = self._check_catch(alive, sim.dt)
                speed = np.linalg.norm(rk.vel, axis=-1)
                impact_speed = np.where(crashed, speed, impact_speed)
                fuel_frac = rk.fuel / cfg.rocket.fuel_capacity
                reward += landed * (rw.landing_bonus + rw.fuel_bonus * fuel_frac)
                reward -= crashed * (rw.crash_penalty + rw.impact_penalty * speed)
                landed_any |= landed
                terminated |= landed | crashed
                alive &= ~(landed | crashed)

        # Flight: mid-air terminations
        tilt = tilt_angle(rk.quat)
        half = cfg.terrain.world_size / 2
        oob = (np.abs(rk.pos[:, 0]) > half) | (np.abs(rk.pos[:, 2]) > half) | (rk.pos[:, 1] > sim.world_height)
        flight_fail = [(tilt > crit.flight_tilt_limit, "flight_tilt"), (oob, "out_of_bounds")]
        if sim.terminate_on_fuel_empty:
            flight_fail.append((rk.fuel <= 0, "fuel_empty"))
        for fail, name in flight_fail:
            hit = fail & alive
            self.reason[hit] = R[name]
            reward -= rw.crash_penalty * hit
            terminated |= hit
            alive &= ~hit

        # Timeout: part of the task (hovering forever is a failure), so it terminates
        # with a penalty rather than truncating
        timeout = alive & (self.time >= sim.max_time - 1e-9)
        self.reason[timeout] = R["timeout"]
        reward -= rw.timeout_penalty * timeout
        terminated |= timeout
        truncated = np.zeros(self.n, bool)

        # Reward: per-step penalties and potential-based shaping
        step_dt = self.decision_dt
        target = self.catch_target() if cfg.task == "catch" else self.terrain.pad
        dist = np.linalg.norm(rk.pos - target, axis=-1)
        speed = np.linalg.norm(rk.vel, axis=-1)
        angvel = np.linalg.norm(rk.omega, axis=-1)
        per_step = (rw.distance * dist / 10.0 + rw.speed * speed / 3.0 + rw.tilt * tilt
                    + rw.angvel * angvel + rw.time) * step_dt + rw.fuel * burned / cfg.rocket.fuel_capacity
        new_pot = self._potential()
        shaping = rw.shaping * (new_pot - self.potential)
        self.potential = new_pot
        reward += (shaping - per_step) * start_alive

        # Bookkeeping
        self.ep_return += reward
        self.ep_len += start_alive
        done_now = (terminated | truncated) & start_alive
        self.done |= done_now

        info = {
            "reason": self.reason.copy(),
            "landed": landed_any,
            "impact_speed": impact_speed,
            "done": done_now,
            "episode_return": self.ep_return.copy(),
            "episode_length": self.ep_len.copy(),
        }
        obs = self.observe()
        if self.autoreset and done_now.any():
            info["final_obs"] = obs.copy()
            idx = np.flatnonzero(done_now)
            self.reset_idx(idx)
            obs[idx] = self.observe()[idx]
        return obs, reward.astype(np.float32), terminated & start_alive, truncated & start_alive, info
