"""Batched 6-DOF rocket dynamics.

Frames
------
World: y is up, x/z horizontal.
Body:  +y points from the engine toward the nose. Origin is the geometric center;
       the state position is the center of mass (COM), which sits slightly below
       the geometric center depending on fuel load.
Quaternions are (w, x, y, z) and rotate body vectors into the world frame.
"""

import numpy as np

from .config import G0, RocketSpec


# ---------------------------------------------------------------------- quaternions
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by quaternions q (..., 4). Broadcasts."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = 0.5 * angle[..., None]
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def up_vector(q: np.ndarray) -> np.ndarray:
    """World-frame direction of the body +y axis."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)


def tilt_angle(q: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(up_vector(q)[..., 1], -1.0, 1.0))


# ---------------------------------------------------------------------- rocket
class RocketBatch:
    """State and dynamics for N rockets sharing one RocketSpec."""

    def __init__(self, spec: RocketSpec, n: int):
        self.spec = spec
        self.n = n
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        self.omega = np.zeros((n, 3))      # body frame
        self.fuel = np.full(n, spec.fuel_capacity)
        self.throttle = np.zeros(n)
        self.gimbal = np.zeros((n, 2))     # (δx, δz)

        L, R = spec.length, spec.leg_radius
        foot_y = -L / 2 - spec.leg_drop
        self.leg_points = np.array([[R, foot_y, 0], [-R, foot_y, 0], [0, foot_y, R], [0, foot_y, -R]], float)
        r = spec.radius
        self.body_points = np.array([[r, -L / 2, 0], [-r, -L / 2, 0], [0, -L / 2, r], [0, -L / 2, -r],
                                     [0, L / 2, 0]], float)

    # ------------------------------------------------------------------ derived
    def derived(self):
        """Mass properties from spec + current fuel. Returns a dict of (N,) / (N,3) arrays."""
        s = self.spec
        L, r = s.length, s.radius
        fuel = self.fuel
        mass = s.dry_mass + fuel
        fuel_y = -s.fuel_com_shift * L / 2              # fuel COM in body frame
        com_y = fuel * fuel_y / mass                    # dry COM is at the geometric center
        i_dry = s.dry_mass * (3 * r * r + L * L) / 12.0
        i_fuel = fuel * (3 * r * r + (L / 2) ** 2) / 12.0
        i_side = i_dry + s.dry_mass * com_y ** 2 + i_fuel + fuel * (fuel_y - com_y) ** 2
        i_roll = mass * r * r / 2.0
        inertia = np.stack([i_side, i_roll, i_side], axis=-1)
        lever_arm = L / 2 + com_y                       # COM to engine (engine at -L/2)
        return {"mass": mass, "com_y": com_y, "inertia": inertia, "lever_arm": lever_arm}

    def thrust_to_weight(self, gravity) -> np.ndarray:
        return self.spec.t_max / ((self.spec.dry_mass + self.fuel) * gravity)

    def world_points(self, body_pts: np.ndarray) -> np.ndarray:
        """Body-frame points (K,3), relative to the geometric center, to world (N,K,3)."""
        com_y = self.derived()["com_y"]
        rel = body_pts[None, :, :] - np.stack([np.zeros(self.n), com_y, np.zeros(self.n)], -1)[:, None, :]
        return self.pos[:, None, :] + quat_rotate(self.quat[:, None, :], rel)

    # ------------------------------------------------------------------ dynamics
    def step(self, dt, target_throttle, target_gimbal, roll_cmd, gravity, density, isp_scale, wind,
             active=None):
        """Advance all rockets by dt. Per-env environment arrays are shape (N,) or (N,3).

        Rockets where `active` is False are frozen. Returns fuel burned (N,) in kg.
        """
        s = self.spec
        if active is None:
            active = np.ones(self.n, bool)
        a = active

        # Actuator rate limits.
        dthr = np.clip(target_throttle - self.throttle, -s.throttle_rate * dt, s.throttle_rate * dt)
        self.throttle = np.where(a, np.clip(self.throttle + dthr, 0.0, 1.0), self.throttle)
        tg = np.clip(target_gimbal, -s.max_gimbal, s.max_gimbal)
        dg = np.clip(tg - self.gimbal, -s.gimbal_rate * dt, s.gimbal_rate * dt)
        self.gimbal = np.where(a[:, None], self.gimbal + dg, self.gimbal)

        d = self.derived()
        mass, inertia, arm = d["mass"], d["inertia"], d["lever_arm"]

        # Engine: off below minimum throttle or with no fuel.
        on = (self.throttle >= s.min_throttle) & (self.fuel > 0) & a
        thrust = np.where(on, self.throttle * s.t_max, 0.0)
        mdot = thrust / (s.isp * isp_scale * G0)
        burn = np.minimum(mdot * dt, self.fuel)
        thrust = np.where(mdot > 0, thrust * burn / np.maximum(mdot * dt, 1e-12), 0.0)
        self.fuel = self.fuel - burn

        # Thrust direction in body frame: R_x(δx) R_z(δz) · (0, 1, 0).
        gx, gz = self.gimbal[:, 0], self.gimbal[:, 1]
        dir_b = np.stack([-np.sin(gz), np.cos(gz) * np.cos(gx), np.cos(gz) * np.sin(gx)], -1)
        f_body = thrust[:, None] * dir_b
        engine_pos = np.stack([np.zeros(self.n), -arm, np.zeros(self.n)], -1)
        torque = np.cross(engine_pos, f_body)
        torque[:, 1] += np.clip(roll_cmd, -1, 1) * s.rcs_roll_torque * a

        # Aerodynamic drag (area blends between end-on and broadside).
        v_rel = self.vel - wind
        speed = np.linalg.norm(v_rel, axis=-1)
        up = up_vector(self.quat)
        cos_a = np.abs(np.sum(up * v_rel, -1)) / np.maximum(speed, 1e-9)
        side_area = 2 * s.radius * s.length
        area = s.ref_area * cos_a + side_area * np.sqrt(np.maximum(1 - cos_a ** 2, 0))
        f_drag = -0.5 * density[:, None] * s.cd * area[:, None] * speed[:, None] * v_rel
        # Aerodynamic rotational damping, proportional to dynamic pressure.
        torque -= (0.5 * density * speed * s.cd * side_area * s.length ** 2 / 12)[:, None] * self.omega * 0.1

        f_world = quat_rotate(self.quat, f_body) + f_drag
        acc = f_world / mass[:, None]
        acc[:, 1] -= gravity

        # Semi-implicit Euler.
        self.vel = np.where(a[:, None], self.vel + acc * dt, self.vel)
        self.pos = np.where(a[:, None], self.pos + self.vel * dt, self.pos)

        # Euler's rotation equations, body frame, diagonal inertia.
        w = self.omega
        w_dot = (torque - np.cross(w, inertia * w)) / inertia
        self.omega = np.where(a[:, None], w + w_dot * dt, w)
        omega_q = np.concatenate([np.zeros((self.n, 1)), self.omega], -1)
        q = self.quat + 0.5 * quat_mul(self.quat, omega_q) * dt
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        self.quat = np.where(a[:, None], q, self.quat)
        return np.where(a, burn, 0.0)
