"""Hand-written guidance + attitude controller. Baseline for RL and a physics sanity check.

It reads the true simulator state (not the normalized observation).
"""

import numpy as np

from .env import VecRocketEnv
from .physics import quat_rotate, up_vector


def quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


class PDController:
    def __init__(self, k_close=1.5, k_vel=0.4, max_tilt=0.25, kp_att=3.0, kd_att=3.0, k_roll=2.0,
                 hold_alt=12.0):
        self.k_close, self.k_vel, self.max_tilt, self.hold_alt = k_close, k_vel, max_tilt, hold_alt
        self.kp_att, self.kd_att, self.k_roll = kp_att, kd_att, k_roll

    def _landing_guidance(self, env, mass, g):
        """Desired horizontal (N,2) and vertical (N,) acceleration for a powered landing on legs."""
        rk, spec = env.rocket, env.cfg.rocket
        clear = np.maximum(env._foot_clearance(), 0.0)

        # Horizontal: fly at the velocity that closes the pad offset in the estimated time-to-go,
        # and null horizontal velocity entirely during the last few meters.
        err = env.terrain.pad[:, [0, 2]] - rk.pos[:, [0, 2]]
        v_h = rk.vel[:, [0, 2]]
        t_go = clear / np.maximum(-rk.vel[:, 1], 1.0) + 3.0
        v_h_des = self.k_close * err / t_go[:, None]
        v_h_des = np.where((clear < 3.0)[:, None], 0.0, v_h_des)
        a_h = self.k_vel * (v_h_des - v_h)

        # Vertical: track a descent-rate profile, plus feedforward for the stopping distance.
        v_y = rk.vel[:, 1]
        v_land, margin = 1.0, 3.0  # reach v_land `margin` meters above the ground
        h = np.maximum(clear - margin, 0.0)
        a_brake = 0.3 * np.maximum(spec.t_max / mass - g, 0.5)  # planned decel: a share of thrust margin
        v_des = -(v_land + np.sqrt(2 * a_brake * h))
        # Hold altitude near the ground until over the pad with low horizontal speed.
        unsettled = (np.linalg.norm(err, axis=-1) > 2.0) | (np.linalg.norm(v_h, axis=-1) > 0.5)
        hold = unsettled & (clear < self.hold_alt)
        v_des = np.where(hold, 0.3 * (self.hold_alt - clear), v_des)
        stop = np.where(v_y < -v_land, (v_y ** 2 - v_land ** 2) / (2 * np.maximum(h, 0.5)), 0.0)
        # The engine can only push along +up, so the vertical command is kept positive.
        a_v = np.maximum(g + stop + 2.0 * (v_des - v_y), 0.1 * g)

        return a_h, a_v

    def _catch_guidance(self, env, mass, g):
        """Desired accelerations for a tower catch: stay above the tower until close, then settle the catch
        point into the arms and hover there."""
        rk, spec, cc = env.rocket, env.cfg.rocket, env.cfg.catch
        cp, tgt = env.catch_point(), env.catch_target()
        err = tgt[:, [0, 2]] - cp[:, [0, 2]]
        hdist = np.linalg.norm(err, axis=-1)
        v_h, v_y = rk.vel[:, [0, 2]], rk.vel[:, 1]

        # Near the tower, keep the whole booster above its top (the catch point sits near the booster's top).
        clear_tower = env.tower_base + cc.tower_height + spec.length + 2.0 - tgt[:, 1]
        # distance from the tower to the straight path (booster -> catch target), so boosters starting
        # behind the tower stay high until they are past it
        a, b = rk.pos[:, [0, 2]], tgt[:, [0, 2]]
        ab = b - a
        u = np.clip(np.sum((env.tower_xz - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-9), 0.0, 1.0)
        d_tower = np.linalg.norm(a + u[:, None] * ab - env.tower_xz, axis=-1)
        floor = clear_tower * np.clip((cc.tower_width / 2 + spec.radius + 9.0 - d_tower) / 4.0, 0.0, 1.0)
        h = cp[:, 1] - tgt[:, 1] - floor

        a_brake = 0.3 * np.maximum(spec.t_max / mass - g, 0.5)
        far = h > 3.0
        v_des = np.where(far, -(1.0 + np.sqrt(2 * a_brake * np.maximum(h - 3.0, 0.0))), np.clip(-0.6 * h, -2.0, 2.0))
        stop = np.where(far & (v_y < -1.0), (v_y ** 2 - 1.0) / (2 * np.maximum(h - 3.0, 0.5)), 0.0)
        a_v = np.maximum(g + stop + 2.0 * (v_des - v_y), 0.1 * g)

        t_go = np.maximum(h, 0.0) / np.maximum(-v_y, 1.0) + 3.0
        v_far = self.k_close * err / t_go[:, None]
        v_far *= np.minimum(1.0, 8.0 / np.maximum(np.linalg.norm(v_far, axis=-1), 1e-9))[:, None]
        v_near = np.clip(0.35 * err, -1.5, 1.5)
        near = (hdist < 6.0)[:, None]
        v_h_des = np.where(near, v_near, v_far)
        a_h = np.where(near, 0.9, self.k_vel) * (v_h_des - v_h)
        return a_h, a_v

    def __call__(self, env: VecRocketEnv) -> np.ndarray:
        rk, spec = env.rocket, env.cfg.rocket
        d = rk.derived()
        mass, inertia, arm = d["mass"], d["inertia"], d["lever_arm"]
        g = env.gravity
        if env.cfg.task == "catch":
            a_h, a_v = self._catch_guidance(env, mass, g)
        else:
            a_h, a_v = self._landing_guidance(env, mass, g)

        # Desired up vector: tilt set by horizontal vs vertical accel demand, then limited.
        h_vec = np.stack([a_h[:, 0], np.zeros(env.n), a_h[:, 1]], -1) / np.maximum(a_v, g)[:, None]
        h_mag = np.linalg.norm(h_vec, axis=-1)
        tilt_des = np.minimum(np.arctan(h_mag), self.max_tilt)
        h_dir = h_vec / np.maximum(h_mag, 1e-9)[:, None]
        up_des = np.sin(tilt_des)[:, None] * h_dir
        up_des[:, 1] = np.cos(tilt_des)

        up = up_vector(rk.quat)
        throttle = np.clip(mass * a_v / (np.maximum(up[:, 1], 0.5) * spec.t_max), 0.0, 1.0)
        # Below minimum throttle the engine must be cut (low-gravity bodies need this to descend).
        throttle = np.where(throttle < 0.7 * spec.min_throttle, 0.0, np.maximum(throttle, spec.min_throttle))

        # Attitude: rotate current up toward desired up; map body torque to gimbal.
        err_world = np.cross(up, up_des)
        err_body = quat_rotate(quat_conj(rk.quat), err_world)
        alpha = self.kp_att * err_body - self.kd_att * rk.omega
        torque = alpha * inertia
        thrust = np.maximum(throttle * spec.t_max, 1.0)
        # τx ≈ -arm·T·δx and τz ≈ -arm·T·δz (see physics.RocketBatch.step).
        dx = -torque[:, 0] / (arm * thrust)
        dz = -torque[:, 2] / (arm * thrust)
        roll = -self.k_roll * rk.omega[:, 1] * inertia[:, 1] / max(spec.rcs_roll_torque, 1e-9)

        return np.stack([2 * throttle - 1,
                         np.clip(dx / spec.max_gimbal, -1, 1),
                         np.clip(dz / spec.max_gimbal, -1, 1),
                         np.clip(roll, -1, 1)], -1)
