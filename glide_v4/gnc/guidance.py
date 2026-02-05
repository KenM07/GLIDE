"""Simple closed-loop guidance in LVLH."""

import numpy as np
from .cw import cw_matrices


class Guidance:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_update_t = -1.0e9
        self.last_mode = None
        self.last_cmd = np.zeros(3)

    def compute(self, t, r_lvh, v_lvh, mode):
        cadence = float(self.cfg["cadence_s"][mode])
        update_now = (t - self.last_update_t) >= cadence or (mode != self.last_mode)
        if update_now:
            method = self.cfg.get("method", "pd")
            if method == "cw_target":
                n = float(self.cfg["n_rad_s"])
                t_end = float(self.cfg["t_end_s"])
                t_go = max(t_end - t, cadence)
                phi_rr, phi_rv, _, _ = cw_matrices(n, t_go)
                dv0 = -np.linalg.pinv(phi_rv) @ (phi_rr @ r_lvh + phi_rv @ v_lvh)
                a_cmd = dv0 / max(cadence, 1e-6)
            else:
                gains = self.cfg["gains"][mode]
                kp = float(gains["kp"])
                kd = float(gains["kd"])
                # PD guidance: reduce miss distance and relative speed.
                a_cmd = -kp * r_lvh - kd * v_lvh

            # Speed shaping near gates.
            speed = np.linalg.norm(v_lvh)
            target_speed = self.cfg.get("target_speeds", {}).get(mode)
            if target_speed is not None and speed > 1e-9:
                speed_gain = float(self.cfg.get("speed_gain", 0.0))
                if speed > target_speed:
                    a_cmd -= speed_gain * (speed - target_speed) * (v_lvh / speed)

            max_accel = float(self.cfg.get("max_accel", 1e-3))
            a_mag = np.linalg.norm(a_cmd)
            if a_mag > max_accel:
                a_cmd = a_cmd * (max_accel / a_mag)

            self.last_cmd = a_cmd
            self.last_update_t = t
            self.last_mode = mode

        return self.last_cmd.copy()
