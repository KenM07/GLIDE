"""Release targeting-lite utilities."""

import numpy as np
from .cw import cw_matrices, cw_propagate


def solve_dv0_minimize_rT(r0, v0, n, tof, dv0_max=None, dv0_axis_max=None):
    phi_rr, phi_rv, _, _ = cw_matrices(n, tof)
    r_pred = phi_rr @ r0 + phi_rv @ v0
    # Least-squares dv0 to minimize r(T)
    dv0 = -np.linalg.pinv(phi_rv) @ r_pred

    if dv0_axis_max is not None:
        dv0_axis_max = np.asarray(dv0_axis_max, dtype=float)
        dv0 = np.clip(dv0, -dv0_axis_max, dv0_axis_max)

    if dv0_max is not None:
        dv_mag = np.linalg.norm(dv0)
        if dv_mag > dv0_max and dv_mag > 1e-12:
            dv0 = dv0 * (dv0_max / dv_mag)

    return dv0, r_pred


def solve_dt_offset_minimize_rT(r0, v0, n, tof, dt_max, dt_step):
    best = {
        "dt": 0.0,
        "r_pred": None,
        "r_norm": None,
        "r0": r0,
        "v0": v0,
    }
    steps = int(np.floor(dt_max / dt_step))
    for k in range(-steps, steps + 1):
        dt = k * dt_step
        r0_dt, v0_dt = cw_propagate(r0, v0, n, dt)
        r_pred, _ = cw_propagate(r0_dt, v0_dt, n, tof)
        r_norm = float(np.linalg.norm(r_pred))
        if best["r_norm"] is None or r_norm < best["r_norm"]:
            best = {
                "dt": float(dt),
                "r_pred": r_pred,
                "r_norm": r_norm,
                "r0": r0_dt,
                "v0": v0_dt,
            }

    return best
