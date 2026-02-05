"""Monte Carlo for dispersion recovery."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sim.scenario import load_config, build_reference_state, get_initial_relative_state
from sim.runner import run_sim_config, apply_release_targeting
from mc.sampler import sample_config


def _hist_plot(values, bins, title, xlabel, out_path):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _failure_reason(gates, terminal_enabled=False):
    for v in gates.violations:
        if v.get("type") == "speed_violation":
            gate = v.get("gate", "unknown")
            return f"speed_violation_at_{gate}"
    if "R_ACQ" not in gates.crossings:
        return "no_R_ACQ"
    if "R_COR" not in gates.crossings:
        return "no_R_COR"
    if terminal_enabled:
        if "R_PRE" not in gates.crossings:
            return "no_R_PRE"
        if "R_LATCH" not in gates.crossings:
            return "no_R_LATCH"
    return "unknown"


def run_dispersion_mc(
    config_path,
    runs=500,
    seed=123,
    output_dir="outputs/dispersion_recovery_mc",
    terminal_enabled=False,
    dt_override=None,
):
    cfg = load_config(config_path)
    rng = np.random.default_rng(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mc_cfg = cfg.get("mc", {})
    mc_dt = dt_override if dt_override is not None else mc_cfg.get("dt_s", None)

    count_acq = 0
    count_cor = 0
    count_pre = 0
    count_latch = 0
    speeds_cor = []
    max_speed = {"R_ACQ": 0.0, "R_COR": 0.0, "R_PRE": 0.0, "R_LATCH": 0.0}
    dv_eqs = []
    fail_counts = {}

    for _ in range(runs):
        cfg_s = sample_config(cfg, rng)
        cfg_s["simulation"]["save_plots"] = False
        if mc_dt is not None:
            cfg_s["simulation"]["dt_s"] = float(mc_dt)
        if terminal_enabled:
            cfg_s["simulation"]["stop_on_gate"] = None
        else:
            cfg_s["simulation"]["stop_on_gate"] = "R_COR"
        cfg_s["terminal_capture"]["enabled"] = bool(terminal_enabled)

        r_ref, v_ref = build_reference_state(cfg_s)
        r0, v0 = get_initial_relative_state(cfg_s)
        r_nom, v_nom, _ = apply_release_targeting(cfg_s, r_ref, v_ref, r0, v0)

        disp_cfg = cfg_s.get("dispersion", {})
        r_bounds = np.array(disp_cfg.get("r_m", [0.0, 0.0, 0.0]), dtype=float)
        v_bounds = np.array(disp_cfg.get("v_mps", [0.0, 0.0, 0.0]), dtype=float)

        r_disp = rng.uniform(-r_bounds, r_bounds)
        v_disp = rng.uniform(-v_bounds, v_bounds)

        r_act = r_nom + r_disp
        v_act = v_nom + v_disp

        metrics, gates, _ = run_sim_config(
            cfg_s,
            control_enabled=True,
            guidance_enabled=True,
            r_rel_lvh0=r_act,
            v_rel_lvh0=v_act,
            r_rel_lvh_nom0=r_nom,
            v_rel_lvh_nom0=v_nom,
            fast_mode=True,
        )

        dv_eqs.append(metrics["dv_eq_applied"])

        if "R_ACQ" in gates.crossings:
            count_acq += 1
            max_speed["R_ACQ"] = max(max_speed["R_ACQ"], gates.crossings["R_ACQ"]["speed_mps"])
        if "R_COR" in gates.crossings:
            count_cor += 1
            speeds_cor.append(gates.crossings["R_COR"]["speed_mps"])
            max_speed["R_COR"] = max(max_speed["R_COR"], gates.crossings["R_COR"]["speed_mps"])
        if "R_PRE" in gates.crossings:
            count_pre += 1
            max_speed["R_PRE"] = max(max_speed["R_PRE"], gates.crossings["R_PRE"]["speed_mps"])
        if "R_LATCH" in gates.crossings:
            count_latch += 1
            max_speed["R_LATCH"] = max(max_speed["R_LATCH"], gates.crossings["R_LATCH"]["speed_mps"])

        if terminal_enabled:
            failed = not metrics["latch_success"]
        else:
            failed = not metrics["corridor_entry_success"]

        if failed:
            reason = _failure_reason(gates, terminal_enabled=terminal_enabled)
            fail_counts[reason] = fail_counts.get(reason, 0) + 1

    p_acq = count_acq / max(runs, 1)
    p_cor = count_cor / max(runs, 1)
    p_pre = count_pre / max(runs, 1)
    p_latch = count_latch / max(runs, 1)

    print(f"MC runs: {runs}")
    print(f"P(R_ACQ): {p_acq:.3f}")
    print(f"P(R_COR): {p_cor:.3f}")
    if terminal_enabled:
        print(f"P(R_PRE): {p_pre:.3f}")
        print(f"P(R_LATCH): {p_latch:.3f}")

    print("Max speed at gates (m/s):")
    print(f"  R_ACQ: {max_speed['R_ACQ']:.3f}")
    print(f"  R_COR: {max_speed['R_COR']:.3f}")
    print(f"  R_PRE: {max_speed['R_PRE']:.3f}")
    print(f"  R_LATCH: {max_speed['R_LATCH']:.3f}")

    if speeds_cor:
        speeds = np.array(speeds_cor)
        print("R_COR speed stats (m/s):")
        print(f"  min={speeds.min():.3f} mean={speeds.mean():.3f} median={np.median(speeds):.3f} max={speeds.max():.3f}")
        _hist_plot(speeds, bins=30, title="R_COR Entry Speed", xlabel="Speed (m/s)", out_path=out_dir / "r_cor_speed_hist.png")
    else:
        print("No R_COR crossings; speed distribution empty.")

    if dv_eqs:
        dv = np.array(dv_eqs)
        print("dv_eq_applied stats (m/s):")
        print(f"  min={dv.min():.4f} mean={dv.mean():.4f} median={np.median(dv):.4f} max={dv.max():.4f}")
        _hist_plot(dv, bins=30, title="dv_eq_applied", xlabel="dv_eq_applied (m/s)", out_path=out_dir / "dv_eq_applied_hist.png")

    if fail_counts:
        print("Failure breakdown:")
        total_fail = sum(fail_counts.values())
        for reason, count in sorted(fail_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / max(total_fail, 1)
            print(f"  {reason}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    run_dispersion_mc("configs/dispersion_recovery.yaml", runs=500, seed=123)
