# GLIDE V4 Simulation (Phased Scaffold)

This repository is a structured, phased simulation scaffold for early proof-of-concept of the GLIDE V4 guidance and transfer physics. Phase 1 focuses on two-body + J2 + simple drag in ECI, with LVLH guidance, and parameterized DDM and mEDT actuation. Phase 2 adds bounded environment variability, Monte Carlo, and terminal capture control.

## Quick Start

1. Create a virtual environment.
2. Install dependencies.
3. Run a deterministic demo.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_demo.py configs/tagged_transfer.yaml
```

To run the dispersion recovery scenario (with targeting + dispersions + dual-run overlay):

```bash
python run_demo.py configs/dispersion_recovery.yaml
```

To run the gate logic sanity case:

```bash
python run_demo.py configs/gate_sanity.yaml
```

To run Monte Carlo on dispersion recovery:

```bash
python run_mc_dispersion.py
```

To compare MC time step sensitivity explicitly:

```bash
python -c "from mc.dispersion_mc import run_dispersion_mc; run_dispersion_mc('configs/dispersion_recovery.yaml', runs=200, seed=123, terminal_enabled=False, dt_override=1.0, output_dir='outputs/dispersion_recovery_mc_dt1')"
python -c "from mc.dispersion_mc import run_dispersion_mc; run_dispersion_mc('configs/dispersion_recovery.yaml', runs=500, seed=123, terminal_enabled=False, dt_override=10.0, output_dir='outputs/dispersion_recovery_mc_dt10')"
```

To run terminal capture MC (uses terminal controller):

```bash
python -c "from mc.dispersion_mc import run_dispersion_mc; run_dispersion_mc('configs/dispersion_recovery.yaml', runs=500, seed=123, terminal_enabled=True, dt_override=5.0, output_dir='outputs/dispersion_recovery_mc_terminal')"
```

Outputs (plots and logs) are written to `outputs/`.

## Project Layout

- `dynamics/`: Orbit propagation, J2, drag, atmosphere, mEDT model, frame transforms.
- `gnc/`: LVLH guidance and control allocation for DDM and mEDT.
- `modes/`: Tag mode logic and corridor gate tracking (with gate-crossing interpolation).
- `sim/`: Scenario runner, logging, metrics, plotting, terminal capture controller.
- `mc/`: Monte Carlo harness and sampling (dispersion MC).
- `configs/`: YAML configs for scenarios.
- `tests/`: Unit tests for frame transforms and a known-case sanity check.

## Phases

- Phase 1: Deterministic transfer with closed-loop guidance. DDM and mEDT are simplified but parameterized.
- Phase 2: Monte Carlo with uncertainty bounds and sensitivity sweeps, plus terminal capture control.

## Notes

- DDM is modeled as a ballistic coefficient modulation (area sweep).
- mEDT uses a simplified dipole field and the force model: F = eta * I * L * B * sin(theta).
- Corridor capture is modeled as gate logic (R_ACQ, R_COR, R_PRE, R_LATCH) with speed limits.
- Gate speeds are checked at interpolated crossing time (not just discrete sample points).
- Terminal controller is only active at/inside R_COR and enforces LOS and total speed limits without changing the official caps.

## Tests

```bash
python -m unittest discover -s tests
```
