# GLIDE V4 Simulation (Phased Scaffold)

This repository is a structured, phased simulation scaffold for early proof-of-concept of the GLIDE V4 guidance and transfer physics. Phase 1 focuses on two-body + J2 + simple drag in ECI, with LVLH guidance, and parameterized DDM and mEDT actuation. Phase 2 adds bounded environment variability and Monte Carlo.

## Quick Start

1. Create a virtual environment.
2. Install dependencies.
3. Run the deterministic demo.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_demo.py configs/tagged_transfer.yaml
```

To run the gate logic sanity case:

```bash
python run_demo.py configs/gate_sanity.yaml
```

Outputs (plots and logs) are written to `outputs/`.

## Project Layout

- `dynamics/`: Orbit propagation, J2, drag, atmosphere, mEDT model, frame transforms.
- `gnc/`: LVLH guidance and control allocation for DDM and mEDT.
- `modes/`: Tag mode logic and corridor gate tracking.
- `sim/`: Scenario runner, logging, metrics, plotting.
- `mc/`: Monte Carlo harness and sampling.
- `configs/`: YAML configs for scenarios.
- `tests/`: Unit tests for frame transforms and a known-case sanity check.

## Phases

- Phase 1: Deterministic transfer with closed-loop guidance. DDM and mEDT are simplified but parameterized.
- Phase 2: Monte Carlo with uncertainty bounds and sensitivity sweeps.

## Notes

- DDM is modeled as a ballistic coefficient modulation (area sweep).
- mEDT uses a simplified dipole field and the force model: F = eta * I * L * B * sin(theta).
- Corridor capture is modeled as gate logic (R_ACQ, R_COR, R_PRE, R_LATCH) with speed limits.

## Tests

```bash
python -m unittest discover -s tests
```
