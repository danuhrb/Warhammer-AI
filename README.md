# wh40k-sim-starter (Python + Shapely)

Minimal, hackable simulator scaffold for a 40k-like tabletop environment. It includes:
- **Shapely-based line of sight** (2.5D: terrain height-aware)
- **Collision checks** for circular bases vs terrain
- **Movement candidate generator** with simple cover/objective heuristics
- A tiny **Gymnasium**-style environment skeleton you can extend

> This is intentionally small and readable. Will build rules/features incrementally.

## Requirements
- Python 3.10+
- `pip install -r requirements.txt`

## Run the demo
```bash
python -m examples.demo
