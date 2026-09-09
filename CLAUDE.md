# holoeml-processing

Phase extraction from holographic EML interferometry. **Read `README.md` first** — it already
documents the module layout, GPU/CPU backend, and quick usage; do not duplicate that here.

## Commands

```
uv sync                              # install/update .venv
uv run pytest                        # fast synthetic tests, seconds
uv run pytest tests/test_phase.py -k <name>
uv sync --extra cuda                 # optional GPU backend, see pyproject.toml comment
```

## Before touching the math, read the matching design doc

Every module in `phase/` is written against `docs/interference_model.md` (the per-frame model,
Eq. 8/17/20) and derives from it explicitly in its own docstrings. Cite the doc's equation number
rather than re-deriving:

- `docs/gauge_conventions.md` — every free-parameter ambiguity (sign branch, piston, `alpha_n`,
  `g_n`) and which convention pins it, and where. **Read before changing any phase-offset,
  sign, or normalization logic** — most "bugs" in this area are a convention mismatch, not math.
- `docs/aia.md`, `docs/step_field_residuals.md` — derivations behind `phase/methods/aia.py` and
  `phase/methods/step_field.py`.
- `docs/frame_moments.md` — why non-uniform phase steps/contrast leak into stack mean/variance.

## Do not read these — use the commands shown instead

- `data/**` — 6.5 GB of `.npz` captures, denied in `.claude/settings.json`.
- `.venv/**` — denied in `.claude/settings.json`.
- `scripts/test/*.ipynb` — exploratory notebooks against real acquisitions, not maintained
  deliverables. Outputs are stripped by nbstripout before commit, but your local working copy
  still carries them (megabytes of base64 PNGs). To see the code: `jq -r '.cells[] |
  select(.cell_type=="code") | .source[]' <file>`. To edit, use NotebookEdit with a known
  `cell_id`, never a full-file Read.

## Conventions worth knowing before editing `phase/`

- Every function takes `xp = get_array_module(...)` and calls `xp.` throughout — never import
  numpy/cupy directly inside `phase/`; that's what makes one source run on both backends.
- Large `(N, H, W)` arrays default to `float32`; small per-iteration linear algebra (condition
  numbers, normal-equation solves) always runs `float64` regardless of `dtype=`.
- Phase differencing/adding on wrapped angles goes through `wrap_add`/`wrap_sub`
  (`phase/backend.py`), not raw `+`/`-` — see their docstrings for why the difference matters.
