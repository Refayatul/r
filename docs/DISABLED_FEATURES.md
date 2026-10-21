# DISABLED FEATURES — registry (Refayatul/r)

Standing rule: dropped features are disabled (default-off flag or removed call site), never deleted.

| feature | flag / default | re-enable | files | measured | why | date | re-enable test |
|---|---|---|---|---|---|---|---|
| Census-1 scoring harness | `workflow_dispatch` only (no schedule) | dispatch `census1.yml` stage=calibration/score/aggregate | `score_nsfw.py`, `harness/`, `.github/workflows/census1.yml` | `SIGLIP_CALIBRATION.md` (bench): 1.817 / 0.0907 / 0.2873 s per frame, 113-clip check at 1e-6 | measurement only, never in the daily pipeline | 2026-10-21 | calibration job: 0 clips outside 1e-3, no label text in logs |
| Freepik + Marqo models | only inside the census workflow | same dispatch as above | `score_nsfw.py` | same | production gate is SigLIP-only; these are measurement tools | 2026-10-21 | three-model calibration cross-check at 1e-6 |
