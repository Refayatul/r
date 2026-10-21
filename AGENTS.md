# AGENTS.md — Refayatul/r (public measurement harness)

## Dropped features: keep the code, disable the calling

When a feature, experiment or data source is dropped, **never delete** the code, scripts, branches
or data. Disable it instead: default-off flag (or removed call site), an entry in
`docs/DISABLED_FEATURES.md` (flag/default, re-enable, files, measurements, why, date, re-enable
test), and a re-enable test. Deletion is an owner-only action with written approval.

## Public-repo hygiene

This repo is public. It may contain clip keys, public preview URLs and opaque query ids only. No
labels, marks, strata, sealed keys, API keys, channel text or frame images. The label text and the
reference maxima live in repository secrets and are never written to logs or artifacts.
