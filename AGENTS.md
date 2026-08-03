# Repository rules

- Keep this repository independent of `/plm-shared/sunsiyuan/DeepFusionMem`.
- Source, tests, and small configs belong here. Data, models, indexes, logs, and checkpoints do not.
- GPU work runs through ClusterX; submission is never implicit.
- New architecture or objective variants require a clear scientific intent in `README.md`.
- Run `pytest` and `bash -n scripts/*.sh` before committing.
