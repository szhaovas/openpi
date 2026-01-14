
## Installation for Linux

Installing latest Pi version:

Inside _this_repository_root_:

```conda create --name openpi --all python=3.11

conda activate openpi

cd openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync --active

GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .```

Installing LIBERO client:

Inside _this_repository_root/openpi/_:

`conda create --name libero python=3.8`

`conda activate libero`

`GIT_LFS_SKIP_SMUDGE=1 uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match`

`GIT_LFS_SKIP_SMUDGE=1 uv pip install -e packages/openpi-client`

`GIT_LFS_SKIP_SMUDGE=1 uv pip install -e third_party/libero`

`export PYTHONPATH="$PYTHONPATH:$PWD/third_party/libero"`

Installing CPLEX for MILP repair:

Inside _this_repository_root/openpi/third_party/CPLEX_Studio201/_:

`python setup.py install`

---

## Generating QD environments

Inside _this_repository_root_:

`python -m src.generate_env env_search=latent_2d qd=cma_me task=task_5`

---


