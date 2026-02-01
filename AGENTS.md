# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds training code (`train.py`), data generation (`generate_training_data.py`), model logic (`models.py`), and utilities.
- `src/conf/` contains task configs (e.g., `parity.yaml`, `copy.yaml`); `src/conf/models/` stores model-specific configs.
- `src/transformers/` is a vendored/modified Transformers codebase used by this project.
- `models/` is the default output directory for checkpoints and logs produced by training runs.
- `requirements.txt` and `environment.yml` define Python/conda dependencies.

## Build, Test, and Development Commands
- Create the conda env: `conda env create -f environment.yml` (env name: `ltf`, Python 3.9).
- Activate the env: `conda activate ltf`.
- Install PyTorch wheels (CUDA 12.1 example):
  - `pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121`
- Run training for a task:
  - `cd src && python train.py --conf ./conf/parity.yaml`
- When running any project scripts (train/eval), activate the conda env first: `conda activate ltf`.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and `snake_case` for functions/variables; modules mirror file names (e.g., `generate_training_data.py`).
- YAML configs use lowercase task names (e.g., `parity.yaml`, `sum_reverse.yaml`).
- No formatter/linter is enforced in the repo; keep edits consistent with nearby code.

## Testing Guidelines
- There is no centralized test runner (no `pytest` config).
- Functional checks live in `src/test_func.py` as callable helpers (e.g., `test_model`, `test_model_multi`).
- If you add tests, document how to execute them (ideally via a simple CLI or script).

## Commit & Pull Request Guidelines
- Recent commits use short, simple messages (e.g., “update readme”, “clean up model config”). Follow that style: concise and task-focused.
- PRs should include:
  - A brief summary of the change and the affected tasks/configs.
  - Any training results or metrics if behavior changes.
  - Notes on new/updated config files under `src/conf/`.

## Configuration Tips
- Before training, populate `src/conf/wandb.yaml` with your Weights & Biases credentials.
- Training outputs (logs/checkpoints) default to `models/`; keep large artifacts out of git.
