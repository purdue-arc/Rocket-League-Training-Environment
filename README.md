# Rocket-League-Training-Environment

## Installation

Requires **Python 3.10 or newer** (developed on 3.12) and git.

### 1. Clone

```bash
git clone https://github.com/purdue-arc/Rocket-League-Training-Environment.git
cd Rocket-League-Training-Environment
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3.12 -m venv .rlSimEnv
source .rlSimEnv/bin/activate
```

Check that the venv's Python is the one running:

```bash
which python        # Windows: where python
python --version    # must be 3.10+
```

`which python` must point inside `.rlSimEnv`. Activate the venv again in every new terminal.

### 3. Install dependencies and the package

```bash
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
pip install -e . --no-deps
```

- `requirements.lock.txt` installs the exact versions the project was tested with.
- `pip install -e .` installs `carSoccer` in editable mode: changes to the source take effect without reinstalling.
- `--no-deps` stops pip from re-resolving packages the lock file already pinned.

### 4. Verify

```bash
python -c "import carSoccer, mujoco, torch, tensordict; print('ok')"
pytest -v
```

### Troubleshooting

- **"Defaulting to user installation because normal site-packages is not writeable"**: the venv is not active. Re-run the activate command and check `which python`.
- **Conda users:** run `conda deactivate` first, or `conda config --set auto_activate_base false`, so conda's Python doesn't shadow the venv's. If `python3.12` resolves to Anaconda, create the venv with Homebrew's Python by full path: `"$(brew --prefix python@3.12)/bin/python3.12" -m venv .rlSimEnv`.
- **Moved or renamed the project folder?** Venvs can't be moved. Delete `.rlSimEnv` and repeat steps 2–3.
- **A pinned version fails to install on your OS** (the lock file was generated on macOS): install from the project's declared dependencies instead with `pip install -e ".[dev]"`.

### Maintainers: adding a dependency

1. `pip install -e ".[dev]"`
2. `pip freeze --exclude-editable > requirements.lock.txt`
3. Commit `requirements.lock.txt`.

## Run

Watch a 3v3 match in the MuJoCo viewer (on macOS the viewer must be started with `mjpython`, which the venv provides):

```bash
mjpython scripts/watchMatch.py                    # random drivers
mjpython scripts/watchMatch.py --policy chase     # scripted ball-chasing bots
mjpython scripts/watchMatch.py --scatter 1.0      # random kickoff positions
mjpython scripts/watchMatch.py --randomize --noise   # new car variant and physics each match, noisy sensors
```

Running it with plain `python` (or the editor's Run button) relaunches it under `mjpython` automatically.

Train a policy (PPO, 3v3 self-play, one shared network for every car) and watch or evaluate it:

```bash
python scripts/train.py                                    # runs/selfplay: progress.csv, ckpt*.pt, latest.pt
python scripts/train.py --resume runs/selfplay/latest.pt --iterations 600
mjpython scripts/watchMatch.py --policy runs/selfplay/latest.pt
mjpython scripts/watchMatch.py --policy runs/selfplay/latest.pt --opponent chase
python scripts/evaluate.py runs/selfplay/latest.pt --opponent random
```

The reward pays a car for each new ball touch and every car of a team for a goal (minus the same for conceding);
weights are in `RewardConfig`. Each training line reports touches per car-minute, goals per world-minute and how fast
cars close on the ball, so you can see play improving before goals appear.

Other scripts:

```bash
python scripts/bench.py --worlds 256 --threads 1 4 9    # env steps per second (--realistic adds randomisation and noise)
python scripts/restitutionSweep.py                      # the M0 contact experiments
```

Tests: `python -m pytest -v` from the repo root. Test files are named `test*.py` (camelCase, configured in `pyproject.toml`).
