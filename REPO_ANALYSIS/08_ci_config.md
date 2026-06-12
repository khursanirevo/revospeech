# 08 -- CI/CD, Build Configuration, and DevOps Analysis

This document provides a detailed analysis of the project's continuous integration, publishing, pre-commit hooks, gitignore, and build configuration.

---

## Table of Contents

1. [CI Pipeline (`ci.yml`)](#1-ci-pipeline-ciyml)
2. [Publish Workflow (`publish.yml`)](#2-publish-workflow-publishyml)
3. [Pre-commit Hooks (`.pre-commit-config.yaml`)](#3-pre-commit-hooks-pre-commit-configyaml)
4. [Gitignore (`.gitignore`)](#4-gitignore-gitignore)
5. [Project Configuration (`pyproject.toml`)](#5-project-configuration-pyprojecttoml)
6. [Overall DevOps / Deployment Strategy Summary](#6-overall-devops--deployment-strategy-summary)

---

## 1. CI Pipeline (`ci.yml`)

**File:** `.github/workflows/ci.yml`

### Triggers

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

The CI pipeline runs on:
- Every push to the `main` branch.
- Every pull request targeting the `main` branch.

**Notable:** The repo's default branch in git metadata is `master`, but CI is configured to watch `main`. This is a potential mismatch -- pushes to `master` will not trigger CI unless the branch name is updated to match. This should be verified and corrected if needed.

### Jobs

Three jobs are defined: `lint`, `test`, and `build`.

#### Job 1: `lint`

```yaml
lint:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
    - run: uv sync --extra dev
    - run: uv run ruff check revos/ tests/
```

| Property    | Value                            |
|-------------|----------------------------------|
| Runner      | `ubuntu-latest`                  |
| Toolchain   | `uv` (via `astral-sh/setup-uv@v4`) |
| Dependencies| Installed with `uv sync --extra dev` |
| Lint scope  | `revos/` and `tests/` directories |
| Linter      | `ruff check` (no autofix in CI)  |

- Uses `setup-uv@v4` from Astral (the uv package manager), avoiding separate Python setup.
- The `--extra dev` flag pulls in dev dependencies (`pytest`, `pytest-cov`, `ruff`) from `pyproject.toml`.
- Only runs `ruff check` (linting), not `ruff format --check`. Format checking is handled by pre-commit hooks locally but is **not enforced in CI**.

#### Job 2: `test`

```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.11", "3.12", "3.13"]
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
    - run: uv sync --extra dev --python ${{ matrix.python-version }}
    - run: uv run pytest tests/ -v --cov=revos
```

| Property         | Value                                           |
|------------------|-------------------------------------------------|
| Runner           | `ubuntu-latest`                                 |
| Strategy         | Matrix with Python `3.11`, `3.12`, `3.13`       |
| Toolchain        | `uv` with explicit Python version per matrix cell |
| Test runner      | `pytest tests/ -v --cov=revos`                  |
| Coverage         | Enabled via `--cov=revos` (generates coverage data) |
| Fail-on-warning  | Not configured (no `-W error` or `--strict`)     |

- Three Python versions are tested, matching the classifiers in `pyproject.toml`.
- `uv sync --python <version>` tells uv to use the specified Python version, leveraging uv's built-in Python management.
- Coverage is collected but there is **no coverage threshold enforcement** (no `--cov-fail-under` flag). Coverage data is generated but not gated.
- Tests run against `tests/` directory only.

#### Job 3: `build`

```yaml
build:
  runs-on: ubuntu-latest
  needs: [lint, test]
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
    - run: uv build
    - run: uv run python -c "import revos; print(revos.__version__)"
```

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Runner         | `ubuntu-latest`                            |
| Dependencies   | `lint` and `test` jobs must pass first     |
| Build tool     | `uv build` (delegates to hatchling)        |
| Verification   | Imports the built package and prints version |

- This is a **smoke test** for the build process -- it ensures the package can be built and imported successfully.
- Runs only after both `lint` and `test` pass (`needs: [lint, test]`).
- Does **not** upload any build artifacts. Artifacts are built and discarded.

### CI Pipeline Summary Diagram

```
push/PR to main
    |
    +---> lint (ruff check revos/ tests/)
    |
    +---> test (pytest, Python 3.11 / 3.12 / 3.13)
    |
    +---> build (needs lint + test)
              uv build
              import + version check
```

### CI Gaps and Observations

1. **No format enforcement in CI:** `ruff format --check` is not run in CI. Format violations could slip in from contributors not using pre-commit.
2. **No coverage gating:** Coverage is collected but no minimum threshold is enforced.
3. **No caching:** No `uv` cache or pip cache configuration, which could speed up CI runs.
4. **Branch mismatch risk:** CI watches `main` but git default branch is `master`.
5. **No matrix for lint/build:** Only test uses a matrix; lint and build use a single implicit Python version (uv default).
6. **No artifact upload:** Build artifacts are not uploaded, so they cannot be inspected or used downstream.

---

## 2. Publish Workflow (`publish.yml`)

**File:** `.github/workflows/publish.yml`

### Triggers

```yaml
on:
  release:
    types: [published]
```

Publishing is triggered when a **GitHub Release is published** (not just created or drafted -- specifically the `published` event type).

### Job: `publish`

```yaml
publish:
  runs-on: ubuntu-latest
  environment: pypi
  permissions:
    id-token: write
```

| Property     | Value                                          |
|--------------|------------------------------------------------|
| Runner       | `ubuntu-latest`                                |
| Environment  | `pypi` (GitHub Environment, requires approval if configured) |
| Permissions  | `id-token: write` (for Trusted Publishing / OIDC) |
| Python       | `3.11` (via `actions/setup-python@v5`)         |

### Steps

```yaml
steps:
  - uses: actions/checkout@v4

  - name: Set up Python
    uses: actions/setup-python@v5
    with:
      python-version: "3.11"

  - name: Install build dependencies
    run: pip install build

  - name: Build package
    run: python -m build

  - name: Publish to PyPI
    uses: pypa/gh-action-pypi-publish@release/v1
```

| Step                  | Tool/Action                         | Purpose                          |
|-----------------------|-------------------------------------|----------------------------------|
| Checkout              | `actions/checkout@v4`              | Clone the repo                   |
| Set up Python         | `actions/setup-python@v5`          | Python 3.11 environment          |
| Install build deps    | `pip install build`                 | Install `python-build` (PEP 517) |
| Build package         | `python -m build`                   | Build sdist + wheel via hatchling |
| Publish to PyPI       | `pypa/gh-action-pypi-publish@release/v1` | Upload to PyPI via OIDC   |

### Key Details

- **Trusted Publishing (OIDC):** Uses `id-token: write` permission and the official PyPA publish action. This means no API tokens are stored as secrets -- authentication uses GitHub's OIDC identity federation with PyPI.
- **Build tool difference from CI:** CI uses `uv build` while publish uses `python -m build` (via `pip install build`). Both delegate to hatchling, but the toolchain is inconsistent.
- **No Test-PyPI step:** The workflow publishes directly to the real PyPI. There is no test-PyPI publish step. This is typical for smaller projects but adds risk for first-time publishing.
- **Environment protection:** The `pypi` environment can be configured in GitHub repo settings with required reviewers, deployment branches, or wait timers -- providing a manual gate before publishing.
- **Single Python version for publish:** Uses Python 3.11, matching the minimum supported version.

---

## 3. Pre-commit Hooks (`.pre-commit-config.yaml`)

**File:** `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### Configured Hooks

| Hook            | Version  | Behavior                                                |
|-----------------|----------|---------------------------------------------------------|
| `ruff`          | v0.11.0  | Runs lint checks with `--fix` (autofixes safe issues)   |
| `ruff-format`   | v0.11.0  | Runs the ruff formatter (autoformats code)              |

### Analysis

- **Single repo, two hooks:** Both hooks come from `astral-sh/ruff-pre-commit`, which bundles the ruff linter and formatter.
- **Autofix enabled:** The `ruff` hook passes `--fix`, so it will automatically fix lint issues (unused imports, sorting, etc.) at commit time. Issues that cannot be autofixed will cause the commit to fail.
- **Formatter runs:** `ruff-format` will reformat code to match the configured style (88-char line length, Python 3.11 target as defined in `pyproject.toml`).
- **Ruff version:** Pinned to `v0.11.0` of the pre-commit mirror. The actual ruff ruleset is configured in `pyproject.toml` under `[tool.ruff.lint]`.
- **Rule set (from pyproject.toml):** `select = ["E", "F", "I", "W"]` which means:
  - **E**: pycodestyle errors
  - **F**: pyflakes
  - **I**: isort (import sorting)
  - **W**: pycodestyle warnings
- **Not enforced in CI:** The CI workflow only runs `ruff check` without `--fix` and does not run `ruff format --check`. A contributor who bypasses pre-commit could merge unformatted code.

### What is NOT configured

- No `prettier` or other non-Python formatters.
- No `mypy` or type checking hook.
- No `check-yaml`, `check-json`, `trailing-whitespace`, or `end-of-file-fixer` hooks from `pre-commit-hooks`.
- No `commit-msg` hooks for conventional commits.
- No `no-commit-to-branch` hook to protect `main`/`master`.

---

## 4. Gitignore (`.gitignore`)

**File:** `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/

# Testing
.pytest_cache/
htmlcov/
.coverage

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# RevoS
.omc/
tmp.py

# Models (downloaded at runtime, not committed)
# Manifests ARE committed (small YAML files)
# Weights are cached in ~/.cache/revos/
```

### Category Breakdown

| Category               | Patterns                                       | Purpose                                    |
|------------------------|------------------------------------------------|--------------------------------------------|
| Python bytecode        | `__pycache__/`, `*.py[cod]`, `*$py.class`, `*.so` | Compiled Python files                    |
| Python packaging       | `*.egg-info/`, `dist/`, `build/`, `*.egg`      | Build artifacts                           |
| Virtual environments   | `.venv/`, `venv/`                               | Local virtualenv directories              |
| Testing                | `.pytest_cache/`, `htmlcov/`, `.coverage`       | Pytest cache, coverage HTML reports       |
| IDE                    | `.vscode/`, `.idea/`, `*.swp`, `*.swo`          | VS Code, JetBrains, Vim swap files        |
| OS                     | `.DS_Store`, `Thumbs.db`                       | macOS and Windows OS metadata             |
| RevoS/OMC project      | `.omc/`, `tmp.py`                               | OMC state directory, temporary scripts    |

### Observations

- **Model weights excluded implicitly:** The comments note that model weights are downloaded at runtime and cached in `~/.cache/revos/`, not committed. Only small YAML manifest files are committed. However, there is no explicit pattern for model files (e.g., `*.onnx`, `*.bin`) -- this relies on weights never being placed in the project directory.
- **No `*.onnx` or model file patterns:** Since sherpa-onnx models are downloaded at runtime to a cache directory outside the project, this is not a problem in practice, but adding explicit exclusions would be defensive.
- **`tmp.py` is gitignored:** Per the project's CLAUDE.md instructions, `tmp.py` is used for experimentation and should never be committed.
- **No `.env` exclusion:** If environment variables are used for API keys or configuration, `.env` files are not excluded.

---

## 5. Project Configuration (`pyproject.toml`)

**File:** `pyproject.toml`

### Build System

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

| Property      | Value                          |
|---------------|--------------------------------|
| Build backend | `hatchling` (from Hatch)       |
| Build target  | Standard Python wheel + sdist  |
| Build command | `uv build` (CI) / `python -m build` (publish) |

Hatchling is a PEP 517/518 compliant build backend. It is lightweight and does not require the full Hatch toolchain.

### Project Metadata

```toml
[project]
name = "revos"
version = "0.1.0"
description = "A unified Python library for speech AI — ASR and TTS using open models"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "RevoS Team" }]
```

| Field           | Value                                                      |
|-----------------|------------------------------------------------------------|
| Package name    | `revos`                                                    |
| Version         | `0.1.0` (Semantic versioning, currently alpha)             |
| License         | MIT                                                        |
| Python minimum  | `>=3.11`                                                   |
| Status          | `Development Status :: 3 - Alpha`                         |

### Keywords and Classifiers

```toml
keywords = ["asr", "tts", "speech", "sherpa-onnx", "zipformer", "omnivoice"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

Classifiers declare support for Python 3.11, 3.12, and 3.13 -- matching the CI test matrix.

### Core Dependencies

```toml
dependencies = [
    "sherpa-onnx>=1.10",
    "sherpa-onnx-core",
    "onnxruntime>=1.16",
    "numpy",
    "soundfile",
    "click>=8.0",
    "pyyaml",
    "huggingface-hub>=1.11.0",
]
```

| Dependency         | Version Constraint | Purpose                                        |
|--------------------|--------------------|------------------------------------------------|
| `sherpa-onnx`      | `>=1.10`          | Primary speech engine (ASR + TTS via ONNX)     |
| `sherpa-onnx-core` | unversioned        | Core runtime for sherpa-onnx models            |
| `onnxruntime`      | `>=1.16`          | ONNX model inference runtime                   |
| `numpy`            | unversioned        | Numerical operations (audio arrays)            |
| `soundfile`        | unversioned        | Audio file I/O (read WAV, FLAC, OGG, etc.)     |
| `click`            | `>=8.0`           | CLI framework                                  |
| `pyyaml`           | unversioned        | YAML parsing (model manifests, config)         |
| `huggingface-hub`  | `>=1.11.0`        | Model catalog fetch and download from HF Hub   |

**Total core dependencies: 8**

### Optional Dependencies (Extras)

```toml
[project.optional-dependencies]
gpu = ["onnxruntime-gpu"]
tts = ["omnivoice"]
all = ["onnxruntime-gpu", "omnivoice"]
dev = ["pytest>=7.0", "pytest-cov", "ruff"]
```

| Extra   | Dependencies                    | Purpose                                  |
|---------|---------------------------------|------------------------------------------|
| `gpu`   | `onnxruntime-gpu`              | GPU-accelerated ONNX inference           |
| `tts`   | `omnivoice`                    | Additional TTS engine (OmniVoice)        |
| `all`   | `onnxruntime-gpu`, `omnivoice` | Everything (GPU + OmniVoice TTS)         |
| `dev`   | `pytest>=7.0`, `pytest-cov`, `ruff` | Development tools (testing + linting) |

**Notes:**
- Installing `gpu` extra adds `onnxruntime-gpu` but does **not** exclude `onnxruntime` (CPU). Both could be installed simultaneously, which may cause conflicts. Users needing GPU should potentially install without the CPU runtime.
- The `tts` extra adds `omnivoice` as a separate TTS engine, but `sherpa-onnx` itself already provides TTS capabilities. OmniVoice appears to be a supplementary/alternative engine.
- `dev` extra includes 3 packages: testing framework, coverage plugin, and linter.

### Entry Points

```toml
[project.scripts]
revos = "revos.cli.main:cli"

[project.entry-points."revos.models"]
```

| Entry Point Type     | Name   | Target                  | Status      |
|----------------------|--------|-------------------------|-------------|
| Console script       | `revos`| `revos.cli.main:cli`   | Active      |
| Model entry points   | --     | (empty)                 | Placeholder |

- The `revos` command-line tool invokes `click` CLI at `revos.cli.main:cli`.
- The `revos.models` entry point group is declared but empty -- this is a plugin/discovery mechanism for future model registration.

### Build Target Configuration

```toml
[tool.hatch.build.targets.wheel]
packages = ["revos"]
```

Tells hatchling to package the `revos/` directory as the wheel content.

### Pytest Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: requires model download and real inference",
]
```

| Setting      | Value                                                |
|--------------|------------------------------------------------------|
| Test paths   | `tests/`                                             |
| Custom marker| `slow` -- marks tests that download models and run real inference |

The `slow` marker allows filtering: `pytest -m "not slow"` to skip heavy integration tests.

### Ruff Configuration

```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

| Setting         | Value                              |
|-----------------|------------------------------------|
| Target Python   | 3.11                               |
| Line length     | 88 characters (Black default)      |
| Selected rules  | E (pycodestyle errors), F (pyflakes), I (isort), W (pycodestyle warnings) |

**Not selected (commonly enabled rules that are omitted):**
- `N` (pep8-naming)
- `UP` (pyupgrade)
- `B` (flake8-bugbear)
- `SIM` (flake8-simplify)
- `C4` (flake8-comprehensions)
- `D` (pydocstyle)
- `S` (flake8-bandit / security)

The ruff configuration is intentionally minimal -- lint-only with no additional complexity rules.

---

## 6. Overall DevOps / Deployment Strategy Summary

### Architecture

The project follows a **simple, modern Python DevOps pattern**:

1. **Local development:** Pre-commit hooks enforce formatting and linting via ruff. `uv` manages dependencies.
2. **Continuous integration:** GitHub Actions runs lint, test (3 Python versions), and build verification on every push/PR.
3. **Publishing:** Manual release process -- create a GitHub Release, which triggers automated PyPI publishing via OIDC Trusted Publishing.

### Toolchain Choices

| Concern          | Tool                         | Rationale                                     |
|------------------|------------------------------|-----------------------------------------------|
| Package manager  | `uv`                         | Fast, modern Python package manager by Astral  |
| Build backend    | `hatchling`                  | Lightweight PEP 517 build backend              |
| Linter/formatter | `ruff` v0.11.0               | Unified lint + format, extremely fast          |
| Test runner      | `pytest` >= 7.0              | Standard Python testing framework              |
| CI runner        | GitHub Actions               | Native GitHub integration                      |
| Publishing       | PyPI via OIDC                | No stored secrets, trusted identity federation  |

### Release Flow

```
Developer pushes to main/master
        |
        v
CI runs: lint + test (3.11, 3.12, 3.13) + build
        |
        v
Developer creates GitHub Release (tag + notes)
        |
        v
Publish workflow triggers
        |
        v
Build sdist + wheel --> Publish to PyPI via OIDC
```

### Strengths

1. **OIDC-based PyPI publishing** eliminates secret management overhead.
2. **Matrix testing** across 3 Python versions ensures broad compatibility.
3. **uv-based CI** is faster than traditional pip-based workflows.
4. **Pre-commit hooks** catch issues locally before they reach CI.
5. **Hatchling** is a modern, low-maintenance build backend.

### Identified Risks and Improvement Opportunities

1. **Branch name mismatch:** CI watches `main` but the git default branch is `master`. Either rename the branch or update the workflow.
2. **No format check in CI:** `ruff format --check` should be added to the lint job to catch formatting issues from contributors who skip pre-commit.
3. **No coverage threshold:** Adding `--cov-fail-under=N` to pytest would enforce minimum test coverage.
4. **Build tool inconsistency:** CI uses `uv build` while publish uses `pip install build && python -m build`. Standardizing on one approach would reduce drift.
5. **No caching:** Adding `uv` cache to GitHub Actions would reduce CI run times.
6. **No type checking:** `mypy` or `pyright` is not configured. As the codebase grows, type safety will become more important.
7. **GPU/CPU runtime conflict:** The `gpu` extra does not exclude the base `onnxruntime` CPU dependency, which could cause runtime conflicts.
8. **No `.env` in gitignore:** If secrets or configuration are ever stored in `.env` files, they could be accidentally committed.
9. **Empty model entry points:** The `revos.models` entry point group is declared but unused -- this is either a future feature or dead configuration.
10. **No release automation:** Version bumping, changelog generation, and release note creation are all manual. Tools like `commitizen` or `release-please` could automate this.
