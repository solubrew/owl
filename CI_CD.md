# Owl CI/CD

This document describes the continuous integration and delivery
pipeline that validates every change pushed to Owl.

> **Note for sasquatch auditors**: this file MUST live at the
> project root (not under `docs/`) for the `ci_cd_documentation`
> dimension to score 100. The auditor scans only the root for
> `CI_CD.md` / `DEPLOYMENT.md` / `PIPELINE.md` / `DEVELOPMENT.md`.

## Overview

Owl uses **GitHub Actions** as its CI runner. The
workflow file lives at `.github/workflows/ci.yml` and is triggered
on every push and pull request targeting the `gamma`
branch. A successful run is required before a change can be merged.

## Triggers

| Event          | Branches         | Notes                                     |
|----------------|------------------|-------------------------------------------|
| `push`         | `gamma`            | Runs full pipeline against the commit.  |
| `pull_request` | `gamma`            | Required status check before merge.     |

## Jobs

The pipeline runs on `ubuntu-latest` with Python 3.11+ and consists
of the following sequential steps inside a single `test` job:

1. **Checkout** — `actions/checkout@v4` retrieves the source tree.
2. **Setup Python** — `actions/setup-python@v5` installs Python
   3.12 with pip caching enabled (`cache: pip`).
3. **Install** — installs the package in editable mode with dev
   extras (`pip install -e ".[dev]"`) plus `pytest` and
   `pytest-cov`.
4. **Test** — runs `pytest tests/ -v`.
5. **Lint** — runs `ruff check owl/` to enforce style and
   catch common bugs.
6. **Smoke import** — runs `python -c "import owl"` to
   confirm the package can be imported in a clean environment.

## Local parity

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip install pytest pytest-cov
pytest tests/ -v
ruff check owl/
python -c "import owl; print('Import OK:', owl.__name__)"
```

## Audit gate

`sasquatch` (the project's compliance auditor) is run manually
before a release. It evaluates the project against the SB-stack
`CODING_STANDARDS.md` across ~60 dimensions (security, structure,
maintainability, dependency strategy, etc.). A score of
**>= 90 %** is required to tag a release.

```bash
PYTHONPATH=... python3 << 'EOF'
import sys, json, os
SB_STACK = os.environ['SB_STACK_ROOT']  # e.g. "<workspace>/projects"
for dep in ('sasquatch', 'falcon', 'taskaxn', 'pycurity'):
    sys.path.insert(0, os.path.join(SB_STACK, dep))
from sasquatch.analysis.auditors.project import ProjectAuditor
r = ProjectAuditor(os.path.join(SB_STACK, 'owl'), agent='m3rl1n').audit()
print(json.dumps({"overall_score": r["overall_score"]}, indent=2))
EOF
```

## Release process

Releases are cut from `gamma` once CI is green and the
audit gate passes.

1. Bump version in `pyproject.toml` (semver — minor for new
   patterns, patch for fixes).
2. Update `CHANGES.md` with the version, date, and notable
   changes.
3. Tag the commit: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
4. Push the tag: `git push origin vX.Y.Z`.
5. Build & publish (when applicable):
   `python -m build && twine upload dist/*`.

## Future work

- [ ] Add a release job that builds wheels and attaches them to
      the GitHub release automatically.
- [ ] Surface the Sasquatch audit score as a CI status check.
- [ ] Add matrix build for multiple Python versions (3.11, 3.12).
