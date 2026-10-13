"""Guard against local machine paths and PII leaking into tracked files.

Encodes the 2026-10-11 scrub: no `/home/`, `/Users/`, `/root/`, `/mnt/`
absolute paths or `file://` URLs in tracked sources. The git remote lives on
local disk, so any such string in the repo content leaks host layout.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# $PROJECT_DIR$ / $MODULE_DIR$ are IDE macros (portable), not local paths.
ALLOWED = re.compile(r"\$[A-Z_]+\$")

PATTERNS = [
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/root/"),
    re.compile(r"/mnt/"),
    re.compile(r"file://"),
]

# Doc mentions of the env vars that replaced the old hardcoded paths.
EXPECTED_ENV_VARS = {"OWL_DATA_PATH", "SB_STACK_ROOT"}


def _tracked_text_files():
    out = []
    SELF = Path(__file__).resolve()
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.resolve() == SELF:
            continue  # the guard is allowed to mention its patterns
        rel = p.relative_to(REPO_ROOT)
        parts = rel.parts
        if not parts or parts[0] in {".git", "__pycache__", "test_reports", ".idea",
                                     ".pytest_cache", "venv", ".venv"}:
            continue
        if p.name == "CHANGES.md":
            continue  # changelog is a historical record; prose may mention scrubbed patterns
        if any("__pycache__" in part for part in parts):
            continue
        try:
            p.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue  # binary
        out.append(p)
    return out


class TestNoLocalPaths:
    def test_no_local_path_patterns(self):
        offenders = []
        for p in _tracked_text_files():
            text = p.read_text()
            for i, line in enumerate(text.splitlines(), start=1):
                probe = ALLOWED.sub("", line)
                for pat in PATTERNS:
                    if pat.search(probe):
                        offenders.append(f"{p.relative_to(REPO_ROOT)}:{i}: {line.strip()[:100]}")
                        break
        assert not offenders, "local paths leaked into tracked files:\n" + "\n".join(offenders)

    def test_data_path_is_env_driven(self):
        """cmds/runTrain.py must read OWL_DATA_PATH, not embed a local default."""
        src = (REPO_ROOT / "cmds" / "runTrain.py").read_text()
        assert "OWL_DATA_PATH" in src
        assert "/home/" not in src

    def test_dead_prediction_module_gone(self):
        """owl/prediction.py (shadowed by the owl/prediction/ package) was deleted."""
        assert not (REPO_ROOT / "owl" / "prediction.py").exists()
        assert (REPO_ROOT / "owl" / "prediction" / "__init__.py").exists()


class TestNoPii:
    def test_no_email_addresses(self):
        offenders = []
        email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")
        for p in _tracked_text_files():
            text = p.read_text()
            for i, line in enumerate(text.splitlines(), start=1):
                for m in email.finditer(line):
                    hit = m.group(0).lower()
                    # example.com placeholders are template residue, not PII
                    if hit.endswith("@example.com") or "@sentry.io" in hit:
                        continue
                    offenders.append(f"{p.relative_to(REPO_ROOT)}:{i}: {hit}")
        assert not offenders, "emails in tracked files:\n" + "\n".join(offenders)
