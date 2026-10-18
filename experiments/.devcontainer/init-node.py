"""Seed node's persistent Pi settings, then replace this process with the command."""

import json
import os
from pathlib import Path
import pwd
import sys
import tempfile


DEFAULTS_PATH = Path("/usr/local/lib/autoresearch/pi-settings.json")


def read_settings(path: Path) -> dict:
    """Reject malformed settings instead of replacing user configuration."""
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid Pi settings in {path}: {error}") from error
    if not isinstance(settings, dict):
        raise ValueError(f"Invalid Pi settings in {path}: expected a JSON object")
    return settings


def seed_settings(home: Path, defaults_path: Path) -> bool:
    """Fill missing defaults without changing existing provider/model choices."""
    defaults = read_settings(defaults_path)
    pi_dir = home / ".pi"
    settings_dir = pi_dir / "agent"
    settings_path = settings_dir / "settings.json"
    try:
        settings = read_settings(settings_path)
    except FileNotFoundError:
        settings = {}

    original = settings.copy()
    if "defaultProvider" not in settings and "defaultModel" not in settings:
        for key in ("defaultProvider", "defaultModel"):
            if key in defaults:
                settings[key] = defaults[key]
    for key, value in defaults.items():
        if key not in ("defaultProvider", "defaultModel"):
            settings.setdefault(key, value)
    if settings == original:
        return False

    pi_dir.mkdir(mode=0o700, exist_ok=True)
    settings_dir.mkdir(mode=0o700, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".settings-", suffix=".json", dir=settings_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(settings, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, settings_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    return True


def main() -> None:
    if os.geteuid() != pwd.getpwnam("node").pw_uid:
        raise SystemExit("init-node: must run as node")
    if len(sys.argv) < 2:
        raise SystemExit("init-node: a command is required")
    try:
        seed_settings(Path.home(), DEFAULTS_PATH)
    except (OSError, ValueError) as error:
        raise SystemExit(f"init-node: {error}") from error
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
