"""Initialize the four public forks and select a registry profile."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("v41", "v4"), nargs="?", default="v41")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    components = json.loads((root / "dependencies/registry.json").read_text())["components"]
    paths = ["third_party/" + name for name in components]

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True)

    # Inspect all checkouts before changing any; do not overwrite user edits.
    for path in paths:
        if (root / path / ".git").exists():
            dirty = subprocess.check_output(["git", "-C", str(root / path), "status", "--porcelain"], text=True)
            if dirty:
                raise SystemExit(f"Local changes in {path}; commit or stash before switching profiles")
    git("submodule", "sync", "--", *paths)
    git("-c", "submodule.recurse=false", "submodule", "update", "--init", "--checkout", "--", *paths)
    for name, component in components.items():
        sha = component["profiles"][args.profile]["commit"]
        path = "third_party/" + name
        exists = subprocess.run(
            ["git", "-C", str(root / path), "cat-file", "-e", sha + "^{commit}"], capture_output=True
        )
        if exists.returncode:
            git("-C", path, "fetch", "--no-tags", "origin", sha)
        git("-C", path, "checkout", "--detach", sha)
    print(f"Sources selected: {args.profile}. Submodule pointers in this recipe remain the default V4.1 pins.")


if __name__ == "__main__":
    main()
