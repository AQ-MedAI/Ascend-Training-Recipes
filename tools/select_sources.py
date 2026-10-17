"""Initialize pinned forks and apply managed patches to upstream submodules."""

import argparse
import json
import subprocess
from pathlib import Path

from patched_sources import ensure_source, inspect_checkout, load_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("v41", "v4"), nargs="?", default="v41")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    components = json.loads((root / "dependencies/registry.json").read_text())["components"]
    paths = ["third_party/" + name for name in components]
    patched = {}

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True)

    # Check every checkout and patch before modifying any source.
    for name, component in components.items():
        checkout = root / "third_party" / name
        if component.get("source_manifest"):
            source, patch = load_source(root, name, component, args.profile)
            alternatives = [
                load_source(root, name, component, profile)
                for profile in component["profiles"]
                if profile != args.profile
            ]
            inspect_checkout(checkout, [source, *(entry[0] for entry in alternatives)])
            patched[name] = (source, patch, alternatives)
        elif (checkout / ".git").exists():
            dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True)
            if dirty:
                raise SystemExit(f"Local changes in {checkout}; commit or stash before switching profiles")
    git("submodule", "sync", "--", *paths)
    plain_paths = ["third_party/" + name for name in components if name not in patched]
    if plain_paths:
        git(
            "-c",
            "submodule.recurse=false",
            "submodule",
            "update",
            "--init",
            "--checkout",
            "--",
            *plain_paths,
        )
    for name, component in components.items():
        path = "third_party/" + name
        checkout = root / path
        if name in patched:
            if not (checkout / ".git").exists():
                git(
                    "-c",
                    "submodule.recurse=false",
                    "submodule",
                    "update",
                    "--init",
                    "--checkout",
                    "--",
                    path,
                )
            ensure_source(checkout, *patched[name])
            continue
        sha = component["profiles"][args.profile]["commit"]
        exists = subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", sha + "^{commit}"], capture_output=True)
        if exists.returncode:
            git("-C", path, "fetch", "--no-tags", "origin", sha)
        git("-C", path, "checkout", "--detach", sha)
    print(f"Sources selected: {args.profile}. Upstream submodules use verified profile patches.")


if __name__ == "__main__":
    main()
