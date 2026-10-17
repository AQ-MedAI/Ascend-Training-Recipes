"""Reconstruct fixed framework sources from public commits and verified patches."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def manifest_sources(manifest, profile=None):
    if "profiles" in manifest:
        selected = profile or manifest.get("default_profile")
        if selected not in manifest["profiles"]:
            raise ValueError(f"Unknown source profile: {selected}")
        return manifest["profiles"][selected]["sources"]
    return manifest["sources"]


def patch_specs(source):
    if "patches" in source:
        return [(entry["path"], entry["sha256"]) for entry in source["patches"]]
    result = [(source["patch"], source["patch_sha256"])]
    if source.get("backport_patch"):
        if "reference_files" not in source:
            raise RuntimeError("Missing pre-backport inventory")
        result.append((source["backport_patch"], source["backport_patch_sha256"]))
    return result


def verify(root, source, *, tracked=False):
    ignored = {".git", "__pycache__", ".DS_Store", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    actual_names = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if (p.is_file() or p.is_symlink())
        and not ignored.intersection(p.relative_to(root).parts)
        and not p.name.endswith(".pyc")
    }
    if tracked:
        # Ignore build artifacts covered by the source repository's .gitignore.
        names = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        )
        actual_names = {name.decode() for name in names.split(b"\0") if name}
    if "gitlinks" in source:
        staged = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"])
        gitlinks = {}
        for entry in staged.split(b"\0"):
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            mode, commit, stage = metadata.decode().split()
            if mode == "160000":
                if stage != "0":
                    raise RuntimeError(f"Conflicted submodule: {name.decode()}")
                gitlinks[name.decode()] = commit
        if gitlinks != source["gitlinks"]:
            raise RuntimeError(f"Nested submodule pins differ: {root}")
        actual_names = {
            name for name in actual_names if not any(name == link or name.startswith(link + "/") for link in gitlinks)
        }
    if actual_names != set(source["files"]):
        raise RuntimeError(f"Source inventory mismatch: {root}")
    for name, expected in source["files"].items():
        path = root / name
        if expected.get("kind") == "symlink":
            actual = {"kind": "symlink", "target": os.readlink(path)}
        else:
            if path.is_symlink():
                raise RuntimeError(f"Unexpected symlink: {path}")
            actual = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executable": bool(path.stat().st_mode & 0o111),
            }
        if actual != expected:
            raise RuntimeError(f"Source checksum or mode mismatch: {path}")


def prepare(root, manifest_path, profile=None):
    manifest = json.loads(manifest_path.read_text())
    for name, source in manifest_sources(manifest, profile).items():
        target = root / name
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite existing source: {target}")
        patches = patch_specs(source)
        # Validate all patch inputs before fetching or creating a checkout.
        for filename, checksum in patches:
            patch = manifest_path.parent / filename
            if hashlib.sha256(patch.read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f"Patch checksum mismatch: {patch}")
        target.mkdir(parents=True)

        def git(*args):
            return subprocess.check_output(["git", "-C", str(target), *args], text=True)

        git("init", "-q")
        git("remote", "add", "origin", source["repository"])
        git("fetch", "--depth=1", "origin", source["commit"])
        git("checkout", "--detach", "FETCH_HEAD")
        if git("rev-parse", "HEAD").strip() != source["commit"]:
            raise RuntimeError(f"Unexpected base commit: {name}")
        if "patches" in source:
            reference = {"files": source["reference_files"]}
            if "gitlinks" in source:
                reference["gitlinks"] = source["gitlinks"]
            verify(target, reference)
        for index, (filename, _) in enumerate(patches):
            patch = manifest_path.parent / filename
            if patch.stat().st_size:
                git("apply", "--check", str(patch.resolve()))
                git("apply", str(patch.resolve()))
            if index == 0 and source.get("backport_patch"):
                verify(target, {"files": source["reference_files"]})
        verify(target, source)
        print(
            f"Verified {name}: {source['commit']} + " + " + ".join(p[0] for p in patches),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dependencies/framework-sources.json",
        help="Source manifest (defaults to the repository framework manifest)",
    )
    parser.add_argument("--profile", help="Profile for a manifest containing multiple source variants")
    args = parser.parse_args()
    prepare(args.root, args.manifest, args.profile)
