"""Apply and verify recipe-managed patches on upstream submodules."""

import hashlib
import json
import subprocess

from prepare_framework_sources import verify


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True)


def load_source(root, name, component, profile):
    manifest = root / component["source_manifest"]
    source = json.loads(manifest.read_text())["sources"][name]
    if source["commit"] != component["profiles"][profile]["commit"] or source["repository"] != component["repository"]:
        raise RuntimeError(f"Registry and patch manifest disagree: {name}")
    patch = (manifest.parent / source["patch"]).resolve()
    if hashlib.sha256(patch.read_bytes()).hexdigest() != source["patch_sha256"]:
        raise RuntimeError(f"Patch checksum mismatch: {patch}")
    return source, patch


def is_prepared(checkout, source):
    """Accept exactly our patched tree; refuse any other local modifications."""
    if not (checkout / ".git").exists():
        return False
    staged = git(checkout, "diff", "--cached", "--name-only")
    if staged:
        raise RuntimeError(f"Staged changes in {checkout}; preserve them before updating sources")
    dirty = git(checkout, "status", "--porcelain")
    if dirty:
        if git(checkout, "rev-parse", "HEAD").strip() != source["commit"]:
            raise RuntimeError(f"Local changes on an unexpected base: {checkout}")
        try:
            verify(checkout, source, tracked=True)
        except (RuntimeError, OSError) as exc:
            raise RuntimeError(
                f"Local changes differ from the managed patch: {checkout}; preserve them before updating"
            ) from exc
        return True
    return False


def ensure_source(checkout, source, patch):
    if is_prepared(checkout, source):
        return
    exists = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", source["commit"] + "^{commit}"], capture_output=True
    )
    if exists.returncode:
        git(checkout, "fetch", "--no-tags", "origin", source["commit"])
    git(checkout, "checkout", "--detach", source["commit"])
    verify(checkout, {"files": source["reference_files"]}, tracked=True)
    git(checkout, "apply", "--check", str(patch))
    git(checkout, "apply", str(patch))
    verify(checkout, source, tracked=True)
