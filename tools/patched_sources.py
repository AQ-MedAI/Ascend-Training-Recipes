"""Apply and switch verified patch profiles on upstream submodules."""

import hashlib
import json
import subprocess
from pathlib import Path

from prepare_framework_sources import manifest_sources, patch_specs, source_commit, verify


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True)


def load_source(root, name, component, profile):
    manifest = root / component["source_manifest"]
    source = manifest_sources(json.loads(manifest.read_text()), profile)[name]
    if source["commit"] != component["profiles"][profile]["commit"] or source["repository"] != component["repository"]:
        raise RuntimeError(f"Registry and patch manifest disagree: {name}")
    return source, checked_patches(manifest, source)


def checked_patches(manifest, source):
    patches = []
    for filename, checksum in patch_specs(source):
        patch = (manifest.parent / filename).resolve()
        if hashlib.sha256(patch.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Patch checksum mismatch: {patch}")
        patches.append(patch)
    return patches


def load_legacy_sources(root, name, component):
    """Recognize previous managed trees for a verified upgrade, not for selection."""
    manifest = root / component["source_manifest"]
    sources = json.loads(manifest.read_text()).get("legacy_sources", {}).get(name, [])
    commits = {entry["commit"] for entry in component["profiles"].values()}
    result = []
    for source in sources:
        if source["repository"] != component["repository"] or source["commit"] not in commits:
            raise RuntimeError(f"Invalid legacy source baseline: {name}")
        result.append((source, checked_patches(manifest, source)))
    return result


def inspect_checkout(checkout, sources):
    """Return the matching profile index, or None for an unmodified checkout."""
    if not (checkout / ".git").exists():
        return None
    if git(checkout, "diff", "--cached", "--name-only"):
        raise RuntimeError(f"Staged changes in {checkout}; preserve them before updating sources")
    if not git(checkout, "status", "--porcelain"):
        return None
    head = source_commit(checkout)
    for index, source in enumerate(sources):
        if head != source["commit"]:
            continue
        try:
            verify(checkout, source, tracked=True)
            return index
        except (RuntimeError, OSError):
            pass
    raise RuntimeError(f"Local changes differ from every managed profile: {checkout}; preserve them before updating")


def is_prepared(checkout, source):
    return inspect_checkout(checkout, [source]) == 0


def ensure_source(checkout, source, patches, alternatives=()):
    candidates = [(source, patches), *alternatives]
    matched = inspect_checkout(checkout, [entry[0] for entry in candidates])
    if matched == 0:
        return
    # Fetch first, so a connection failure cannot remove the installed profile.
    exists = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", source["commit"] + "^{commit}"],
        capture_output=True,
    )
    if exists.returncode:
        git(checkout, "fetch", "--no-tags", "origin", source["commit"])
    if matched is not None:
        old_source, old_patches = candidates[matched]
        if isinstance(old_patches, Path):
            old_patches = [old_patches]
        # Inventory verification above authorizes removal only of known patch edits.
        for patch in reversed(old_patches):
            git(checkout, "apply", "--reverse", "--check", str(patch))
            git(checkout, "apply", "--reverse", str(patch))
        reference = {"files": old_source["reference_files"]}
        if "gitlinks" in old_source:
            reference["gitlinks"] = old_source["gitlinks"]
        verify(checkout, reference, tracked=True)
    git(checkout, "checkout", "--detach", source["commit"])
    reference = {"files": source["reference_files"]}
    if "gitlinks" in source:
        reference["gitlinks"] = source["gitlinks"]
    verify(checkout, reference, tracked=True)
    if isinstance(patches, Path):
        patches = [patches]
    for patch in patches:
        git(checkout, "apply", "--check", str(patch))
        git(checkout, "apply", str(patch))
    verify(checkout, source, tracked=True)
