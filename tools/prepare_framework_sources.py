"""Reconstruct fixed framework sources from public commits and verified patches."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
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


def verify(root, source, *, tracked=False, generated_files=()):
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
    generated = set(generated_files)
    if generated.intersection(source["files"]):
        raise ValueError("Generated files must not mask manifest source files")
    actual_names -= generated
    if actual_names != set(source["files"]):
        missing = sorted(set(source["files"]) - actual_names)
        extra = sorted(actual_names - set(source["files"]))
        raise RuntimeError(f"Source inventory mismatch: {root}; missing={missing[:30]}, extra={extra[:30]}")
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


def fetch_commit(target, commit):
    """Retry shallow fetches without relaxing commit or source verification."""
    command = [
        "git",
        "-C",
        str(target),
        "-c",
        "http.version=HTTP/1.1",
        "fetch",
        "--depth=1",
        "--no-tags",
        "origin",
        commit,
    ]
    for attempt in range(1, 4):
        print(f"Fetching {target.name} @ {commit} (attempt {attempt}/3)", flush=True)
        try:
            subprocess.check_call(command)
            return
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            delay = 5 * attempt
            print(f"Fetch failed; retrying in {delay}s", flush=True)
            time.sleep(delay)


def source_commit(target):
    """Return the upstream pin for a verified archive baseline or a Git checkout."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(target), *args], text=True).strip()

    head = git("rev-parse", "HEAD")
    baseline = subprocess.run(
        ["git", "-C", str(target), "config", "--get", "ascend.archiveHead"], capture_output=True, text=True
    )
    if baseline.returncode == 0 and baseline.stdout.strip() == head:
        return git("config", "--get", "ascend.upstreamCommit")
    return head


def fetch_archive(target, source):
    """Download the pinned public snapshot; final inventory verification is mandatory."""
    commit = source["commit"]
    repository = re.fullmatch(r"https://github.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?", source["repository"])
    if repository is None or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Archive transport requires a public GitHub repository and a full commit")
    owner, name = repository.groups()
    url = f"https://codeload.github.com/{owner}/{name}/tar.gz/{commit}"
    print(f"Downloading {target.name} @ {commit} from {url}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "source.tar.gz"
        subprocess.check_call(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--http1.1",
                "--connect-timeout",
                "20",
                "--max-time",
                "600",
                "--retry",
                "3",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--output",
                str(archive),
                url,
            ]
        )
        prefix = f"{name}-{commit}"
        symlinks = []
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                parts = member.name.split("/")
                if parts[0] != prefix or any(p in ("..", ".git") for p in parts):
                    raise RuntimeError(f"Unexpected archive path: {member.name}")
                destination = target.joinpath(*parts[1:])
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    destination.chmod(0o755 if member.mode & 0o111 else 0o644)
                elif member.issym():
                    link = Path(member.linkname)
                    if link.is_absolute() or not (destination.parent / link).resolve().is_relative_to(target.resolve()):
                        raise RuntimeError(f"Unsafe archive symlink: {member.name}")
                    symlinks.append((destination, member.linkname))
                else:
                    raise RuntimeError(f"Unsupported archive entry: {member.name}")
        # Create links last so extraction never writes through an archive symlink.
        for destination, link in symlinks:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(link)


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
        transport = os.environ.get("FRAMEWORK_SOURCE_TRANSPORT", "git")
        if name == "Megatron-LM":
            transport = os.environ.get("MEGATRON_SOURCE_TRANSPORT") or transport
        if transport == "archive":
            fetch_archive(target, source)
            # Keep a real local baseline/index for editable installs, SCM versioning,
            # patch inspection and nested gitlink verification; do not fake upstream HEAD.
            git("add", "--all", "--force")
            for link, commit in source.get("gitlinks", {}).items():
                (target / link).mkdir(parents=True, exist_ok=True)
                git("update-index", "--add", "--cacheinfo", f"160000,{commit},{link}")
            git(
                "-c",
                "user.name=Source Archive",
                "-c",
                "user.email=archive@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                f"Archive of {source['commit']}",
            )
            git("config", "ascend.archiveHead", git("rev-parse", "HEAD").strip())
            git("config", "ascend.upstreamCommit", source["commit"])
        elif transport == "git":
            fetch_commit(target, source["commit"])
            git("checkout", "--detach", "FETCH_HEAD")
            if git("rev-parse", "HEAD").strip() != source["commit"]:
                raise RuntimeError(f"Unexpected base commit: {name}")
        else:
            raise ValueError(f"Unknown framework source transport: {transport}")
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
