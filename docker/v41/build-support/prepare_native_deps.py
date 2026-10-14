"""Stage Dockerfile's local native dependencies without network access."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

CATLASS_COMMIT = '41bf90da655bba3c66d0acd7e00abe33960ecfd6'
ARCHIVES = {
    'catlass.tar.gz': 'f21973a067b8eeec94075e69b88f5f06b6b3900a5f694426af0ac793bdbe243e',
    'catlass_git.tar.gz': '6583d33920c62da404f83635fb4beec636f4be0aea62ec9b460a3bed3b0ecb15',
    'makeself-release-2.5.0-patch1.tar.gz': 'bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a',
    'include.zip': 'a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d',
    'abseil-cpp-20230802.1.tar.gz': '987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed',
    'protobuf-25.1.tar.gz': '9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42',
}

def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    return root.joinpath(*rel.parts)


def untar(archive, root):
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            dest = safe_path(root, member.name)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, dest.open('wb') as out:
                    shutil.copyfileobj(src, out)
                dest.chmod(member.mode & 0o777)
            else:
                raise ValueError(f'Unsupported archive member: {member.name}')


# ExternalProject replaces SOURCE_DIR, so receipts must live outside it.
CMAKE_DEPS = {
    'abseil-cpp': ('abseil-cpp-20230802.1', 'protobuf-hide_absl_symbols.patch'),
    'ascend_protobuf': ('protobuf-25.1', 'protobuf_25.1_change_version.patch'),
}

def validate_cmake_source(src, dst, ascend):
    """Accept original or upstream-patched files; preserve generated outputs."""
    version_dir, patch_name = CMAKE_DEPS[dst.name]
    original = src / version_dir
    original_hashes = {p.relative_to(original): hashlib.sha256(p.read_bytes()).digest()
                       for p in original.rglob('*') if p.is_file()}
    patch = ascend / 'csrc/cmake/third_party/build/modules/patch' / patch_name
    subprocess.run(['patch', '-p1', '--batch'], cwd=original,
                   input=patch.read_bytes().replace(b'\r\n', b'\n'),
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    for rel, digest in original_hashes.items():
        current = dst / rel
        patched = original / rel
        allowed = {digest, hashlib.sha256(patched.read_bytes()).digest()}
        if (current.is_symlink() or not current.is_file()
                or hashlib.sha256(current.read_bytes()).digest() not in allowed):
            raise ValueError(f'Existing CMake dependency differs from archive/upstream patch: {current}')


def prepare(root):
    root = root.resolve()
    workspace = root.parent.parent
    for name, digest in ARCHIVES.items():
        path = workspace / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Missing or changed offline archive: {path}')
    ascend = Path(os.environ.get('VLLM_ASCEND_ROOT', workspace.parent / 'vllm-ascend'))
    config = ascend / '.gitmodules'
    if CATLASS_COMMIT not in config.read_text():
        raise ValueError('Ascend Catlass pin differs; re-audit archives before building.')
    target = ascend / 'csrc/third_party'
    target.mkdir(parents=True, exist_ok=True)
    # Prepare completely before publishing; retain existing native build outputs.
    (root / 'build-src').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.native-deps-', dir=root / 'build-src') as tmp:
        temp = Path(tmp)
        for name in ('catlass.tar.gz', 'catlass_git.tar.gz', 'makeself-release-2.5.0-patch1.tar.gz'):
            untar(workspace / name, temp)
        (temp / 'makeself-release-2.5.0').rename(temp / 'makeself')
        for directory, archive in [('abseil-cpp', 'abseil-cpp-20230802.1.tar.gz'), ('ascend_protobuf', 'protobuf-25.1.tar.gz')]:
            untar(workspace / archive, temp / directory)
        with zipfile.ZipFile(workspace / 'include.zip') as zf:
            for item in zf.infolist():
                dest = safe_path(temp / 'json', item.filename)
                if item.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(item))
        marker_text = json.dumps(ARCHIVES, sort_keys=True) + '\n'
        for directory in ('catlass', 'makeself', 'json', 'abseil-cpp', 'ascend_protobuf'):
            src, dst = temp / directory, target / directory
            marker = dst / '.dsv41-offline-archives.json'
            receipt = target / ('.dsv41-' + directory + '-archives.json')
            # A non-recursive Git checkout creates empty submodule placeholders.
            # Reject links (including dangling links) before inspecting/removing a directory.
            if dst.is_symlink():
                raise ValueError(f'Dependency must not be a symlink: {dst}')
            if receipt.exists() and receipt.read_text() != marker_text:
                raise ValueError(f'Dependency receipt differs: {receipt}')
            if dst.exists() and not dst.is_dir():
                raise ValueError(f'Dependency must be a directory: {dst}')
            if dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()  # Only empty directories; never remove existing dependency contents.
            if dst.exists():
                if not receipt.exists():
                    if marker.is_file() and marker.read_text() == marker_text:
                        pass  # Migrate original per-directory receipt.
                    elif directory in CMAKE_DEPS:
                        validate_cmake_source(src, dst, ascend)
                    else:
                        raise ValueError(f'Existing dependency is unmanaged: {dst}; preserve it and inspect before retrying.')
            else:
                (src / marker.name).write_text(marker_text)
                src.rename(dst)
            receipt.write_text(marker_text)
        # CMake also consumes original archives through CANN_3RD_PKG_PATH.
        pkg = target / 'pkg'
        pkg.mkdir(exist_ok=True)
        for name in ARCHIVES:
            dst = pkg / name
            if not dst.exists():
                shutil.copy2(workspace / name, dst)
            elif hashlib.sha256(dst.read_bytes()).hexdigest() != ARCHIVES[name]:
                raise ValueError(f'Changed build archive: {dst}')
    if not (target / 'catlass/include').is_dir():
        raise ValueError('Catlass include directory missing.')
    return {'status': 'prepared', 'scope': 'offline_native_dependencies_not_build_acceptance',
            'catlass_commit': CATLASS_COMMIT, 'archives_checked': len(ARCHIVES), 'path': str(target)}

if __name__ == '__main__':
    print(json.dumps(prepare(Path(__file__).resolve().parent), indent=2))
