"""Verify vendored archives and stage a build copy; never access the network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tarfile


def extract_archive(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            rel = Path(*Path(member.name).parts[1:])
            if not rel.parts:
                continue
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archive entry')
            member.name = str(rel)
            tar.extract(member, dest, filter='data')


def ensure_tensor_api(output, vendor, manifest):
    entry = next(e for e in manifest if e['name'] == 'tensor-api')
    dest = output / 'third_party/ops-tensor/include/tensor_api'
    if not (dest / 'impl/tensor_api').is_dir():
        extract_archive(vendor / entry['file'], dest)
    if not (dest / 'impl/tensor_api').is_dir():
        raise ValueError('Missing tensor_api implementation after extraction')


def ensure_abseil_patch(output, vendor):
    # Do not modify the image's shared abseil source symlink.
    for subdir in ('pkg', 'pkg/device'):
        dest = output / 'third_party' / subdir
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vendor / 'backport-CVE-2025-0838.patch',
                        dest / 'backport-CVE-2025-0838.patch')


def ensure_library_fragment(output):
    # This staged framework only builds the additive Premix vendor package.
    # CANN 9.1 owns DEF and may be imported after this package.
    builder = output / 'torch_extension/cann_ops_transformer/op_builder/builder.py'
    text = builder.read_text()
    old = 'Library("cann_ops_transformer", "DEF")'
    new = 'Library("cann_ops_transformer", "FRAGMENT")'
    if old not in text and new not in text:
        raise ValueError('Unrecognized Premix library registration')
    builder.write_text(text.replace(old, new))


def prepare(output):
    root = Path(__file__).resolve().parents[3]
    vendor = root / 'workspace/third_party/mhc-premix'
    manifest = json.loads((vendor / 'manifest.json').read_text())
    for entry in manifest:
        archive = vendor / entry['file']
        if hashlib.sha256(archive.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Archive checksum mismatch: ' + str(archive))
    output = output.resolve()
    marker = output / '.v41-source-manifest.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        old_core = [e for e in previous if e['name'] not in ('tensor-api', 'abseil-cve-patch')]
        new_core = [e for e in manifest if e['name'] not in ('tensor-api', 'abseil-cve-patch')]
        if old_core != new_core or any(e not in manifest for e in previous):
            raise ValueError('Build source differs; select a new V41_MHC_BUILD_ROOT')
        ensure_library_fragment(output)
        ensure_tensor_api(output, vendor, manifest)
        ensure_abseil_patch(output, vendor)
        marker.write_text(json.dumps(manifest, indent=2) + '\n')
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite existing build source: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        if entry['name'] == 'abseil-cve-patch':
            continue
        dest = output if entry['name'] == 'ops-transformer' else output / 'third_party' / entry['name']
        if entry['name'] == 'cmake':
            dest = output / 'third_party/cann-cmake'
        if entry['name'] == 'tensor-api':
            dest = output / 'third_party/ops-tensor/include/tensor_api'
        extract_archive(vendor / entry['file'], dest)
    ensure_library_fragment(output)
    ensure_tensor_api(output, vendor, manifest)
    ensure_abseil_patch(output, vendor)
    # Archives have no .git; replace only dependency resolution, not kernels.
    third = output / 'third_party'
    (output / 'cmake/third_party/opbase.cmake').write_text(
        'set(OPBASE_SOURCE_PATH "${CANN_3RD_LIB_PATH}/opbase")\n')
    (output / 'cmake/third_party/ops-tensor.cmake').write_text(
        'set(OPTENSOR_SOURCE_PATH "${CANN_3RD_LIB_PATH}/ops-tensor")\n'
        'get_filename_component(TENSOR_API "${OPTENSOR_SOURCE_PATH}/include/tensor_api" REALPATH)\n')
    # Reuse the exact offline preparation used by the image Dockerfile.
    subprocess.run([sys.executable, str(root / 'workspace/rollout/dsv41/prepare_native_deps.py')], check=True)
    existing = Path(os.environ.get('VLLM_ASCEND_ROOT', root / 'vllm-ascend')) / 'csrc/third_party'
    for name in ('makeself', 'json', 'abseil-cpp', 'ascend_protobuf'):
        src = existing / name
        if not src.is_dir():
            raise FileNotFoundError(src)
        (third / name).symlink_to(src.resolve(), target_is_directory=True)
    for name in ('abseil-cpp-20230802.1.tar.gz', 'protobuf-25.1.tar.gz',
                 'makeself-release-2.5.0-patch1.tar.gz'):
        (third / name).symlink_to(root / 'workspace' / name)
    # Existing image headers already carry the vendor namespace patch.
    (third / 'cann-cmake/third_party/json.cmake').write_text(
        'add_library(json INTERFACE)\n'
        'set(JSON_INCLUDE_DIR "${CANN_3RD_LIB_PATH}/json/include")\n'
        'target_include_directories(json SYSTEM INTERFACE "${JSON_INCLUDE_DIR}")\n'
        'target_compile_definitions(json INTERFACE nlohmann=ascend_nlohmann)\n')
    marker.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args().output)
