#!/usr/bin/env bash
# Python source and SHA-256: https://www.python.org/downloads/release/python-31115/
set -euo pipefail
prefix=/usr/local/python3.11.15
build_dir="$(mktemp -d /tmp/training-python.XXXXXX)"
trap 'rm -rf "$build_dir"' EXIT
cd "$build_dir"
curl --fail --location --retry 5 --retry-delay 3 \
    https://www.python.org/ftp/python/3.11.15/Python-3.11.15.tar.xz -o python.tar.xz
printf '%s  %s\n' 272179ddd9a2e41a0fc8e42e33dfbdca0b3711aa5abf372d3f2d51543d09b625 python.tar.xz | sha256sum --check -
tar -xf python.tar.xz
cd Python-3.11.15
./configure --prefix="$prefix" --enable-shared --with-ensurepip=install
make -j "${PYTHON_BUILD_JOBS:-4}"
make altinstall
printf '%s\n' "$prefix/lib" > /etc/ld.so.conf.d/training-python.conf
ldconfig
ln -s python3.11 "$prefix/bin/python3"
ln -s python3.11 "$prefix/bin/python"
ln -s pip3.11 "$prefix/bin/pip3"
ln -s pip3.11 "$prefix/bin/pip"
"$prefix/bin/python3" -I - <<'PY_CHECK'
import bz2
import ctypes
import lzma
import sqlite3
import ssl
import sys
import sysconfig
import zlib
from pathlib import Path
assert sys.version_info[:3] == (3, 11, 15), sys.version
assert (Path(sysconfig.get_path("include")) / "Python.h").is_file()
print(sys.version)
PY_CHECK
