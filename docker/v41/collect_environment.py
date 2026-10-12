"""Collect versions without device imports, credentials, environment variables or package URLs."""
import importlib.metadata as metadata
import json
import platform
import sys

packages = sorted(
    ({"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
     for dist in metadata.distributions()),
    key=lambda item: item["name"].lower(),
)
print(json.dumps({
    "python": sys.version,
    "architecture": platform.machine(),
    "libc": platform.libc_ver(),
    "packages": packages,
}, indent=2))
