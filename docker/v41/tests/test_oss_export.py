"""CPU tests for local rootfs export and temporary-container lifecycle."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

script = Path(__file__).resolve().parents[1] / "export_oss_image.py"
spec = importlib.util.spec_from_file_location("oss_export", script)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "Id": "sha256:" + "1" * 64, "Os": "linux", "Architecture": "arm64",
            "Config": {
                "Env": ['PATH=/usr/bin:/bin', 'LITERAL=$HOME "quoted"', 'EMPTY='],
                "Entrypoint": ["/bin/bash", "/usr/local/bin/v41-entrypoint"],
                "Cmd": ["/bin/bash"], "WorkingDir": "/workspace/Ascend-Training-Recipes",
                "User": "root", "Shell": ["/bin/bash", "-c"],
                "Volumes": {"/data": {}}, "ExposedPorts": {"8000/tcp": {}},
                "Labels": {"org.example.source": "recipe"}, "StopSignal": "SIGTERM",
                "Healthcheck": {"Test": ["CMD", "true"], "Interval": 1000000000},
            },
        }

    def test_invalid_platform_is_rejected_before_container_creation(self):
        for key, value in (("Architecture", "amd64"), ("Os", "windows")):
            info = copy.deepcopy(self.info)
            info[key] = value
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "out"
                with patch.object(exporter, "docker", return_value=json.dumps([info])) as docker:
                    with self.assertRaises(ValueError):
                        exporter.export_image("image:test", target)
                self.assertEqual(docker.call_count, 1)
                self.assertFalse(target.exists())

    def test_export_pins_image_id_and_cleans_up(self):
        payload = b"fake filesystem archive for lifecycle test"
        def output(command, **kwargs):
            if command[1:3] == ["image", "inspect"]: return json.dumps([self.info])
            self.assertEqual(command[1:3], ["container", "create"])
            self.assertEqual(command[-1], self.info["Id"])
            return "temporary-container-id\n"
        def run(command, **kwargs):
            if command[1:3] == ["container", "export"]: Path(command[4]).write_bytes(payload)
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-output"
            with patch.object(exporter.subprocess, "check_output", side_effect=output), \
                 patch.object(exporter.subprocess, "run", side_effect=run) as runner:
                exporter.export_image("image:mutable", target, part_size=8)
            parts = sorted((target / "rootfs-parts").iterdir())
            self.assertEqual(b"".join(part.read_bytes() for part in parts), payload)
            self.assertTrue(all(part.stat().st_size == 8 for part in parts[:-1]))
            self.assertLessEqual(parts[-1].stat().st_size, 8)
            self.assertFalse((target / "rootfs.tar").exists())
            self.assertFalse((target / "Dockerfile.oss").exists())
            self.assertEqual((target / "rootfs.tar.sha256").read_text(),
                             hashlib.sha256(payload).hexdigest() + "  rootfs.tar\n")
            report = json.loads((target / "image-config.json").read_text())
            self.assertEqual(report["Config"], self.info["Config"])
            self.assertEqual(report["parts"], [part.name for part in parts])
            self.assertEqual(runner.call_args.args[0], ["docker", "container", "rm", "-v", "temporary-container-id"])
            self.assertFalse((target / "rootfs.tar.partial").exists())

    def test_export_failure_still_removes_only_temporary_container(self):
        def output(command, **kwargs):
            return json.dumps([self.info]) if command[1] == "image" else "temporary-container-id"
        def run(command, **kwargs):
            if command[2] == "export": raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(exporter.subprocess, "check_output", side_effect=output), \
                 patch.object(exporter.subprocess, "run", side_effect=run) as runner:
                with self.assertRaises(subprocess.CalledProcessError):
                    exporter.export_image("image:test", Path(directory) / "out")
            self.assertEqual(runner.call_args.args[0], ["docker", "container", "rm", "-v", "temporary-container-id"])


if __name__ == "__main__":
    unittest.main()
