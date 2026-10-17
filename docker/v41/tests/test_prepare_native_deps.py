"""Offline regression tests for Git submodule placeholders and native dependency receipts."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

script = Path(__file__).resolve().parents[1] / 'build-support/prepare_native_deps.py'
spec = importlib.util.spec_from_file_location('native_deps', script)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class NativeDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / 'workspace'
        self.root = self.workspace / 'rollout/dsv41'
        self.root.mkdir(parents=True)
        self.ascend = self.workspace.parent / 'vllm-ascend'
        self.target = self.ascend / 'csrc/third_party'
        self.target.mkdir(parents=True)
        (self.ascend / '.gitmodules').write_text('commit = ' + native.CATLASS_COMMIT)
        contents = {
            'catlass.tar.gz': 'catlass/include/example.h',
            'catlass_git.tar.gz': 'catlass/.git/HEAD',
            'makeself-release-2.5.0-patch1.tar.gz': 'makeself-release-2.5.0/makeself.sh',
            'abseil-cpp-20230802.1.tar.gz': 'abseil-cpp-20230802.1/README',
            'protobuf-25.1.tar.gz': 'protobuf-25.1/README',
        }
        for name, member in contents.items():
            with tarfile.open(self.workspace / name, 'w:gz') as archive:
                info = tarfile.TarInfo(member)
                info.size = 4
                archive.addfile(info, io.BytesIO(b'test'))
        with zipfile.ZipFile(self.workspace / 'include.zip', 'w') as archive:
            archive.writestr('include/nlohmann/json.hpp', 'test')
        hashes = {name: hashlib.sha256((self.workspace / name).read_bytes()).hexdigest()
                  for name in native.ARCHIVES}
        patched = patch.dict(native.ARCHIVES, hashes, clear=True)
        patched.start()
        self.addCleanup(patched.stop)

    def test_empty_submodule_placeholder_and_repeat(self):
        (self.target / 'catlass').mkdir()
        self.assertEqual(native.prepare(self.root)['status'], 'prepared')
        self.assertEqual((self.target / 'catlass/include/example.h').read_bytes(), b'test')
        self.assertTrue((self.target / '.dsv41-catlass-archives.json').is_file())
        self.assertEqual(native.prepare(self.root)['status'], 'prepared')

    def test_nonempty_unmanaged_directory_is_preserved(self):
        catlass = self.target / 'catlass'
        catlass.mkdir()
        (catlass / 'keep.txt').write_text('local changes')
        with self.assertRaisesRegex(ValueError, 'unmanaged'):
            native.prepare(self.root)
        self.assertEqual((catlass / 'keep.txt').read_text(), 'local changes')
        self.assertFalse((self.target / '.dsv41-catlass-archives.json').exists())

    def test_symlinks_including_dangling_links_are_rejected(self):
        catlass = self.target / 'catlass'
        for exists in (False, True):
            outside = Path(self.temp.name) / ('existing' if exists else 'absent')
            if exists:
                outside.mkdir()
            catlass.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                native.prepare(self.root)
            self.assertTrue(catlass.is_symlink())
            catlass.unlink()

    def test_conflicting_receipt_does_not_remove_empty_directory(self):
        (self.target / 'catlass').mkdir()
        (self.target / '.dsv41-catlass-archives.json').write_text('different archives')
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            native.prepare(self.root)
        self.assertTrue((self.target / 'catlass').is_dir())

    def test_file_at_dependency_path_is_preserved(self):
        (self.target / 'catlass').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'must be a directory'):
            native.prepare(self.root)
        self.assertEqual((self.target / 'catlass').read_text(), 'keep')

    def test_changed_archive_is_rejected_before_placeholder_removal(self):
        (self.target / 'catlass').mkdir()
        (self.workspace / 'catlass.tar.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed'):
            native.prepare(self.root)
        self.assertTrue((self.target / 'catlass').is_dir())


if __name__ == '__main__':
    unittest.main()
