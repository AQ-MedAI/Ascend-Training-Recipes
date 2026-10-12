"""Fetch checksum-pinned native build inputs, or reuse a verified local cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(entries):
    seen = set()
    for entry in entries:
        path = PurePosixPath(entry['path'])
        sha = entry['sha256']
        if path.is_absolute() or '..' in path.parts or not path.parts or entry['path'] in seen:
            raise ValueError('Invalid or duplicate asset path: ' + entry['path'])
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid SHA256: ' + entry['path'])
        if entry.get('transport', 'https') not in ('https', 'git-blob'):
            raise ValueError('Unsupported transport')
        if entry.get('transport') == 'git-blob':
            commit = entry.get('commit', '')
            source = PurePosixPath(entry.get('source_path', ''))
            if not entry.get('repository', '').startswith('https://'):
                raise ValueError('Git repository must use HTTPS')
            if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
                raise ValueError('Git source requires a full commit SHA')
            if source.is_absolute() or '..' in source.parts or not source.parts:
                raise ValueError('Invalid Git source path')
        if entry.get('url') and not entry['url'].startswith('https://'):
            raise ValueError('Only HTTPS downloads are supported')
        seen.add(entry['path'])


def checked_existing(path, sha):
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or digest(path) != sha:
            raise ValueError('Checksum mismatch or invalid file; inspect/remove before retrying: ' + str(path))
        return True
    return False


def download_git_blob(entry, destination):
    # Never check out or execute third-party repository files. Fetch only the pinned blob.
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    with tempfile.TemporaryDirectory(prefix='v41-git-blob-') as directory:
        def git(*args, **kwargs):
            return subprocess.run(['git', '-C', directory, *args], env=env,
                                  check=True, timeout=180, **kwargs)
        git('init', '--quiet')
        git('remote', 'add', 'origin', entry['repository'])
        git('fetch', '--quiet', '--depth=1', '--filter=blob:none', 'origin', entry['commit'])
        with destination.open('wb') as output:
            git('show', entry['commit'] + ':' + entry['source_path'], stdout=output)


def acquire(entry, workspace, cache, offline):
    target = workspace / entry['path']
    # Do not follow existing directory symlinks outside the requested workspace.
    if not target.resolve().is_relative_to(workspace.resolve()):
        raise ValueError('Target escapes workspace: ' + str(target))
    sha = entry['sha256']
    if checked_existing(target, sha):
        return 'existing'
    cached = cache / sha
    if not checked_existing(cached, sha):
        if offline:
            raise ValueError('Offline cache miss: ' + entry['path'])
        if not entry.get('url') and entry.get('transport') != 'git-blob':
            raise ValueError('No verified public archive URL: ' + entry['path'] + '; ' + entry['unresolved'])
        cache.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            fd, name = tempfile.mkstemp(prefix='.download-', dir=cache)
            os.close(fd)
            temp = Path(name)
            try:
                if entry.get('transport') == 'git-blob':
                    download_git_blob(entry, temp)
                else:
                    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'Ascend-Training-Recipes/1'})
                    with urllib.request.urlopen(request, timeout=30) as response, temp.open('wb') as stream:
                        if not response.geturl().startswith('https://'):
                            raise ValueError('Refusing a non-HTTPS redirect')
                        shutil.copyfileobj(response, stream)
                if digest(temp) != sha:
                    raise ValueError('Downloaded SHA256 differs: ' + entry['path'])
                os.replace(temp, cached)
                break
            except ValueError:
                raise
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
            finally:
                temp.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.asset-', dir=target.parent)
    os.close(fd)
    temp = Path(name)
    try:
        shutil.copyfile(cached, temp)
        if digest(temp) != sha:
            raise ValueError('Cache changed while copying: ' + entry['path'])
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return 'ready'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/ascend-training-recipes')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('native-dependencies.json'))
    parser.add_argument('--group', choices=('all', 'native', 'premix'), default='all')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())['assets']
    validate(entries)
    selected = [e for e in entries if args.group == 'all' or e['group'] == args.group]
    failures = []
    for entry in selected:
        try:
            status = acquire(entry, args.workspace, args.cache, args.offline)
            print(status + ': ' + entry['path'], flush=True)
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            failures.append(str(exc))
            print('FAILED: ' + str(exc), flush=True)
    if args.group in ('all', 'premix'):
        premix = [{k: e[k] for k in ('name', 'ref', 'file', 'sha256', 'url')}
                  for e in entries if e['group'] == 'premix']
        path = args.workspace / 'third_party/mhc-premix/manifest.json'
        if not path.resolve().is_relative_to(args.workspace.resolve()):
            failures.append('Premix manifest escapes workspace')
        elif path.exists() and json.loads(path.read_text()) != premix:
            failures.append('Existing Premix manifest differs; refusing to overwrite')
        elif not failures:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(premix, indent=2) + '\n')
    if failures:
        raise SystemExit(f'{len(failures)} unresolved/failed inputs; verified downloads remain cached.')
    print(f'All {len(selected)} selected assets verified; this does not validate the NPU build.')


if __name__ == '__main__':
    main()
