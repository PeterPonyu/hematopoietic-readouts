#!/usr/bin/env python3
"""Check the extracted scientific supplements without executing an analysis.

Extract Supplementary Code 1 and every numbered Supplementary Data ZIP into the
same directory, preserving their common project folder. This standard-library
command checks its closed inventory, sizes and SHA-256 bindings. It neither
downloads files nor establishes scientific or biological correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(root):
    root = root.resolve()
    record = json.loads((root / 'SUPPLEMENTARY_MANIFEST.json').read_text())
    if record.get('schema_version') != 1:
        raise ValueError('Unknown scientific-supplement manifest schema')
    expected, errors = set(), []
    for item in record['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in expected:
            raise ValueError('Unsafe or repeated manifest path: ' + str(relative))
        expected.add(str(relative))
        target = root / relative
        if not target.is_file():
            errors.append(f'Missing {relative}; extract {item["archive"]}')
            continue
        if any(part.is_symlink() for part in (target, *target.parents) if part != root):
            errors.append('Symlink not permitted: ' + str(relative))
        elif not target.resolve().is_relative_to(root):
            errors.append('File escapes project: ' + str(relative))
        elif target.stat().st_size != item['bytes'] or sha256(target) != item['sha256']:
            errors.append('Content differs: ' + str(relative))
    actual = {str(path.relative_to(root)) for path in root.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    extra = actual - expected - {'SUPPLEMENTARY_MANIFEST.json'}
    errors.extend('Unlisted file: ' + name for name in sorted(extra))
    if errors:
        raise ValueError('\n'.join(errors))
    return {'status': 'pass', 'files_checked': len(expected),
            'data_archives': record['data_archives'],
            'scope': 'Complete extracted-file inventory, size and checksum agreement; '
                     'scientific execution is a separate step.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == '__main__':
    main()
