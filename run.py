"""Verify the small release or render from its aggregate figure inputs."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    for name, info in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise SystemExit('Missing or changed release file: ' + name)
    print(json.dumps({'status': 'pass', 'files_checked': len(manifest['files']),
                      'scope': 'Released file integrity; no scientific recalculation'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'display'])
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    verify()
    if args.command == 'display':
        if args.output_root is None:
            parser.error('display requires --output-root')
        destination = args.output_root.resolve()
        if destination.exists() or destination == ROOT or destination.is_relative_to(ROOT):
            parser.error('Use a new output folder outside the source release')
        shutil.copytree(ROOT, destination)
        subprocess.run([sys.executable, str(destination / 'scripts/build_presentation_assets.py')],
                       cwd=destination, check=True)


if __name__ == '__main__':
    main()
