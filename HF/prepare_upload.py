"""Verify linked HF payloads or copy one into a new upload directory. Never uploads."""
import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

HF = Path(__file__).resolve().parent


def safe_path(root, relative):
    """Manifest entries may name payload files, never absolute paths or parents."""
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or 'bloatware-donotopen' in parts.parts:
        raise ValueError(f'invalid payload path: {relative}')
    return root.joinpath(*parts.parts)


def verify(hf, entry):
    directory = safe_path(hf, entry['directory'])
    for relative, expected in entry['files'].items():
        source = safe_path(directory, relative)
        if 'bloatware-donotopen' in source.resolve().parts:
            raise ValueError('excluded payload target')
        # Inspect only the manifest-listed file, never discover additional lab data.
        if not source.is_file():
            raise ValueError(f'missing payload file: {source}; see HF/README.md for snapshot downloads')
        if source.stat().st_size != expected['bytes']:
            raise ValueError(f'byte-size mismatch: {source}')
    return directory


def materialize(hf, entry, destination):
    if 'bloatware-donotopen' in destination.absolute().parts:
        raise ValueError('excluded destination')
    if destination.exists():
        raise ValueError('destination already exists; choose a new directory')
    source = verify(hf, entry)
    if shutil.disk_usage(destination.parent).free < sum(f['bytes'] for f in entry['files'].values()):
        raise ValueError('destination does not have enough free space')
    destination.mkdir()
    for relative, expected in entry['files'].items():
        target = safe_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(source, relative), target, follow_symlinks=True)
        if target.stat().st_size != expected['bytes']:
            raise ValueError(f'copied file size mismatch: {target}')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--payload', help='one label from manifest.json')
    parser.add_argument('--manifest', type=Path, default=HF / 'manifest.json', help='payload inventory; defaults to the published artifacts')
    parser.add_argument('--materialize', type=Path, help='new directory for one payload, dereferencing links')
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())['payloads']
    if args.payload and args.payload not in entries:
        parser.error(f'choose a payload from {", ".join(entries)}')
    if args.materialize and not args.payload:
        parser.error('--materialize needs --payload')
    if not args.check and args.materialize is None:
        parser.error('choose --check or --materialize')
    selected = {args.payload: entries[args.payload]} if args.payload else entries
    try:
        if args.check:
            for label, entry in selected.items():
                verify(HF, entry)
                print(f'ok {label}: {len(entry["files"])} files; sizes match')
        if args.materialize:
            materialize(HF, entries[args.payload], args.materialize)
            print(f'prepared {args.materialize}; no upload performed')
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
