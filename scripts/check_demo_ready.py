"""Check local demo files without loading weights or downloading anything."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(mode: str) -> dict:
    checks = []

    def file(label, path, required=True):
        path = Path(path).expanduser()
        exists = path.is_file()
        checks.append(dict(label=label, path=str(path), required=required, exists=exists))
        return path, exists

    if mode in {'global', 'all'}:
        checkpoint, present = file('Global checkpoint', os.environ.get('KPLACENET_CHECKPOINT',
            ROOT / 'checkpoints/l2_imagenet_1.0_layer4/last.pt'))
        calibration, cal_present = file('Global calibration (optional)', os.environ.get(
            'KPLACENET_CALIBRATION', ROOT / 'checkpoints/l3_uncertainty_results.json'), False)
        file('Global full cell bounds (optional)', os.environ.get('KPLACENET_CELLS',
            ROOT / 'checkpoints/l2_cells_full.json'), False)
        if cal_present:
            from src.demo_config import load_calibration, sha256_file, verify_calibration_checkpoint
            try:
                config = load_calibration(calibration)
                if present:
                    verified = verify_calibration_checkpoint(config, sha256_file(checkpoint))
                    checks.append(dict(label='Calibration checkpoint identity', required=True,
                        exists=True, detail='hash verified' if verified else 'no hash in artifact; unverified'))
            except ValueError as error:
                checks.append(dict(label='Calibration validity', required=True, exists=False,
                                   detail=str(error)))
    if mode in {'kerala', 'all'}:
        from src.demo_config import kerala_artifact_paths
        regional_checkpoint, regional_cells = kerala_artifact_paths(ROOT)
        file('Kerala checkpoint', regional_checkpoint)
        file('Kerala cell map', regional_cells)
    if mode == 'game':
        file('Game global checkpoint', ROOT / 'checkpoints/l2_imagenet_1.0_layer4/last.pt')
        file('Game metadata', ROOT / 'data/osv5m_test/metadata.csv')
        file('Game calibration (optional)', ROOT / 'checkpoints/l3_uncertainty_results.json', False)
    examples = {
        'kerala': len(list((ROOT / 'data/kerala_osv5m/test/images').glob('*.jpg'))),
        'global': len(list((ROOT / 'data/osv5m_test/images').glob('*.jpg'))),
    }
    return dict(mode=mode, ready_for_file_loading=all(c['exists'] for c in checks if c['required']),
        checks=checks, example_image_counts=examples,
        limitation='File presence only, plus available global calibration checks. Model state, '
                   'cell compatibility, image decoding, and inference must still be checked in the demo.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['global', 'kerala', 'game', 'all'], default='all')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = check(args.mode)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for item in result['checks']:
            print(f"[{'FOUND' if item['exists'] else 'MISSING'}] {item['label']}")
            print('  ' + item.get('path', item.get('detail', '')))
        print('Example image counts:', result['example_image_counts'])
        print(result['limitation'])
        print('Required files present:', result['ready_for_file_loading'])
    return 0 if result['ready_for_file_loading'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
