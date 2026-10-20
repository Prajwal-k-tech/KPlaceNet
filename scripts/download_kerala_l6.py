"""Install the measured L6 Kerala pack from Prajwal's fork release, with SHA-256 checks."""
import json
from pathlib import Path
from download_weights import download
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'docs/results/l6_kerala_20261020/deployment.json'

def main():
    manifest=json.loads(MANIFEST.read_text())
    base=manifest['release'].replace('/tag/','/download/')
    for asset,record in manifest['assets'].items():
        download(asset,ROOT/record['destination'],base_url=base,sha256=record['sha256'])
    print('L6 Kerala installed. Restart the demo to select the improved regional model.')

if __name__=='__main__':main()
