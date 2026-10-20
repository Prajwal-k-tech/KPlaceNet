"""Install the L6 model and its matching training-only L7 retrieval bank."""
import json
from pathlib import Path
from download_weights import download
from download_kerala_l6 import main as install_l6
ROOT=Path(__file__).resolve().parents[1]
def main():
    install_l6()
    manifest=json.loads((ROOT/'docs/results/l7_kerala_knn_20261020/deployment.json').read_text())
    for asset,record in manifest['assets'].items():
        download(asset,ROOT/record['destination'],base_url=manifest['release'].replace('/tag/','/download/'),sha256=record['sha256'])
    print('L7 training-photo retrieval installed. Restart the demo; uncheck KNN to compare ResNet alone.')
if __name__=='__main__':main()
