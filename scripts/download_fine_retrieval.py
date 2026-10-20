"""Install fingerprinted L10 fine retrieval with pinned, local DINOv2 source."""
import json,zipfile,shutil,tempfile
from pathlib import Path
from download_weights import download
from download_kerala_l7 import main as install_l7
ROOT=Path(__file__).resolve().parents[1]
def main():
 install_l7()
 m=json.loads((ROOT/'docs/results/fine_deployment_20261021/deployment.json').read_text())
 for name,a in m['assets'].items():download(name,ROOT/a['destination'],base_url=m['release'].replace('/tag/','/download/'),sha256=a['sha256'])
 url=m['dino_url'];base,name=url.rsplit('/',1)
 folder=ROOT/'checkpoints/fine_retrieval'
 download(name,folder/name,base_url=base,sha256=m['dino_sha256'])
 with tempfile.TemporaryDirectory(dir=folder) as tmp:
  with zipfile.ZipFile(folder/'dinov2-source.zip') as archive:
   for info in archive.infolist():
    target=(Path(tmp)/info.filename).resolve()
    if not target.is_relative_to(Path(tmp).resolve()):raise ValueError('Unsafe source archive path')
   archive.extractall(tmp)
  source=Path(tmp)/('dinov2-'+m['source_commit'])
  if not (source/'hubconf.py').is_file():raise ValueError('Pinned DINO source is incomplete')
  shutil.copytree(source,folder/'dinov2_source',dirs_exist_ok=True)
 print('Fine coordinates installed. Restart the frontend. References are training-only; similarity is not confidence.')
if __name__=='__main__':main()
