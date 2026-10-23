#!/usr/bin/env python3
"""Restore hash-verified archived KPlaceNet checkpoints; never starts training."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, tarfile, tempfile, threading, urllib.request
BASE='https://github.com/Prajwal-k-tech/KPlaceNet/releases/download/offload-recovery-20261023/'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def download(url,path,expected=None):
 with urllib.request.urlopen(url,timeout=120) as response,path.open('wb') as output:shutil.copyfileobj(response,output)
 if expected and sha(path)!=expected:raise RuntimeError('SHA256 mismatch: '+path.name)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',choices=['integrated','evaluation'],default='integrated');p.add_argument('--output-root',type=Path,default=Path('.'));p.add_argument('--only',action='append');p.add_argument('--plan-only',action='store_true');p.add_argument('--offline-dir',type=Path);args=p.parse_args();root=args.output_root.resolve()
 with tempfile.TemporaryDirectory(prefix='kplacenet-restore-') as temporary:
  temp=Path(temporary);metadata=args.offline_dir or temp
  if not args.offline_dir:
   download(BASE+'models-manifest.json',metadata/'models-manifest.json');download(BASE+'model-bundle-parts.json',metadata/'model-bundle-parts.json')
  manifest=json.loads((metadata/'models-manifest.json').read_text());parts=json.loads((metadata/'model-bundle-parts.json').read_text())['parts'];rows=[r for r in manifest['checkpoints'] if r['workspace']==args.workspace and (not args.only or r['path'] in args.only)]
  if args.only and set(args.only)-{r['path'] for r in rows}:raise RuntimeError('Requested path absent from manifest')
  print('Selected',len(rows),'files;',sum(r['bytes'] for r in rows),'uncompressed bytes')
  if args.plan_only:return
  missing=[]
  for row in rows:
   rel=Path(row['path']);dest=root/rel
   if rel.is_absolute() or '..' in rel.parts or not dest.resolve().is_relative_to(root):raise RuntimeError('Unsafe output path')
   if dest.exists():
    if sha(dest)!=row['sha256']:raise RuntimeError('Refusing to overwrite different existing file: '+row['path'])
   else:missing.append(row)
  for row in [r for r in missing if r['remote']]:
   file=temp/row['sha256'];download(row['remote']['url'],file,row['sha256']);dest=root/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
  wanted={r['sha256'] for r in missing if not r['remote']};objects=temp/'objects';objects.mkdir()
  if wanted:
   if not shutil.which('zstd'):raise RuntimeError('Install zstd before restoring the research bundle')
   files=[]
   for part in parts:
    file=metadata/part['name']
    if args.offline_dir:
     if sha(file)!=part['sha256']:raise RuntimeError('Part SHA256 mismatch: '+file.name)
    else:download(BASE+part['name'],file,part['sha256'])
    files.append(file)
   decoder=subprocess.Popen(['zstd','-dc','--long=29'],stdin=subprocess.PIPE,stdout=subprocess.PIPE);errors=[]
   def feed():
    try:
     for file in files:
      with file.open('rb') as f:shutil.copyfileobj(f,decoder.stdin)
     decoder.stdin.close()
    except BaseException as error:errors.append(error)
   thread=threading.Thread(target=feed,daemon=True);thread.start();found=set()
   try:
    with tarfile.open(fileobj=decoder.stdout,mode='r|') as tar:
     for member in tar:
      if member.name.startswith('objects/') and member.name[8:] in wanted and member.isfile():
       key=member.name[8:];file=objects/key
       with tar.extractfile(member) as source,file.open('wb') as target:shutil.copyfileobj(source,target)
       if sha(file)!=key:raise RuntimeError('Object SHA256 mismatch')
       found.add(key)
    if decoder.wait()!=0:raise RuntimeError('zstd decode failed')
    thread.join()
    if errors:raise errors[0]
    if found!=wanted:raise RuntimeError('Missing objects from bundle')
   except BaseException:
    decoder.kill();decoder.wait();raise
   for row in [r for r in missing if not r['remote']]:
    dest=root/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(objects/row['sha256'],dest)
  print('Restored',len(missing),'files; existing matching files kept. No training started.')
if __name__=='__main__':main()
