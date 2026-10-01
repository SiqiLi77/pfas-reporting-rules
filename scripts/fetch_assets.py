#!/usr/bin/env python3
"""Download/check the tagged GitHub assets and safely expand their relative paths."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-directory',type=Path,help='Use already downloaded release ZIPs')
    p.add_argument('--asset',action='append',help='Select an asset name; default: all')
    p.add_argument('--verify-only',action='store_true',help='Verify already expanded data')
    args=p.parse_args()
    manifest=json.loads((ROOT/'reproducibility/ASSETS.json').read_text())
    wanted=set(args.asset or [a['name'] for a in manifest['assets']])
    valid={a['name'] for a in manifest['assets']}
    if not wanted<=valid:p.error('Unknown asset: '+', '.join(wanted-valid))
    files=list(csv.DictReader((ROOT/'reproducibility/ASSET_FILES.csv').open()))
    cache=ROOT/'.release-cache';cache.mkdir(exist_ok=True)
    public_assets={}
    if not args.from_directory and not args.verify_only:
        request=urllib.request.Request('https://api.github.com/repos/SiqiLi77/pfas-reporting-rules/releases/tags/'+manifest['release'],headers={'User-Agent':'pfas-reporting-rules-reproduction','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(request,timeout=30) as response:
            release=json.load(response)
        public_assets={a['name']:a for a in release['assets']}
    for asset in manifest['assets']:
        if asset['name'] not in wanted:continue
        if not args.verify_only:
            archive=(args.from_directory or cache)/asset['name']
            if not archive.exists() or sha(archive)!=asset['sha256']:
                if args.from_directory:raise ValueError('Local ZIP missing or checksum mismatch: '+asset['name'])
                partial=archive.with_suffix('.download')
                try:
                    if asset['name'] not in public_assets:raise ValueError('Asset absent from tagged public release')
                    entry=public_assets[asset['name']]
                    if entry['size']!=asset['bytes']:raise ValueError('Published asset size differs from manifest')
                    # Public official API download; no token or account required.
                    request=urllib.request.Request(entry['url'],headers={'User-Agent':'pfas-reporting-rules-reproduction','Accept':'application/octet-stream'})
                    with urllib.request.urlopen(request,timeout=120) as r,partial.open('wb') as f:shutil.copyfileobj(r,f)
                    if sha(partial)!=asset['sha256']:raise ValueError('Downloaded ZIP checksum mismatch')
                    os.replace(partial,archive)
                finally:
                    if partial.exists():partial.unlink()
            expected={r['path']:r for r in files if r['asset']==asset['name']}
            with zipfile.ZipFile(archive) as z:
                if set(z.namelist())!=set(expected):raise ValueError('Unexpected archive members')
                for member in z.infolist():
                    target=(ROOT/member.filename).resolve()
                    if not target.is_relative_to(ROOT) or member.filename.startswith(('/', '\\')):raise ValueError('Unsafe member path')
                    if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Symbolic link not allowed')
                    if target.exists():
                        if sha(target)!=expected[member.filename]['sha256']:raise ValueError('Will not overwrite changed data: '+member.filename)
                        continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(member) as r,target.open('wb') as f:shutil.copyfileobj(r,f)
        for record in files:
            if record['asset']!=asset['name']:continue
            path=ROOT/record['path']
            if not path.exists() or sha(path)!=record['sha256']:raise ValueError('File checksum mismatch: '+record['path'])
        print('Verified:',asset['name'],flush=True)

if __name__=='__main__':main()
