"""Build or verify the frozen source-domain split without scoring any model."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '52db0107dfb3ea98c581509af3422594148864b9'
MANIFEST = ROOT/'docs/GATE4B_MANIFEST.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shingles(data):
    words = re.findall(rb'[a-z0-9_]+', data.lower())
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}


def build():
    files = git('ls-tree','-r','--name-only',BASE).decode().splitlines()
    paths = sorted(p for p in files if p.startswith('nexus_expanse/') and p.endswith('.py'))
    paths += sorted(p for p in files if p.endswith('.md'))
    blocks, hashes, grams, words = [], set(), set(), []
    excluded = []
    for p in paths:
        data = git('show',BASE+':'+p)
        digest = sha(data)
        if digest in hashes:
            raise ValueError('duplicate full file')
        hashes.add(digest)
        split = 'train' if p.endswith('.py') else 'dev'
        spans=[]
        for start in range(0,len(data)-255,256):
            chunk=data[start:start+256]
            literals={chunk[i:i+32] for i in range(225)}
            normalized=shingles(chunk)
            near=any(len(normalized & old)/max(1,len(normalized | old))>=.35 for old in words)
            if literals & grams or near:
                excluded.append([p,start])
                continue
            spans.append(dict(start=start,bytes=256,sha256=sha(chunk)))
            grams.update(literals)
            words.append(normalized)
            if len(spans)==(4 if split=='train' else 2):
                break
        blocks.append(dict(path=p,split=split,file_bytes=len(data),sha256=digest,spans=spans))
    return dict(base=BASE,files=blocks,rejected_spans=excluded)


def read_split():
    manifest=json.loads(MANIFEST.read_text())
    if build()!=manifest:
        raise ValueError('frozen manifest mismatch')
    result={'train':[],'dev':[]}
    for file in manifest['files']:
        data=git('show',BASE+':'+file['path'])
        for span in file['spans']:
            start=span['start']
            result[file['split']].append(data[start:start+span['bytes']])
    return result


if __name__=='__main__':
    with MANIFEST.open('x') as out:
        json.dump(build(),out,indent=2)
        out.write('\n')
