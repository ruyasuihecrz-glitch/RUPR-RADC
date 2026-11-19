"""Fetch only the learned method artifacts; base models are separate resources."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo-id',default='DavideChen/RUPR-RADC')
    p.add_argument('--revision');p.add_argument('--output',default='artifacts')
    p.add_argument('--recipe',default='configs/locked_v1.json');a=p.parse_args()
    dest=Path(a.output).resolve()
    os.environ.setdefault('HF_HOME',str(dest.parent/'.cache/huggingface'))
    from huggingface_hub import snapshot_download
    recipe=json.loads(Path(a.recipe).read_text())
    revision=a.revision or recipe.get('weights_revision','v1.0.0')
    snapshot_download(a.repo_id,revision=revision,local_dir=dest,allow_patterns=list(recipe['artifacts']))
    for name,expected in recipe['artifacts'].items():
        if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Checksum mismatch: '+name)
    print('All learned artifacts match the locked release recipe.')


if __name__=='__main__':main()
