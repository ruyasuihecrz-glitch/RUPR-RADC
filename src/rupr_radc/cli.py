"""Small, explicit commands; model inference and label-based scoring are separate."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='command',required=True)
    f=subs.add_parser('predict-features');f.add_argument('--features',required=True);f.add_argument('--output',required=True)
    f.add_argument('--artifacts',default='artifacts');f.add_argument('--recipe',default='configs/locked_v1.json');f.add_argument('--device',default='cpu')
    raw=subs.add_parser('predict');raw.add_argument('--input',required=True);raw.add_argument('--output',required=True)
    for name in ['reader-path','reranker-path','reranker-helper','clip-path']:raw.add_argument('--'+name,required=True)
    raw.add_argument('--artifacts',default='artifacts');raw.add_argument('--repository',default='.')
    score=subs.add_parser('score');score.add_argument('--predictions',required=True);score.add_argument('--labels',required=True);score.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='predict-features':
        from .api import Calibrator
        rows=Calibrator(a.artifacts,a.recipe,a.device).predict_npz(a.features)
    elif a.command=='predict':
        from .pipeline import Pipeline
        model=Pipeline(a.reader_path,a.reranker_path,a.reranker_helper,a.clip_path,a.artifacts,a.repository)
        rows=[]
        # Save after each completed example. Existing IDs must not be repeated.
        output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():raise FileExistsError('Use a new output path; do not silently overwrite results')
        seen=set()
        with output.open('w') as stream:
            for line in Path(a.input).read_text().splitlines():
                if not line.strip():continue
                row=json.loads(line)
                if row['case_id'] in seen:raise ValueError('Duplicate case ID')
                seen.add(row['case_id']);stream.write(json.dumps(model.predict(row),ensure_ascii=False)+'\n');stream.flush()
        return
    else:
        import numpy as np
        predictions=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()]
        labels=[json.loads(x) for x in Path(a.labels).read_text().splitlines() if x.strip()]
        pred={r['case_id']:r for r in predictions};truth={r['case_id']:r['label'] for r in labels}
        if len(pred)!=len(predictions) or len(truth)!=len(labels) or pred.keys()!=truth.keys():raise ValueError('IDs must match exactly, without omissions or duplicates')
        ids=sorted(pred);y=np.array([truth[k]=='real' for k in ids]);v=np.array([pred[k]['label']=='real' for k in ids])
        if any(truth[k] not in ['real','fake'] or pred[k]['label'] not in ['real','fake'] for k in ids):raise ValueError('Invalid label')
        f1=[]
        for c in [False,True]:
            tp=np.sum((y==c)&(v==c));fp=np.sum((y!=c)&(v==c));fn=np.sum((y==c)&(v!=c))
            f1.append(float(2*tp/max(1,2*tp+fp+fn)))
        result=dict(n=len(ids),correct=int((v==y).sum()),accuracy=float((v==y).mean()),macro_f1=float(np.mean(f1)))
        dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n');return
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Use a new output path')
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


if __name__=='__main__':main()
