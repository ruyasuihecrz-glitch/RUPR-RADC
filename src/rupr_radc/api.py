"""Label-free feature inference and a small adapter for benchmark runners."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .core.unified_calibration import UnifiedCalibrator, statistical_features

FEATURE_NAMES=('base_score','answer','nodes','node_valid','node_roles','image','text',
               'reference_image','reference_text','clip_present','signals')


class Calibrator:
    def __init__(self, artifacts='artifacts', recipe='configs/locked_v1.json', device='cpu'):
        artifacts=Path(artifacts);self.device=device
        self.recipe=json.loads(Path(recipe).read_text())
        for name,expected in self.recipe['artifacts'].items():
            if name.startswith('rupr/'):continue
            if hashlib.sha256((artifacts/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Artifact checksum mismatch: '+name)
        saved=torch.load(artifacts/'radc.pt',map_location='cpu',weights_only=True)
        self.model=UnifiedCalibrator(saved['architecture']).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(saved['state_dict'],strict=True)
        with np.load(artifacts/'projection.npz',allow_pickle=False) as p:
            self.projection={k:p[k] for k in p.files}

    @torch.inference_mode()
    def predict_features(self, features, case_ids=None, batch_size=64):
        if not 1<=batch_size<=64:raise ValueError('Inference batch must be in [1,64]')
        if set(features)!=set(FEATURE_NAMES):raise ValueError('Expected exactly the documented feature arrays; no labels')
        bank={k:np.asarray(v) for k,v in features.items()};n=len(bank['base_score'])
        if any(len(v)!=n for v in bank.values()):raise ValueError('Unaligned feature arrays')
        if case_ids is None:case_ids=[str(i) for i in range(n)]
        if len(case_ids)!=n or len(set(case_ids))!=n:raise ValueError('Case IDs must be unique and aligned')
        bank['statistics']=statistical_features(bank['answer'],bank['signals'],self.projection)
        rows=[]
        for start in range(0,n,batch_size):
            inputs={k:torch.from_numpy(np.ascontiguousarray(v[start:start+batch_size])).to(self.device) for k,v in bank.items()}
            result=self.model(**inputs)
            score=result['score'].cpu().numpy()
            if not np.isfinite(score).all():raise ValueError('Nonfinite calibrated score')
            for j,value in enumerate(score):
                i=start+j
                rows.append(dict(case_id=str(case_ids[i]),label='real' if value>0 else 'fake',
                                 score=float(value),raw_score=float(bank['base_score'][i])))
        return rows

    def predict_npz(self,path,batch_size=64):
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(FEATURE_NAMES)|{'case_id'}:raise ValueError('Unexpected feature bank fields')
            ids=z['case_id'].tolist();features={k:z[k] for k in FEATURE_NAMES}
        return self.predict_features(features,ids,batch_size)


class DriftBenchAdapter:
    """Integration hook: external evaluators supply legal records or feature banks.

    No specific unpublished organizer API is assumed. Ground truth is handled
    by the evaluator, never passed to this class.
    """
    def __init__(self, calibrator, feature_provider=None):
        self.calibrator=calibrator;self.feature_provider=feature_provider

    def predict(self, records):
        if self.feature_provider is None:
            raise ValueError('Register a feature_provider(records)->(case_ids, feature_arrays)')
        forbidden={'label','target','ground_truth','original_label','fake_cls','group'}
        if any(forbidden.intersection(row) for row in records):
            raise ValueError('Remove evaluation labels before calling the model')
        ids,features=self.feature_provider(records)
        return self.calibrator.predict_features(features,ids)
