"""News + image + supplied evidence -> RUPR -> frozen Reader -> locked RADC."""
import functools
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .api import Calibrator
from .core.aitr_clip import FrozenAITRClip
from .core.ccn_calibration_features import reader_features,clip_summary
from .core.evidence_reranking import FrozenReranker
from .core.model import BatchedQwenModel


class Pipeline:
    def __init__(self,reader_path,reranker_path,reranker_helper,clip_path,
                 artifacts='artifacts',repository='.',clip_device='cpu'):
        from peft import PeftModel
        root=Path(repository);self.calibrator=Calibrator(artifacts,root/'configs/locked_v1.json','cuda:0')
        self.recipe=self.calibrator.recipe
        adapter=Path(artifacts)/'rupr'
        for name in ['adapter_config.json','adapter_model.safetensors']:
            if hashlib.sha256((adapter/name).read_bytes()).hexdigest()!=self.recipe['artifacts']['rupr/'+name]:
                raise ValueError('RUPR artifact checksum mismatch')
        cfg=dict(self.recipe['reader_config'],model_path=str(reader_path))
        self.reader_config=cfg;self.reader=BatchedQwenModel(cfg)
        self.reader.processor.apply_chat_template=functools.partial(self.reader.processor.apply_chat_template,enable_thinking=False)
        self.reader.model.eval().requires_grad_(False)
        self.reranker=FrozenReranker(dict(cfg,reranker_path=str(reranker_path),reranker_helper=str(reranker_helper)))
        self.reranker.native.model=PeftModel.from_pretrained(self.reranker.native.model,str(adapter),is_trainable=False).eval()
        self.clip=FrozenAITRClip(clip_path,self.recipe['clip_sha256'],clip_device)
        self.prompt=(root/'prompts/evidence_reranking_reader.md').read_text()
        self.instruction=(root/'prompts/evidence_reranking_instruction.md').read_text()

    @torch.inference_mode()
    def predict(self, record):
        allowed={'case_id','news','image_path','sources','reference_images'}
        if set(record)-allowed:raise ValueError('Only documented news/evidence fields are accepted')
        row=dict(record);row.setdefault('reference_images',[])
        ids=[s['source_id'] for s in row['sources']]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate evidence IDs')
        scores=[]
        with Image.open(row['image_path']) as original:image=original.convert('RGB')
        for source in row['sources']:
            pair=self.reranker.pair(row['news'],image,source['text'],self.instruction)
            # Use one pair at a time, preserving the locked full-input budget.
            inputs=self.reranker.tokenize(pair,True)
            hidden=self.reranker.native.model(**inputs,use_cache=False).last_hidden_state[:,-1]
            scores.append(float(self.reranker.native.score_linear(hidden).squeeze().float()))
        row['order']=[ids[i] for i in sorted(range(len(ids)),key=lambda i:(-scores[i],i))]
        row['_root']=Path('.')
        nodes,answer,value=reader_features(self.reader,row,self.reader_config,self.prompt)
        image_vector=self.clip.encode_images([row['image_path']])[0]
        refs=[v['image_path'] for v in row['reference_images']]
        reference_vectors=self.clip.encode_images(refs) if refs else []
        lookup={s['source_id']:s for s in row['sources']}
        texts=[lookup[k]['text'] for k in row['order']]
        text_vectors=self.clip.encode_texts([row['news']]+texts)
        clip,_=clip_summary(image_vector,text_vectors[0],reference_vectors,text_vectors[1:],value['score'])
        roles=np.array([0,1]+[2]*(len(nodes)-2),np.int64)
        features=dict(base_score=np.array([value['score']],np.float64),answer=answer[None],nodes=nodes[None],
                      node_valid=np.ones((1,len(nodes)),bool),node_roles=roles[None],
                      **{k:v[None] for k,v in clip.items()})
        result=self.calibrator.predict_features(features,[row['case_id']])[0]
        result['evidence_order']=row['order']
        return result
