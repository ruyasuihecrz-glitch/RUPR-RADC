import torch

def label_tokens(tokenizer, config):
    prefix=tokenizer.encode(config['score_prefix'],add_special_tokens=False)
    labels={label:tokenizer.encode(label,add_special_tokens=False) for label in config['score_labels']}
    if any(len(ids)!=1 for ids in labels.values()):
        raise ValueError('This protocol requires one token per label; do not approximate multi-token probabilities')
    for label,ids in labels.items():
        assert tokenizer.encode(config['score_prefix']+label,add_special_tokens=False)==prefix+ids
    return prefix,{k:v[0] for k,v in labels.items()}

def append_prefix(inputs, prefix):
    result=dict(inputs);ids=result['input_ids']
    if ids.shape[0]!=1:raise ValueError('Scoring recipe requires batch one')
    extra=torch.tensor([prefix],dtype=ids.dtype,device=ids.device)
    result['input_ids']=torch.cat([ids,extra],dim=1)
    result['attention_mask']=torch.cat([result['attention_mask'],torch.ones_like(extra)],dim=1)
    for name in ['mm_token_type_ids','token_type_ids']:
        if name in result:
            old=result[name];result[name]=torch.cat([old,torch.zeros_like(extra,dtype=old.dtype)],dim=1)
    if 'position_ids' in result:raise ValueError('Unexpected precomputed position IDs; must recompute RoPE after prefix append')
    return result

def extract_score(logits, ids):
    logp=torch.log_softmax(logits.float(),dim=-1)
    a,b=float(logp[ids['real']]),float(logp[ids['fake']])
    score=a-b
    if not all(torch.isfinite(torch.tensor(x)).item() for x in [a,b,score]):raise ValueError('Non-finite label likelihood')
    return {'log_p_real':a,'log_p_fake':b,'score':score,
        'raw_real_fake_probability_mass':float(logp[[ids['real'],ids['fake']]].exp().sum()),
        'real_probability_normalized_over_two_labels':float(torch.sigmoid(torch.tensor(score))),
        'prediction':'real' if score>0 else 'fake','exact_tie':score==0}
