"""Keep successful native rankings; learn verified repairs on native-wrong queries."""
import math

import torch

from .rank_policy import multi_rank_dpo_loss, pl_distribution_kl, pl_log_probability


def native_targets(item, candidates, values, truth, config):
    """Construct training-only targets from existing frozen Reader feedback."""
    if truth.get('split') != 'train':
        raise ValueError('Only external training feedback may create targets')
    if len(candidates) != len(values):
        raise ValueError('Unaligned candidates and feedback')
    ids=[s['source_id'] for s in item['sources']]
    native=[i for i,c in enumerate(candidates) if 'native' in c['aliases']]
    if len(native)!=1:raise ValueError('Exactly one native order is required')
    anchor=native[0];label=truth['label']
    if label not in ['real','fake']:raise ValueError('Unknown training label')
    for candidate,value in zip(candidates,values):
        if sorted(candidate['order'])!=sorted(ids):raise ValueError('Changed evidence set')
        if value['prediction'] not in ['real','fake']:raise ValueError('Invalid prediction')
        if not all(math.isfinite(value[k]) for k in ['score','log_p_real_two_labels','log_p_fake_two_labels']):
            raise ValueError('Nonfinite Reader feedback')
    to_indices=lambda order:[ids.index(k) for k in order]
    native_order=to_indices(candidates[anchor]['order'])
    native_correct=values[anchor]['prediction']==label
    pairs=[]
    if not native_correct:
        reward=f'log_p_{label}_two_labels';sign=1 if label=='real' else -1
        for c,v in zip(candidates,values):
            if v['prediction']!=label:continue
            if v[reward]-values[anchor][reward]<config['min_logp_gap']:continue
            if sign*(v['score']-values[anchor]['score'])<config['min_margin_gap']:continue
            pair=dict(chosen=to_indices(c['order']),rejected=native_order)
            if pair not in pairs:pairs.append(pair)
    return dict(case_id=item['case_id'],active=int(bool(pairs)),
        chosen=pairs[0]['chosen'] if pairs else native_order,rejected=native_order,
        preference_pairs=pairs,retain_native=not bool(pairs),
        target_analysis=dict(native_correct=native_correct,
                             role='repair' if pairs else 'keep_correct' if native_correct else 'keep_without_observed_repair'))


def preserving_loss(logits,reference,pairs,retain_native,config,weight=1.):
    if weight<=0:raise ValueError('Expected positive training weight')
    objective=config.get('repair_objective','dpo')
    if objective not in ['dpo','dpo_nll','absolute_margin']:
        raise ValueError('Unknown repair objective')
    beta=float(config['beta']);temperature=float(config['sampling_temperature'])
    margin=float(config.get('absolute_preference_margin',1.))
    positive_weight=float(config.get('repair_positive_nll_weight',1.)) if objective=='dpo_nll' else 0.
    if not all(math.isfinite(x) for x in [beta,temperature,margin,positive_weight]) or beta<=0 or temperature<=0 or margin<0 or positive_weight<0:
        raise ValueError('Invalid repair loss coefficients')
    zero=logits.float().sum()*0.;nll=zero;raw_win=zero;mean_gap=zero
    preference=zero
    if pairs:
        orders={tuple(p[k]) for p in pairs for k in ['chosen','rejected']}
        policy={order:pl_log_probability(logits,list(order),temperature) for order in orders}
        gaps=torch.stack([policy[tuple(p['chosen'])]-policy[tuple(p['rejected'])] for p in pairs])
        raw_win=(gaps.detach()>0).float().mean();mean_gap=gaps.detach().mean()
        if objective=='absolute_margin':
            preference=torch.nn.functional.softplus(beta*(margin-gaps)).mean()
        else:
            preference=multi_rank_dpo_loss(logits,reference,pairs,beta,temperature)
            if objective=='dpo_nll':
                chosen=sorted({tuple(p['chosen']) for p in pairs})
                nll=-torch.stack([policy[order] for order in chosen]).mean()/max(1,len(logits)-1)
    kl=pl_distribution_kl(logits,reference,config['sampling_temperature'])
    coefficient=config['keep_kl_weight'] if retain_native else config['repair_kl_weight']
    loss=float(weight)*(preference+positive_weight*nll+coefficient*kl)
    statistics=torch.stack([preference.detach(),kl.detach(),logits.new_tensor(float(retain_native)),
                            logits.new_tensor(float(bool(pairs))),nll.detach(),raw_win,mean_gap]).detach()
    return loss,statistics
