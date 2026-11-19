from torch.nn import functional as F
import math
import functools
import torch

def pl_log_probability(logits, order, temperature=1.):
    if logits.ndim != 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('PL requires a vector and a finite positive temperature')
    validate_order(order, len(logits))
    if len(logits) == 0:
        return logits.sum()*0.
    if not torch.isfinite(logits).all():
        raise ValueError('PL received nonfinite logits')
    values = logits.float()[torch.as_tensor(order, device=logits.device)]/temperature
    return (values-torch.logcumsumexp(values.flip(0), dim=0).flip(0)).sum()

@functools.lru_cache(maxsize=32)
def _pl_subset_layers(n, device):
    """Subset DP topology only; never cache trainable probabilities."""
    layers=[]
    for size in range(n,1,-1):
        masks=[m for m in range(1<<n) if m.bit_count()==size]
        next_masks=[m for m in range(1<<n) if m.bit_count()==size-1]
        lookup={m:j for j,m in enumerate(next_masks)}
        remaining=torch.tensor([[(m>>i)&1 for i in range(n)] for m in masks],device=device,dtype=torch.bool)
        children=torch.tensor([[lookup[m^(1<<i)] if (m>>i)&1 else 0 for i in range(n)] for m in masks],device=device)
        layers.append((remaining,children,len(next_masks)))
    return layers

def pl_distribution_kl(logits, reference, temperature=1.):
    """Exact KL(PL(policy)||PL(fixed reference)) over ALL complete permutations.

    Sum conditional KL under policy reach probabilities over remaining subsets.
    O(n*2**n), feasible for our ten evidence records; no sampled KL surrogate,
    no positive-only likelihood anchor, and no gradients to the reference.
    """
    if logits.ndim!=1 or logits.shape!=reference.shape or len(logits)>12:
        raise ValueError('Exact PL KL requires matching vectors of at most 12 documents')
    if temperature<=0 or not math.isfinite(temperature) or not torch.isfinite(logits).all() or not torch.isfinite(reference).all():
        raise ValueError('Invalid KL scores or temperature')
    z=logits.double()/temperature;ref=reference.detach().double()/temperature
    total=z.sum()*0.;mass=z.new_ones(1)
    for remaining,children,next_size in _pl_subset_layers(len(z),str(z.device)):
        logp=torch.log_softmax(z.expand(len(remaining),-1).masked_fill(~remaining,-torch.inf),dim=-1)
        logq=torch.log_softmax(ref.expand(len(remaining),-1).masked_fill(~remaining,-torch.inf),dim=-1)
        delta=logp.masked_fill(~remaining,0.)-logq.masked_fill(~remaining,0.)
        p=logp.exp();local=(p*delta).sum(-1)
        total=total+(mass*local).sum()
        mass=z.new_zeros(next_size).scatter_add(0,children.flatten(),(mass[:,None]*p).flatten())
    return (total/max(1,len(z)-1)).float()

def multi_rank_dpo_loss(logits, reference, pairs, beta=.1, temperature=1., positive_nll_weight=0.):
    """Average reliable full-order preferences within one query, not across documents.

    The reference remains fixed; all pairs belong to this query and must be
    built from actual correct/incorrect Reader outcomes outside model inputs.
    """
    if not pairs or logits.shape != reference.shape or logits.ndim != 1 or beta <= 0 or positive_nll_weight<0:
        raise ValueError('Invalid multi-preference query')
    orders = {}
    for pair in pairs:
        for field in ['chosen', 'rejected']:
            order = tuple(pair[field]);validate_order(list(order),len(logits))
            orders[order] = None
        if pair['chosen'] == pair['rejected']:raise ValueError('Identical preference orders')
    policy = {order:pl_log_probability(logits,list(order),temperature) for order in orders}
    with torch.no_grad():
        fixed = {order:pl_log_probability(reference,list(order),temperature) for order in orders}
    differences = [(policy[tuple(p['chosen'])]-policy[tuple(p['rejected'])])-
                   (fixed[tuple(p['chosen'])]-fixed[tuple(p['rejected'])]) for p in pairs]
    loss=F.softplus(-beta*torch.stack(differences)).mean()
    if positive_nll_weight:
        positive_orders=sorted({tuple(p['chosen']) for p in pairs})
        nll=-torch.stack([policy[order] for order in positive_orders]).mean()/max(1,len(logits)-1)
        loss=loss+positive_nll_weight*nll
    return loss

def greedy_order(item, logits):
    ids=[s['source_id'] for s in item['sources']]
    if len(logits)!=len(ids) or not all(math.isfinite(v) for v in logits):
        raise ValueError('A finite score is required for each evidence record')
    # The exact native greedy recipe uses source-list position to break ties.
    return [ids[j] for j in sorted(range(len(ids)),key=lambda j:(-logits[j],j))]



def rank_dpo_loss(logits, reference, chosen, rejected, active, beta=.1, temperature=1.):
    if logits.shape != reference.shape or logits.ndim != 1:
        raise ValueError('Reference and policy scores must cover the same documents')
    validate_order(chosen,len(logits));validate_order(rejected,len(logits))
    if active not in (0, 1, False, True) or beta<=0:
        raise ValueError('Invalid preference mask or beta')
    if not active:
        return logits.float().sum()*0.
    if chosen == rejected:
        raise ValueError('An active preference cannot compare the same order')
    policy_gap=pl_log_probability(logits,chosen,temperature)-pl_log_probability(logits,rejected,temperature)
    with torch.no_grad():
        reference_gap=pl_log_probability(reference,chosen,temperature)-pl_log_probability(reference,rejected,temperature)
    return -F.logsigmoid(beta*(policy_gap-reference_gap))


def validate_order(order, n):
    if any(isinstance(i, bool) or not isinstance(i, int) for i in order) or sorted(order) != list(range(n)):
        raise ValueError('Order must contain every evidence index exactly once')
