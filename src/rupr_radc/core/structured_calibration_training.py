"""Training utilities for the external structured corrector only."""
import random

import numpy as np
import torch
from torch.nn import functional as F

from .structured_calibration import expert_magnitude_penalty


INPUT_NAMES = ('base_score', 'answer', 'nodes', 'node_valid', 'node_roles',
               'image', 'text', 'reference_image', 'reference_text',
               'clip_present', 'signals')


def correction_loss(result, base_score, target, weights, config):
    """Labels stay in the loss; they are never model/cohort inputs."""
    if target.shape != base_score.shape or weights.shape != target.shape:
        raise ValueError('One external target and weight are required per question')
    if not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('External labels must be binary')
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Training weights must be positive and finite')
    target, weights = target.detach(), weights.detach()
    smoothing=config.get('label_smoothing',0.)
    if not 0<=smoothing<1:raise ValueError('label_smoothing must be in[0,1)')
    soft_target=target if smoothing==0 else target*(1-smoothing)+.5*smoothing
    classification = (weights * F.binary_cross_entropy_with_logits(
        result['score'], soft_target, reduction='none')).mean()
    correct_before = ((base_score.detach() > 0) == (target > .5)).to(weights.dtype)
    preservation = (weights * correct_before * result['correction'].square()).mean()
    expert_penalty = expert_magnitude_penalty(result)
    total = (classification + config['preservation_penalty'] * preservation
             + config['expert_penalty'] * expert_penalty)
    return dict(total=total, classification=classification,
                preservation=preservation, expert_penalty=expert_penalty)


def training_step(model, optimizer, inputs, target, weights, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model(**inputs)
    losses = correction_loss(result, inputs['base_score'], target, weights, config)
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Nonfinite correction loss')
    losses['total'].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_gradient_norm'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite correction gradient')
    optimizer.step()
    return {**{k: float(v.detach()) for k, v in losses.items()},
            'gradient_norm': float(norm),
            'mean_correction': float(result['correction'].detach().mean()),
            'mean_gates': result['gates'].detach().mean(0).cpu().tolist()}


def accumulated_training_step(model, optimizer, batches, config):
    """One optimizer update over microbatches, weighted by actual example count."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = sum(len(target) for _, target, _ in batches)
    if count <= 0:
        raise ValueError('An optimizer update requires examples')
    totals = {}
    for inputs, target, weights in batches:
        fraction = len(target) / count
        result = model(**inputs)
        losses = correction_loss(result, inputs['base_score'], target, weights, config)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite accumulated correction loss')
        (losses['total'] * fraction).backward()
        values = {**losses, 'mean_correction': result['correction'].mean()}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * fraction
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_gradient_norm'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite accumulated correction gradient')
    optimizer.step()
    return dict(totals, gradient_norm=float(norm))


def rng_state(generator):
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                python=random.getstate(), generator=generator.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state, generator):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    generator.set_state(state['generator'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
