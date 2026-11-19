"""External relation/cohort correction; inputs are detached frozen features.

This module contains no Reader, encoder loader, labels, case IDs, or training
loop. A forward processes one fixed cohort (at most64 news items).
"""
import torch
from torch import nn


class SignedResidual(nn.Module):
    def __init__(self, dimensions, width, bound):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimensions, width), nn.GELU(), nn.Linear(width, 1))
        self.bound = None if bound is None else float(bound)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        residual = self.layers(value).squeeze(-1)
        if self.bound is None:
            return residual
        return self.bound * torch.tanh(residual)


class StructuredCalibrator(nn.Module):
    """All new weights live outside the original frozen verification pipeline."""

    def __init__(self, config, use_cohort=True):
        super().__init__()
        local = config['local_encoder']
        cohort = config['cohort_encoder']
        decoder = config['correction_decoder']
        dimensions = config['feature_dimensions']
        self.reader_dimension = dimensions['reader']
        self.clip_dimension = dimensions['clip']
        self.signal_dimension = len(dimensions['signal_names'])
        self.width = local['width']
        self.context_size = cohort['context_size']
        self.use_cohort = bool(use_cohort)
        self.bound = config['residual_bound']
        self.decoder_mode = decoder.get('mode', 'mixture')
        w = self.width
        if self.decoder_mode not in ['mixture', 'local_gated_residual']:
            raise ValueError('Unknown correction decoder mode')
        expected_experts = 1 if self.decoder_mode == 'local_gated_residual' else 3
        if cohort['width'] != w or decoder['learned_experts'] != expected_experts:
            raise ValueError('Decoder expert count or shared width differs from configuration')
        if self.decoder_mode == 'local_gated_residual' and self.use_cohort:
            raise ValueError('Local gated correction requires use_cohort=False')
        self.reader_projection = nn.Sequential(nn.LayerNorm(self.reader_dimension), nn.Linear(self.reader_dimension, w), nn.GELU())
        self.image_projection = nn.Linear(self.clip_dimension, w)
        self.text_projection = nn.Linear(self.clip_dimension, w)
        self.signal_projection = nn.Sequential(nn.Linear(self.signal_dimension + 4, w), nn.GELU())
        self.roles = nn.Embedding(9, w)
        self.rank_projection = nn.Linear(1, w, bias=False)
        self.local_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=w, nhead=local['heads'], dim_feedforward=local['feedforward_width'],
            dropout=config['dropout'], activation='gelu', batch_first=True, norm_first=True)
            for _ in range(local['self_attention_blocks'])])
        self.local_seeds = nn.Parameter(torch.randn(local['pooled_queries'], w)*.02)
        self.local_pool = nn.MultiheadAttention(w, local['heads'], dropout=config['dropout'], batch_first=True)
        self.local_summary = nn.Sequential(nn.Linear(local['pooled_queries']*w, w), nn.GELU(), nn.LayerNorm(w))
        if self.use_cohort:
            self.context_input = nn.Sequential(nn.Linear(2*w, w), nn.GELU())
            self.context_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
                d_model=w, nhead=cohort['heads'], dim_feedforward=local['feedforward_width'],
                dropout=config['dropout'], activation='gelu', batch_first=True, norm_first=True)
                for _ in range(cohort['blocks'])])
            self.context_seed = nn.Parameter(torch.randn(1, 1, w)*.02)
            self.context_pool = nn.MultiheadAttention(w, cohort['heads'], dropout=config['dropout'], batch_first=True)
            self.context_norm = nn.LayerNorm(w)
        if self.decoder_mode == 'local_gated_residual':
            self.experts = nn.ModuleList([SignedResidual(2*w, decoder['hidden_width'], self.bound)])
            router_dimension = 2*w
        else:
            self.experts = nn.ModuleList([
                SignedResidual(w, decoder['hidden_width'], self.bound),
                SignedResidual(2*w, decoder['hidden_width'], self.bound),
                SignedResidual(2*w, decoder['hidden_width'], self.bound)])
            router_dimension = 3*w
        self.router = nn.Sequential(nn.Linear(router_dimension, decoder['router_hidden_width']), nn.GELU(),
                                    nn.Linear(decoder['router_hidden_width'], expected_experts+1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def _check(self, value, shape, name, boolean=False):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError('Invalid shape for '+name)
        if boolean:
            if value.dtype != torch.bool:
                raise ValueError('Expected boolean '+name)
        elif not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('Expected finite floating-point '+name)
        if value.device != self.local_seeds.device:
            raise ValueError('Input/model device mismatch: '+name)

    def forward(self, *, base_score, answer, nodes, node_valid, node_roles,
                image, text, reference_image, reference_text, clip_present, signals):
        if not isinstance(answer, torch.Tensor) or answer.ndim != 2:
            raise ValueError('Expected answer matrix')
        b = len(answer)
        if not 0 < b <= self.context_size:
            raise ValueError('Use a non-empty fixed cohort no larger than context_size')
        if not isinstance(nodes, torch.Tensor) or nodes.ndim != 3 or nodes.shape[0] != b:
            raise ValueError('Expected node tensor')
        n = nodes.shape[1]
        self._check(base_score, (b,), 'base_score')
        self._check(answer, (b, self.reader_dimension), 'answer')
        self._check(nodes, (b, n, self.reader_dimension), 'nodes')
        self._check(node_valid, (b,n), 'node_valid', boolean=True)
        self._check(clip_present, (b,4), 'clip_present', boolean=True)
        self._check(signals, (b,self.signal_dimension), 'signals')
        if node_roles.shape != (b,n) or node_roles.dtype != torch.long or node_roles.device != answer.device:
            raise ValueError('Expected long node_roles with node shape')
        if ((node_roles[node_valid] < 0) | (node_roles[node_valid] > 2)).any():
            raise ValueError('Valid Reader node roles must be image0/news1/evidence2')
        clip_features = [image,text,reference_image,reference_text]
        for name,v in zip(['image','text','reference_image','reference_text'],clip_features):
            self._check(v,(b,self.clip_dimension),name)
        dtype = self.local_seeds.dtype
        # Explicitly detach every upstream signal. Gradients train only this module.
        nodes = self.reader_projection(nodes.detach().to(dtype))
        roles = torch.where(node_valid, node_roles, 0)
        evidence = (roles == 2) & node_valid
        rank = evidence.cumsum(1).to(dtype) / evidence.sum(1,keepdim=True).clamp_min(1)
        rank = rank * evidence
        nodes = nodes + self.roles(roles) + self.rank_projection(rank.unsqueeze(-1))
        answer_token = self.reader_projection(answer.detach().to(dtype)).unsqueeze(1)+self.roles.weight[3]
        clips = torch.stack([
            self.image_projection(image.detach().to(dtype)),
            self.text_projection(text.detach().to(dtype)),
            self.image_projection(reference_image.detach().to(dtype)),
            self.text_projection(reference_text.detach().to(dtype))],dim=1)
        clips = clips+self.roles(torch.arange(4,8,device=answer.device))[None]
        scalar = self.signal_projection(torch.cat([signals.detach().to(dtype),clip_present.to(dtype)],dim=-1))
        tokens = torch.cat([nodes,answer_token,clips,(scalar+self.roles.weight[8]).unsqueeze(1)],dim=1)
        always_present = torch.ones(b,1,dtype=torch.bool,device=answer.device)
        valid = torch.cat([node_valid,always_present,clip_present,always_present],dim=1)
        for block in self.local_blocks:
            tokens = block(tokens,src_key_padding_mask=~valid)
        seeds = self.local_seeds[None].expand(b,-1,-1)
        pooled,_ = self.local_pool(seeds,tokens,tokens,key_padding_mask=~valid,need_weights=False)
        local = self.local_summary(pooled.flatten(1))
        if self.use_cohort:
            context_members = self.context_input(torch.cat([local,scalar],dim=-1))[None]
            for block in self.context_blocks:
                context_members = block(context_members)
            pooled_context,_ = self.context_pool(self.context_seed,context_members,context_members,need_weights=False)
            context = self.context_norm(pooled_context[:,0]).expand(b,-1)
        else:
            context = torch.zeros_like(local)
        if self.decoder_mode == 'local_gated_residual':
            inputs = [torch.cat([local,scalar],dim=-1)]
            router_input = inputs[0]
        else:
            inputs = [local,torch.cat([local,scalar],dim=-1),torch.cat([local,context],dim=-1)]
            router_input = torch.cat([local,scalar,context],dim=-1)
        residuals = torch.stack([expert(v) for expert,v in zip(self.experts,inputs)],dim=1)
        # Expert0 is structurally zero; it cannot learn to rewrite the base score.
        all_residuals = torch.cat([torch.zeros_like(residuals[:,:1]),residuals],dim=1)
        gates = torch.softmax(self.router(router_input),dim=-1)
        correction = (gates*all_residuals).sum(1)
        return dict(score=base_score.detach()+correction.to(base_score.dtype),correction=correction,
            gates=gates,expert_residuals=all_residuals,local_representation=local,cohort_representation=context)


def expert_magnitude_penalty(result):
    """Charge for each route's amplitude even when signed routes cancel."""
    return (result['gates']*result['expert_residuals'].square()).sum(-1).mean()
