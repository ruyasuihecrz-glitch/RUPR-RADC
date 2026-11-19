"""One jointly trainable readout over statistical and local evidence relations."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .structured_calibration import StructuredCalibrator


def statistical_features(answer, signals, projection):
    z=(answer-projection['mean'])@projection['components'].T/np.sqrt(projection['variance'])
    matching=(signals[:,1]-projection['match_center'])/projection['match_scale']
    return np.column_stack([z,matching]).astype(np.float64)


class UnifiedCalibrator(StructuredCalibrator):
    def __init__(self,config):
        super().__init__(config,use_cohort=False)
        if self.decoder_mode!='local_gated_residual' or self.bound is not None:
            raise ValueError('Unified readout needs the unbounded local relation encoder')
        self.relation_hidden=self.experts[0].layers[:2]
        del self.experts
        hidden=config['correction_decoder']['hidden_width']
        self.readout=nn.Linear(17+hidden+1,1,dtype=torch.float64)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
        self.gate_scope=config.get('correction_decoder',{}).get('gate_scope','relation')
        if self.gate_scope not in ['relation','whole_correction']:
            raise ValueError('Unknown correction gate scope')

    def initialize_from(self,legacy_state,linear_weights):
        mapped={}
        for key,value in legacy_state.items():
            if key.startswith('experts.0.layers.2.'): continue
            new_key=key.replace('experts.0.layers.0.','relation_hidden.0.')
            mapped[new_key]=value
        result=self.load_state_dict(mapped,strict=False)
        if set(result.missing_keys)!={'readout.weight','readout.bias'} or result.unexpected_keys:
            raise ValueError('Incompatible initialization weights: '+str(result))
        with torch.no_grad():
            self.readout.weight[0,:17].copy_(torch.as_tensor(linear_weights[:17]))
            self.readout.weight[0,17:-1].copy_(legacy_state['experts.0.layers.2.weight'].flatten())
            self.readout.weight[0,-1].copy_(legacy_state['experts.0.layers.2.bias'][0])
            self.readout.bias[0].copy_(torch.as_tensor(linear_weights[17]))

    def forward(self,*,statistics,base_score,answer,nodes,node_valid,node_roles,
                image,text,reference_image,reference_text,clip_present,signals):
        b=len(answer)
        if answer.ndim!=2 or nodes.ndim!=3 or not 0<b<=self.context_size:
            raise ValueError('Expected a nonempty local batch no larger than64')
        n=nodes.shape[1]
        for name,value,shape in [('base_score',base_score,(b,)),('statistics',statistics,(b,17)),
            ('answer',answer,(b,self.reader_dimension)),('nodes',nodes,(b,n,self.reader_dimension)),
            ('signals',signals,(b,self.signal_dimension))]: self._check(value,shape,name)
        self._check(node_valid,(b,n),'node_valid',boolean=True)
        self._check(clip_present,(b,4),'clip_present',boolean=True)
        if node_roles.shape!=(b,n) or node_roles.dtype!=torch.long or node_roles.device!=answer.device:
            raise ValueError('Invalid node roles')
        if ((node_roles[node_valid]<0)|(node_roles[node_valid]>2)).any():
            raise ValueError('Invalid valid role')
        for name,value in [('image',image),('text',text),('reference_image',reference_image),('reference_text',reference_text)]:
            self._check(value,(b,self.clip_dimension),name)
        dtype=self.local_seeds.dtype
        roles=torch.where(node_valid,node_roles,0)
        evidence=(roles==2)&node_valid
        rank=evidence.cumsum(1).to(dtype)/evidence.sum(1,keepdim=True).clamp_min(1)
        rank=rank*evidence
        node_tokens=self.reader_projection(nodes.detach().to(dtype))+self.roles(roles)+self.rank_projection(rank.unsqueeze(-1))
        answer_token=self.reader_projection(answer.detach().to(dtype)).unsqueeze(1)+self.roles.weight[3]
        clips=torch.stack([self.image_projection(image.detach().to(dtype)),self.text_projection(text.detach().to(dtype)),
            self.image_projection(reference_image.detach().to(dtype)),self.text_projection(reference_text.detach().to(dtype))],dim=1)
        clips=clips+self.roles(torch.arange(4,8,device=answer.device))[None]
        scalar=self.signal_projection(torch.cat([signals.detach().to(dtype),clip_present.to(dtype)],dim=-1))
        tokens=torch.cat([node_tokens,answer_token,clips,(scalar+self.roles.weight[8]).unsqueeze(1)],dim=1)
        present=torch.ones(b,1,dtype=torch.bool,device=answer.device)
        valid=torch.cat([node_valid,present,clip_present,present],dim=1)
        for block in self.local_blocks: tokens=block(tokens,src_key_padding_mask=~valid)
        pooled,_=self.local_pool(self.local_seeds[None].expand(b,-1,-1),tokens,tokens,key_padding_mask=~valid,need_weights=False)
        local=self.local_summary(pooled.flatten(1))
        relation_input=torch.cat([local,scalar],dim=-1)
        hidden=self.relation_hidden(relation_input)
        gates=torch.softmax(self.router(relation_input),dim=-1)
        gate=gates[:,1:2]
        features=torch.cat([statistics.detach().double(),(gate*hidden).double(),gate.double()],dim=-1)
        statistical_correction=F.linear(statistics.detach().double(),self.readout.weight[:,:17],self.readout.bias).squeeze(-1)
        ungated=(hidden.double()*self.readout.weight[0,17:-1]).sum(-1)+self.readout.weight[0,-1]
        if self.gate_scope=='whole_correction':
            correction=gate[:,0].double()*(statistical_correction+ungated)
        else:
            correction=self.readout(features).squeeze(-1)
        magnitude=(gate[:,0].double()*ungated.square()).mean()
        return dict(score=base_score.detach()+correction,correction=correction,
                    statistical_correction=statistical_correction,ungated_relation_correction=ungated,
                    relation_magnitude=magnitude,gates=gates,local_representation=local)


def bernoulli_js(left,right):
    """Symmetric, differentiable probability consistency; neither view is a teacher."""
    a=torch.stack([F.logsigmoid(left),F.logsigmoid(-left)],dim=-1)
    b=torch.stack([F.logsigmoid(right),F.logsigmoid(-right)],dim=-1)
    mixture=torch.logaddexp(a,b)-np.log(2.)
    return .5*((a.exp()*(a-mixture)).sum(-1)+(b.exp()*(b-mixture)).sum(-1))


def bernoulli_kl(teacher,student,temperature=2.):
    """Distill detached final predictions, with the usual temperature correction."""
    if temperature<=0:raise ValueError('Temperature must be positive')
    a=teacher.detach()/temperature;b=student/temperature
    p=torch.stack([F.logsigmoid(a),F.logsigmoid(-a)],dim=-1)
    q=torch.stack([F.logsigmoid(b),F.logsigmoid(-b)],dim=-1)
    return temperature**2*(p.exp()*(p-q)).sum(-1)


def unified_loss(result,target,weights,fact_edges,appearance_edges,config,population_sizes,teacher_score=None):
    score=result['score'];target=target.detach();weights=weights.detach()
    if target.shape!=score.shape or weights.shape!=score.shape:
        raise ValueError('Invalid supervision shape')
    if not torch.isfinite(target).all() or not ((target==0)|(target==1)).all():
        raise ValueError('External labels must be binary')
    if not torch.isfinite(weights).all() or not (weights>0).all():
        raise ValueError('Weights must be positive and finite')
    classification=(weights*F.binary_cross_entropy_with_logits(score,target,reduction='none')).mean()
    fact=score.sum()*0.;appearance=score.sum()*0.
    for edges,name in [(fact_edges,'fact'),(appearance_edges,'appearance')]:
        if not len(edges): continue
        indices=edges[:,:2].long();edge_weight=edges[:,2]
        left,right=score[indices[:,0]],score[indices[:,1]]
        if name=='fact':
            if not ((target[indices[:,0]]==1)&(target[indices[:,1]]==0)).all():
                raise ValueError('Fact pairs require original real and edited fake')
            mode=config.get('fact_objective','relative')
            if mode=='relative':values=F.softplus(config['fact_margin']-(left-right))
            elif mode=='absolute':values=.5*(F.softplus(config['fact_margin']-left)+F.softplus(config['fact_margin']+right))
            else:raise ValueError('Unknown fact objective')
        else:
            if not (target[indices[:,0]]==target[indices[:,1]]).all():
                raise ValueError('Appearance pairs must retain their existing label')
            values=bernoulli_js(left,right)
        loss=(edge_weight*values).sum()/len(score)*population_sizes['fit']/population_sizes[name]
        if name=='fact': fact=loss
        else: appearance=loss
    magnitude=result['relation_magnitude']
    distillation=score.sum()*0.
    if config.get('distillation_weight',0)>0:
        if teacher_score is None or teacher_score.shape!=score.shape or not torch.isfinite(teacher_score).all():
            raise ValueError('Need frozen teacher scores for the same training inputs')
        mask=((teacher_score.detach()>0)==(target>.5)).to(score.dtype)
        distillation=(weights*mask*bernoulli_kl(teacher_score,score,config.get('distillation_temperature',2.))).mean()
    total=(classification+config['fact_weight']*fact+config['appearance_weight']*appearance+
           config['magnitude_weight']*magnitude+config.get('distillation_weight',0)*distillation)
    return dict(total=total,classification=classification,fact=fact,appearance=appearance,magnitude=magnitude,distillation=distillation)
