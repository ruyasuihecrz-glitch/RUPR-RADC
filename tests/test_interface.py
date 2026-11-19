import itertools
import math
import unittest

import torch

from rupr_radc import DriftBenchAdapter
from rupr_radc.core.rank_policy import pl_distribution_kl,pl_log_probability
from rupr_radc.core.native_preserving_dpo import preserving_loss


class InterfaceChecks(unittest.TestCase):
    def test_benchmark_truth_is_rejected_before_feature_extraction(self):
        def provider(records):
            raise AssertionError('Provider must not see ground truth')
        adapter=DriftBenchAdapter(None,provider)
        with self.assertRaises(ValueError):adapter.predict([{'case_id':'synthetic','label':'real'}])

    def test_full_permutation_kl_matches_enumeration(self):
        a=torch.tensor([.3,-.7,.8],requires_grad=True);b=torch.tensor([-.2,.6,.1])
        expected=a.sum()*0
        for order in itertools.permutations(range(3)):
            lp=pl_log_probability(a,list(order));lq=pl_log_probability(b,list(order))
            expected=expected+lp.exp()*(lp-lq)
        actual=pl_distribution_kl(a,b)
        # The frozen recipe normalizes the full-order KL by n-1 choices.
        self.assertLess(abs(float((actual-expected/2).detach())),1e-6)

    def test_absolute_preference_has_finite_gradient(self):
        logits=torch.tensor([0.,0.,0.],requires_grad=True)
        cfg=dict(beta=.1,sampling_temperature=1.,repair_objective='absolute_margin',
                 absolute_preference_margin=1.,keep_kl_weight=1.,repair_kl_weight=.03)
        loss,_=preserving_loss(logits,logits.detach(),[dict(chosen=[0,1,2],rejected=[2,1,0])],False,cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0]),0)
        self.assertGreater(float(logits.grad[2]),0)


if __name__=='__main__':unittest.main()
