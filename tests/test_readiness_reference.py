import unittest
import numpy as np
from amvi.diagnostics import binary_diagnostic


class BinaryReferenceTests(unittest.TestCase):
    def test_probability_screen_is_complement_invariant(self):
        rng=np.random.default_rng(429)
        for probability in [.005,.01,.2,.5,.8,.99,.995]:
            x=(rng.random((4,2000))<probability).astype(float)
            a,b=binary_diagnostic(x),binary_diagnostic(1-x)
            self.assertEqual(a['eligible'],b['eligible'])
            for key in ['raw_ess','bulk_ess','rhat','probability_mcse']:
                self.assertAlmostEqual(a[key],b[key],places=8)

    def test_near_certain_iid_event_no_longer_rejected_for_tail_quantile(self):
        x=(np.random.default_rng(429).random((4,2000))<.99).astype(float)
        d=binary_diagnostic(x)
        self.assertIsNone(d['tail_ess'])
        self.assertTrue(d['eligible'])
        self.assertGreater(d['probability_mcse'],0)

    def test_disagreeing_chains_are_rejected(self):
        x=np.zeros((4,2000));x[:2]=1
        self.assertFalse(binary_diagnostic(x)['eligible'])

    def test_persistent_chains_are_rejected(self):
        x=np.tile(np.repeat([0.,1.],1000),(4,1))
        self.assertFalse(binary_diagnostic(x)['eligible'])

    def test_constant_observations_are_flagged_without_zero_mcse(self):
        for value in [0,1]:
            d=binary_diagnostic(np.full((4,2000),value))
            self.assertTrue(d['constant'])
            self.assertIsNone(d['probability_mcse'])
            self.assertIn('no mixing evidence',d['note'])


if __name__=='__main__':unittest.main()
