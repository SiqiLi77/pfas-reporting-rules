import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import environmental_reconstruction_20260927 as a

class ReconstructionTests(unittest.TestCase):
    def test_state_map_and_family(self):
        self.assertEqual(a.CHEMS,['PFBS','PFHpA','PFHxS','PFNA','PFOA','PFOS'])
        np.testing.assert_array_equal(a.BITS@(2**np.arange(6)),np.arange(64))
        self.assertEqual(a.FAMILY[16],1)
        self.assertEqual(a.FAMILY[32],2)
        self.assertEqual(a.FAMILY[48],3)
        self.assertEqual(a.H[48,a.ENDPOINTS.index('PFOA+PFOS')],1)
        self.assertEqual(a.H[49,a.ENDPOINTS.index('PFOA+PFOS')],1)
        self.assertEqual(a.H[48,a.ENDPOINTS.index('PFHxS+PFOS')],0)

    def test_uniform_count_conditioning(self):
        q=np.full((64,64),1/64)
        p,s,_=a.summarize_probabilities(q,np.arange(64),np.zeros(64,int))
        np.testing.assert_allclose(p[:,a.ENDPOINTS.index('PFOA+PFOS')],.25)
        np.testing.assert_allclose(p[:,1],(1-1/8)**2)
        np.testing.assert_allclose(s['identity_given_count_nll'][a.K==1],np.log(6))
        np.testing.assert_allclose(s['family_given_count_nll'][a.K==1],np.log(2))
        c=s['pair_given_count_nll__PFOA+PFOS']
        self.assertAlmostEqual(c[48],np.log(15))
        self.assertAlmostEqual(c[3],-np.log(14/15))
        self.assertAlmostEqual(c[0],0)
        self.assertAlmostEqual(c[63],0)

    def test_deterministic_and_chain_rule(self):
        p,s,_=a.summarize_probabilities(np.eye(64),np.arange(64),np.arange(64))
        np.testing.assert_array_equal(p,a.H)
        self.assertLess(max(s['family_given_count_nll']),1e-10)
        np.testing.assert_allclose(s['pattern_nll'],s['count_nll']+s['identity_given_count_nll'],atol=2e-12)
        for e in a.ENDPOINTS:np.testing.assert_array_equal(s['brier__'+e],np.zeros(64))

    def test_nonempty_information_beyond_count(self):
        # Both distributions have K=2 with certainty but allocate family composition differently.
        q=np.zeros((2,64));q[0,48]=1;q[1,36]=1
        self.assertEqual(a.K[48],a.K[36])
        p,_,_=a.summarize_probabilities(q,np.array([48,36]),np.zeros(2,int))
        self.assertEqual(p[0,0],p[1,0]);self.assertNotEqual(p[0,1],p[1,1])

    def test_pws_weights_not_record_weights(self):
        d=pd.DataFrame({'state':['A']*3+['B'],'pws_id':['1','1','2','3']})
        x=np.array([[0.],[0.],[1.],[1.]])
        previous=a.REPS;a.REPS=100
        try:
            p,b=a.score_bootstrap(d,x)
            self.assertAlmostEqual(p[0],.75)
            self.assertEqual(b.shape,(100,1))
            self.assertTrue(np.all((b>=.5)&(b<=1)))
        finally:a.REPS=previous

    def test_rejects_invalid_probability(self):
        with self.assertRaises(AssertionError):a.summarize_probabilities(np.ones((1,64)),np.array([0]),np.array([0]))
        with self.assertRaises(AssertionError):a.summarize_probabilities(np.full((1,64),1/64),np.array([1]),np.array([1]))

if __name__=='__main__':unittest.main()
