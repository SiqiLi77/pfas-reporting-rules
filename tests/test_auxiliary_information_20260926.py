import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_auxiliary_information_20260926 import BITS,COUNTS,SIX,cmi,feature_matrix,scores,shuffle_aux

class AuxiliaryTests(unittest.TestCase):
    def frame(self):
        d=pd.DataFrame({'region':['1']*8,'water_type':['GW']*8,'pws_size':['S']*8,
                        'month_sin':[0.]*8,'month_cos':[1.]*8,'aux_report__PFBA':[0,1]*4,
                        'aux_logratio__PFBA':[0.,1.]*4})
        for a in SIX:
            d['old_visible__'+a]=0;d['old_logratio__'+a]=0.;d['y_native_detect__'+a]=0
        return d

    def test_target_values_never_features(self):
        d=self.frame();x,s=feature_matrix(d,['PFBA'])
        for a in SIX:d['y_native_detect__'+a]=1
        np.testing.assert_array_equal(x,feature_matrix(d,['PFBA'],s)[0])
        self.assertFalse(any(n.startswith('y_') for n in s['names']))

    def test_conditional_information(self):
        y=np.tile([1,2],50);k=COUNTS[y];a=np.tile([0,1],50)
        self.assertAlmostEqual(cmi(a,y,k),np.log(2))
        self.assertAlmostEqual(cmi(np.zeros(100,dtype=int),y,k),0.)

    def test_joint_normalization_without_observed_k(self):
        rng=np.random.default_rng(7);n=20
        logits=rng.normal(size=(n,64));old=rng.integers(0,2,(n,6));count=rng.normal(size=(n,7))
        allowed=np.arange(7)[None,:]>=old.sum(1)[:,None]
        mass=np.exp(count-count.max(1,keepdims=True))*allowed;mass/=mass.sum(1,keepdims=True)
        q=np.zeros((n,64))
        for k in range(7):
            ix=COUNTS==k;okay=np.all(BITS[None,ix,:]>=old[:,None,:],axis=2)
            z=np.exp(logits[:,ix]-logits[:,ix].max(1,keepdims=True))*okay
            total=z.sum(1,keepdims=True);p=np.divide(z,total,out=np.zeros_like(z),where=total>0)
            q[:,ix]=p*mass[:,k,None]
        np.testing.assert_allclose(q.sum(1),1,atol=1e-12)
        np.testing.assert_allclose((q@BITS)[old==1],1,atol=1e-12)
        y=old.copy();s=scores(q,y)
        np.testing.assert_allclose(s[:,0],s[:,1]+s[:,2],atol=1e-12)

    def test_count_boundaries_identity_zero(self):
        q=np.ones((2,64))/64;y=np.array([[0]*6,[1]*6])
        np.testing.assert_allclose(scores(q,y)[:,2],0,atol=1e-12)

    def test_shuffle_preserves_auxiliary_vectors_not_targets(self):
        d=self.frame();z,a=shuffle_aux(d,['PFBA'],17)
        np.testing.assert_array_equal(d[[f'y_native_detect__{a}' for a in SIX]],z[[f'y_native_detect__{a}' for a in SIX]])
        self.assertEqual(sorted(d.aux_logratio__PFBA),sorted(z.aux_logratio__PFBA))
        np.testing.assert_array_equal(z.aux_report__PFBA,z.aux_logratio__PFBA)
        self.assertGreater(a['values_changed_fraction'],0)

    def test_unknown_category_and_training_scaling(self):
        d=self.frame();x,s=feature_matrix(d,[]);d.loc[0,'region']='99'
        y,s2=feature_matrix(d,[],s)
        self.assertEqual(s,s2);self.assertEqual(x.shape,y.shape);self.assertTrue(np.isfinite(y).all())

if __name__=='__main__':unittest.main()
