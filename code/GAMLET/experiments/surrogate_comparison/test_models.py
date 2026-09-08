import unittest
import numpy as np
import torch
from .models import descriptor,listmle,RFFGaussianProcess
class SurrogateTests(unittest.TestCase):
    def test_listmle_orientation_shift_and_reference(self):
        scores=torch.tensor([-2.,0.,1.],requires_grad=True);target=torch.tensor([0.,1.,2.])
        expected=sum(torch.logsumexp(-scores[i:],0)+scores[i] for i in range(3))/3
        torch.testing.assert_close(listmle(scores,target),expected)
        self.assertLess(float(listmle(scores,target)),float(listmle(-scores,target)))
        torch.testing.assert_close(listmle(scores+100,target),expected)
        listmle(scores,target).backward();self.assertTrue(torch.isfinite(scores.grad).all())
    def test_descriptor_node_permutation_and_inactive_padding(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(2,7,3));a=rng.random((2,7,7))>.6;m=np.ones((2,7));c=rng.normal(size=(2,6));p=rng.permutation(7)
        ref=descriptor(x,a,m,c);np.testing.assert_allclose(ref,descriptor(x[:,p],a[:,p][:,:,p],m[:,p],c),atol=1e-12)
        xx=np.pad(x,((0,0),(0,2),(0,0)),constant_values=999);aa=np.pad(a,((0,0),(0,2),(0,2)),constant_values=True);mm=np.pad(m,((0,0),(0,2)))
        np.testing.assert_allclose(ref,descriptor(xx,aa,mm,c),atol=1e-12)
    def test_gp_matches_kernel_posterior(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(25,4));y=np.sin(x[:,0]);q=rng.normal(size=(8,4))
        net=RFFGaussianProcess(17,1,.1,features=20).fit(x,y);phi=net.phi(x);pq=net.phi(q)
        direct=pq@phi.T@np.linalg.solve(phi@phi.T+.1*np.eye(len(x)),y-y.mean())+y.mean()
        np.testing.assert_allclose(net.predict(q),direct,rtol=1e-9,atol=1e-10)
if __name__=='__main__':unittest.main()
