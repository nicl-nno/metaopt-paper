"""Alternative losses and invariant classical surrogate adapters."""
import copy,hashlib,time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.linalg import cho_factor,cho_solve
from sklearn.ensemble import RandomForestRegressor
from experiments.gendesign_rank.model import RepositoryGraphSurrogate

def listmle(scores,targets):
    # Stable sorting randomizes exact ties via the already shuffled minibatch.
    order=torch.argsort(targets,stable=True)
    logits=-scores[order]
    return (torch.logcumsumexp(logits.flip(0),0).flip(0)-logits).mean()

def descriptor(nodes,adjacency,mask,context):
    """Permutation-invariant moments of node features and two diffusion steps.

    Only the same GNN inputs are used; no objective-derived features are added.
    Input adjacency weights are converted to connectivity and row normalized.
    """
    n,a,m,c=[v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for v in (nodes,adjacency,mask,context)]
    result=[]
    for lo in range(0,len(n),128):
        x=n[lo:lo+128].astype(np.float64);active=m[lo:lo+128]>0
        edges=(a[lo:lo+128]!=0)&active[:,:,None]&active[:,None,:]
        count=active.sum(1)[:,None];assert (count>0).all()
        weights=edges/np.maximum(edges.sum(2,keepdims=True),1)
        def moments(h):
            mean=(h*active[:,:,None]).sum(1)/count
            var=(((h-mean[:,None,:])**2)*active[:,:,None]).sum(1)/count
            return [mean,np.sqrt(np.maximum(var,0)),np.where(active[:,:,None],h,np.inf).min(1),np.where(active[:,:,None],h,-np.inf).max(1)]
        h=x;pieces=moments(h)
        for _ in range(2):h=weights@h;pieces+=moments(h)
        pieces += [(np.abs(x-weights@x)*active[:,:,None]).sum(1)/count,c[lo:lo+128],np.log1p(count),np.log1p(edges.sum((1,2)))[:,None]]
        result.append(np.concatenate(pieces,axis=1))
    out=np.concatenate(result);assert np.isfinite(out).all();return out

def standardized_targets(datasets,kind,transform,normalization):
    values=[d['values'] if kind=='truss' else 1-d['values'] for d in datasets]
    if transform=='log':values=[np.log(np.maximum(v,1e-12)) for v in values]
    pooled=np.concatenate(values);mean,std=pooled.mean(),max(pooled.std(),1e-8)
    return [(y-mean)/std if normalization=='pooled' else (y-y.mean())/max(y.std(),1e-8) for y in values]

def validation(net,datasets,kind):
    scores=[]
    for d in datasets.values():
        if d['split']!='validation':continue
        with torch.no_grad():j=int(net(*d['features']).argmin())
        v=d['values'];scores.append(np.log(v[j]/v.min()) if kind=='truss' else np.log(v.max()/v[j]))
    return float(np.mean(scores))

def train_gnn(recipe,seed,data,cfg,kind,progress):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    first=next(iter(data.values()))['features']
    net=RepositoryGraphSurrogate(node_dim=first[0].shape[-1],context_dim=6,hidden=32,layers=3)
    initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.parameters())).hexdigest()
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4)
    train=[d for d in data.values() if d['split']=='train']
    ys=[torch.tensor(y,dtype=torch.float32) for y in standardized_targets(train,kind,recipe.get('transform','raw'),'task')]
    best=float('inf');state=None;history=[];steps=0;tick=time.time()
    for epoch in range(cfg['epochs']):
        net.train()
        for i in rng.permutation(len(train)):
            order=rng.permutation(len(ys[i]))
            for offset in range(0,len(order),cfg['batch_size']):
                idx=order[offset:offset+cfg['batch_size']];pred=net(*(x[idx] for x in train[i]['features']));target=ys[i][idx]
                loss=listmle(pred,target) if recipe['method']=='listmle' else F.huber_loss(pred,target,delta=1.)
                assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();steps+=1
        net.eval();regret=validation(net,data,kind);history.append(dict(epoch=epoch+1,validation_log_regret=regret))
        if regret<best:best=regret;state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        progress(dict(epoch=epoch+1,epochs=cfg['epochs'],validation_log_regret=regret))
        if epoch==0 or (epoch+1)%20==0:print(kind,recipe,seed,'epoch',epoch+1,'validation',regret,flush=True)
    net.load_state_dict(state);net.eval()
    return net,dict(recipe=recipe,model_seed=seed,initial_weights_sha256=initial,optimizer_steps=steps,parameters=sum(v.numel() for v in net.parameters()),validation_log_regret=best,selected_epoch=best_epoch,history=history,seconds=time.time()-tick)

class Classical:
    def eval(self):return self
    def __call__(self,*features):return torch.tensor(self.predict(descriptor(*features)),dtype=torch.float64)

class Forest(Classical):
    def __init__(self,seed,leaf):self.model=RandomForestRegressor(n_estimators=128,min_samples_leaf=leaf,max_features=.5,bootstrap=True,random_state=seed,n_jobs=1)
    def fit(self,x,y):self.model.fit(x,y);return self
    def predict(self,x):return self.model.predict(x)

class RFFGaussianProcess(Classical):
    """Posterior mean under a finite-feature approximation of an RBF GP.

    No uncertainty acquisition: the common top-3 plus random policy is retained.
    Ridge is observation noise variance with unit prior covariance on weights.
    """
    def __init__(self,seed,length,noise,features=256):self.seed,self.length,self.noise,self.features=seed,length,noise,features
    def fit(self,x,y):
        self.mean=x.mean(0);self.scale=np.maximum(x.std(0),1e-8);z=(x-self.mean)/self.scale
        rng=np.random.default_rng(self.seed+91000)
        left=rng.integers(len(z),size=1024);right=rng.integers(len(z),size=1024)
        self.distance=max(float(np.median(np.linalg.norm(z[left]-z[right],axis=1))),1e-8)
        self.omega=rng.normal(size=(x.shape[1],self.features))/(self.length*self.distance)
        self.phase=rng.uniform(0,2*np.pi,size=self.features)
        phi=self.phi(x);self.target_mean=y.mean();target=y-self.target_mean
        self.weights=cho_solve(cho_factor(phi.T@phi+self.noise*np.eye(self.features),lower=True),phi.T@target)
        return self
    def phi(self,x):return np.sqrt(2/self.features)*np.cos(((x-self.mean)/self.scale)@self.omega+self.phase)
    def predict(self,x):return self.phi(x)@self.weights+self.target_mean

def recipes(method):
    if method=='listmle':return [dict(method=method,transform='raw')]
    if method=='huber':return [dict(method=method,transform=t) for t in ['raw','log']]
    base=[dict(method=method,transform=t,normalization=n) for t in ['raw','log'] for n in ['task','pooled']]
    if method=='rf':return [dict(b,leaf=l) for b in base for l in [1,5]]
    if method=='rff_gp':return [dict(b,length=l,noise=n) for b in base for l in [.5,1.,2.] for n in [.01,.1]]
    raise ValueError(method)
