import math
import networkx as nx
import numpy as np
from experiments.network_robustness.problem import NetworkDesign, Task, graph_key


def cascade(adj, alpha, priority, reference=False):
    graph = nx.from_numpy_array(adj)
    def loads(g):
        if not reference:
            return nx.betweenness_centrality(g, normalized=False)
        result = dict.fromkeys(g.nodes, 0.)
        nodes = list(g)
        for i,u in enumerate(nodes):
            for v in nodes[i+1:]:
                if not nx.has_path(g,u,v):
                    continue
                paths=list(nx.all_shortest_paths(g,u,v))
                for path in paths:
                    for w in path[1:-1]:
                        result[w] += 1/len(paths)
        return result
    initial=loads(graph)
    capacities={v:(1+alpha)*value for v,value in initial.items()}
    attacked=min(graph, key=lambda v:(-initial[v], priority[v]))
    graph.remove_node(attacked)
    rounds=[[int(attacked)]]
    while graph:
        current=loads(graph)
        failed=[v for v in graph if current[v] > capacities[v]+1e-9]
        if not failed:
            break
        rounds.append(sorted(map(int,failed)))
        graph.remove_nodes_from(failed)
    return max(map(len,nx.connected_components(graph)),default=0)/len(adj), rounds


class CascadeDesign(NetworkDesign):
    node_dim=6
    def __init__(self, seed, index, split):
        n=(32,48)[index%2]
        task=Task(f'{split}_{index:02d}',split,('er','ba')[(index//2)%2],n,seed,seed+500)
        super().__init__(task,dict(max_changed_fraction=.2,attack_scenarios=1))
        self.alpha=float(np.random.default_rng(seed+700).uniform(.1,.5))
        self.spec=dict(seed=seed,index=index,split=split,alpha=self.alpha,task=task.to_dict())
    def evaluate(self, x):
        assert self.valid(x)
        return cascade(x,self.alpha,self.priorities[0])[0]
    def features(self,x):
        nodes,adj,mask,ctx=super().features(x)
        ctx[-1]=self.alpha
        return nodes,adj,mask,ctx
    def mutate(self,x,rng):
        for _ in range(int(rng.integers(1,4))):
            x=self.swap(x,rng)
        return x
    @staticmethod
    def key(x): return graph_key(x)


def visible(a,b,rect):
    """Segment--closed rectangle slab intersection; endpoints are sampled outside."""
    lo,hi=0.,1.
    for d in range(2):
        delta=b[d]-a[d]
        if abs(delta)<1e-12:
            if a[d]<rect[d] or a[d]>rect[d+2]: return True
        else:
            p=(rect[d]-a[d])/delta; q=(rect[d+2]-a[d])/delta
            lo=max(lo,min(p,q)); hi=min(hi,max(p,q))
            if lo>hi: return True
    return False


class CoverageDesign:
    node_dim=9
    def __init__(self,seed,index,split):
        rng=np.random.default_rng(seed)
        centers=rng.uniform(.2,.8,(2,2)); widths=rng.uniform(.06,.14,(2,2))
        self.rects=np.concatenate((centers-widths/2,centers+widths/2),axis=1)
        points=[]
        while len(points)<64:
            p=rng.uniform(0,1,2)
            if not any(np.all(p>=r[:2]) and np.all(p<=r[2:]) for r in self.rects): points.append(p)
        self.xy=np.array(points); self.weights=rng.lognormal(0,1,40)
        self.weights/=self.weights.sum()
        delta=self.xy[24:][None,:,:]-self.xy[:24,None,:]
        self.distance=np.linalg.norm(delta,axis=2)
        self.bearing=np.arctan2(delta[:,:,1],delta[:,:,0])
        self.los=np.array([[all(visible(a,b,r) for r in self.rects) for b in self.xy[24:]] for a in self.xy[:24]])
        self.adj=np.zeros((64,64),np.float32)
        potential=self.los & (self.distance<=.55)
        self.adj[:24,24:]=potential; self.adj[24:,:24]=potential.T
        self.spec=dict(seed=seed,index=index,split=split,points=self.xy.tolist(),weights=self.weights.tolist(),obstacles=self.rects.tolist())
    def sample(self,rng):
        x=np.full(24,np.nan)
        x[rng.choice(24,6,replace=False)]=rng.uniform(-math.pi,math.pi,6)
        return x
    def mutate(self,x,rng):
        x=x.copy()
        for _ in range(int(rng.integers(1,4))):
            on=np.flatnonzero(np.isfinite(x))
            i=int(rng.choice(on))
            if rng.random()<.35:
                j=int(rng.choice(np.flatnonzero(~np.isfinite(x))))
                x[j]=x[i]; x[i]=np.nan; i=j
            x[i]=(x[i]+rng.normal(0,.45)+math.pi)%(2*math.pi)-math.pi
        return x
    @staticmethod
    def key(x): return np.nan_to_num(x,nan=10.).astype('<f8').tobytes()
    def evaluate(self,x):
        on=np.flatnonzero(np.isfinite(x)); assert len(on)==6
        diff=np.arctan2(np.sin(self.bearing[on]-x[on,None]),np.cos(self.bearing[on]-x[on,None]))
        p=.95*np.exp(-self.distance[on]/.8)*self.los[on]*(self.distance[on]<=.55)*(np.abs(diff)<=math.pi/4)
        return float(self.weights @ (1-np.prod(1-p,axis=0)))
    def reference(self,x):
        total=0.
        for j,b in enumerate(self.xy[24:]):
            miss=1.
            for i in np.flatnonzero(np.isfinite(x)):
                d=math.dist(self.xy[i],b)
                angle=math.atan2(b[1]-self.xy[i,1],b[0]-self.xy[i,0])-x[i]
                angle=math.atan2(math.sin(angle),math.cos(angle))
                if d<=.55 and abs(angle)<=math.pi/4 and all(visible(self.xy[i],b,r) for r in self.rects):
                    miss*=1-.95*math.exp(-d/.8)
            total+=self.weights[j]*(1-miss)
        return total
    def features(self,x):
        nodes=np.zeros((64,9),np.float32); nodes[:,:2]=self.xy
        on=np.isfinite(x); nodes[:24,2]=1; nodes[:24,3]=on
        nodes[:24,4]=np.where(on,np.cos(x),0); nodes[:24,5]=np.where(on,np.sin(x),0)
        nodes[24:,6]=self.weights*40; nodes[:24,7]=.55; nodes[:24,8]=.25
        return nodes,self.adj,np.ones(64,np.float32),np.array([24/100,40/100,6/24,.55,.25,1],np.float32)
