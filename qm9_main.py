import torch
import timeit
import itertools as it
from torch import bmm, matmul, float32
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.nn.utils.parametrizations import orthogonal
from torch import nn
import timeit
import itertools as it
import torch
from torch import bmm, matmul, float32
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.nn.utils.parametrizations import orthogonal
from torch import nn
from scipy.sparse import random
from scipy.stats import rv_continuous
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
from scipy.stats import bernoulli




class Embed():
    def __init__(self, sparse=False, density = 1, delta = 1.):
        if torch.cuda.is_available():
            print('cuda')
            self.device = torch.device('cuda')
        else:
            print('cpu')
            self.device = torch.device('cpu')
        self.d = 3
        self.sparse=sparse
        self.density=density    
        self.delta = delta
        #self.n = 29 #placeholder
        self.device = torch.device(self.device)
        self.rng = default_rng()
        self.seed = 42
        self.rvs = stats.bernoulli(.5).rvs
    def centralize(self, cloud):
        """
        Recieve 3xn point cloud
        Returns centralized point cloud
        """
        mult = torch.ones(self.n, self.n).to(self.device)
        diff = torch.div(1,torch.clamp(torch.tensor(self.n), min=1))*torch.matmul(cloud, mult).to(self.device)
        val = torch.sub(cloud, diff).clone().to(self.device)
        cloud = val.clone().to(self.device)
        return cloud
    def normalize_features(self, features):
        """
        recieves 1xn feature vector
        returns normalized feature vector
        """
        return F.normalize(features, dim=0).to(self.device) #normalize z values
    def process_cloud(self, cloud):
        """
        recieves 4xn QM9 cloud
        returns centralized positions cloud and normalized node features
        """
        d_cloud = self.centralize(cloud[:3,:].clone().to(self.device))
        lower = self.normalize_features(cloud[3,:].clone().to(self.device))
        self.cloud = torch.vstack((d_cloud,lower)).to(self.device)
    def wf(self):
        if self.sparse:
            S = random(self.embed_dim, 3, density=self.density, random_state=self.rng, data_rvs=self.rvs)
            w = 0.01*torch.tensor(S.A, dtype=torch.float32).reshape(1, self.embed_dim, 3).to(self.device)
        elif not self.sparse:
            torch.manual_seed(self.seed)
            w = 0.01*torch.randn(1, self.embed_dim, 3).to(self.device)
        w = w.repeat(self.num, 1, 1).to(self.device)
        w = torch.reshape(w, (self.num*self.embed_dim, 3, 1)).to(self.device)
        self.w = w
    def Wf(self):
        if self.sparse:
            S = random( self.embed_dim, self.n-2, density=self.density, random_state=self.rng, data_rvs=self.rvs)
            W= 0.01*torch.tensor(S.A, dtype=torch.float32).reshape(1, self.embed_dim, self.n-2).to(self.device)
        elif not self.sparse:
            torch.manual_seed(self.seed)
            W= 0.01*torch.randn(1, self.embed_dim, self.n-2).to(self.device)
        W = W.repeat(self.num, 1, 1).to(self.device)
        W = torch.reshape(W, (self.num*self.embed_dim, 1, (self.n-2))).to(self.device)
        self.W = W
    def sovec_scalars(self):
        torch.manual_seed(self.seed)
        all_idx_list = torch.arange(self.n).to(self.device)
        perms = torch.tensor(list(it.permutations(all_idx_list, r=2))).to(self.device)
        perms_one_hot = one_hot(perms, num_classes=self.n).to(float32).to(self.device)
        kaboom = torch.bmm(perms_one_hot.transpose(1,2), perms_one_hot)
        eyes = torch.reshape(F.one_hot(torch.arange(self.num*self.n) % self.n), (self.num,self.n,self.n)).to(self.device)
        comp = torch.sub(eyes, kaboom).to(self.device)
        comp_indices = torch.reshape(torch.nonzero(comp, as_tuple=True)[1], (self.num,(self.n-2))).to(self.device) # list of index values that aren't chosen 
        comp_mat = F.one_hot( comp_indices, num_classes=self.n ).to(float32).squeeze().to(self.device)
        comp_mat = torch.transpose(comp_mat,1,2).to(self.device)
        ug_mat = torch.cat((torch.transpose(perms_one_hot,1,2), comp_mat), dim=2).to(self.device) # some games with the transpose
        cloud_big = self.cloud.expand(self.num, 4, self.n ).to(self.device)
        big = torch.bmm( cloud_big, ug_mat ).to(self.device)
        scalars = big[:,3,:2].clone().to(self.device)#decompose to lower 4th row and point cloud
        scalars = scalars.reshape(self.num, 2, 1).to(self.device)
        rest_scalars, indices = torch.sort(big[:,3,2:].clone(), dim=1)
        self.rest_scalars = rest_scalars.reshape(self.num, (self.n-2), 1).to(self.device)
        big = big[:,:3,:].to(self.device)
        a = big[:, :, 0].to(self.device)
        b = big[:, :, 1].to(self.device)
        c = torch.cross(a, b).reshape(self.num, 3,1).to(self.device) #torch.linalg.cross not deprecated
        sovec = torch.cat([c, big],dim= 2).to(self.device) # this is all the point clouds arranged such that taking their uuper gram matrix seals the deal
        self.sovec, self.scalars = sovec, scalars
    def distances(self):
        #######################################################################
        ##################dist matrix##########################################
        #######################################################################
        gram_part = torch.transpose(self.sovec[:,:,:3].clone(), 1,2).repeat(1,(self.n+1),1).to(self.device)
        onew = 3*torch.ones((self.n+1), dtype=torch.int32).to(self.device)
        gram_mul = torch.repeat_interleave(self.sovec.clone().transpose(1,2), onew, dim=1).to(self.device)
        pdist = nn.PairwiseDistance(p=2)
        output = pdist(gram_mul, gram_part).reshape(self.num,(self.n+1),3).transpose(1,2).to(self.device)
        #######################################################################
        ################# add norms to "diagonal"##############################
        #######################################################################
        actual_gram = torch.bmm( torch.transpose(self.sovec[:,:,:3].clone(), 1,2), self.sovec[:,:,:3].clone()).to(self.device)
        a = torch.diagonal(actual_gram, dim1=-2, dim2=-1).to(self.device)
        a = torch.diag_embed(a)#creates vector of zero valued matrices with diagonal entries corresponding to norms of gram matrix part
        add_candidate = output[:,:,:3].clone().to(self.device)
        addition = (add_candidate + a).to(self.device)
        output[:,:,:3] = addition.to(self.device)
        ug_so = torch.exp(torch.div((-1)*output.clone(), self.delta)).clone().to(self.device)
        gram, projections = torch.split( ug_so, [3, (self.n-2)] , dim=2 )
        ones = torch.mul(torch.ones(self.num),self.embed_dim).to(dtype=torch.int32).to(self.device) #just to pass to repeat interleave, means nothing
        projections = torch.repeat_interleave(projections, ones, dim=0).to(self.device)
        projections = projections.transpose(1,2).to(self.device)
        #######################################################################
        ##################initial embedding####################################
        #######################################################################
        unsorted = torch.bmm(projections, self.w).to(self.device)
        sorted, indices = torch.sort(unsorted, dim=1)
        dot_prod = torch.bmm( self.W, sorted).to(self.device)
        dot_prod = dot_prod.reshape((self.num, self.embed_dim, 1)).to(self.device)
        gram = gram.reshape((self.num, 9 , 1)).to(self.device)
        self.append = torch.cat(( self.scalars, self.rest_scalars,gram,dot_prod), dim=1).reshape(1, self.num, self.embed_dim + 9 + self.scald).to(self.device)
    def w_2f(self):
        if self.sparse:
            S = random(self.embed_dim, self.embed_dim + 9 + self.scald, density=self.density, random_state=self.rng, data_rvs=self.rvs)
            self.w_2 = 0.01*torch.tensor(S.A, dtype=torch.float32).reshape(self.embed_dim, self.embed_dim + 9 + self.scald, 1).to(self.device)
        if not self.sparse:
            torch.manual_seed(42)
            self.w_2 = torch.randn(self.embed_dim, self.embed_dim + 9 + self.scald, 1 ).to(self.device)*0.01
    def W_2f(self):
        if self.sparse:
            S = random(self.embed_dim, self.num, density=self.density, random_state=self.rng, data_rvs=self.rvs)
            self.W_2 = 0.01*torch.tensor(S.A, dtype=torch.float32).reshape(self.embed_dim, 1, self.num).to(self.device)
        if not self.sparse:
            torch.manual_seed(42)
            self.W_2 = torch.randn(self.embed_dim, 1, self.num ).to(self.device)*0.01
    def final_embed(self):
        embed_2 = self.append.expand( ( self.embed_dim, self.num, self.embed_dim + 9 + self.scald ) )
        first = torch.bmm(embed_2, self.w_2).to(self.device)
        first_sorted, indices = torch.sort(first, dim = 1)
        self.final = torch.bmm( self.W_2 , first_sorted).squeeze().to(self.device)
    def forward(self, cloud, n):
        self.n = n
        self.embed_dim = 2*3*self.n + 1
        self.num = self.n*(self.n-1)
        self.scald = self.n
        self.process_cloud(cloud)
        self.wf()
        self.Wf()
        self.sovec_scalars()
        self.distances()
        self.w_2f()
        self.W_2f()
        print(self.W)
        self.final_embed()
        return self.final.clone()


def main():
    print("hi")
    cloud = torch.randn(4, 9)
    embed = Embed()
    perm = torch.randperm(9)
    orth_linear = orthogonal(nn.Linear(3, 3))
    Q = orth_linear.weight
    upper = cloud[:3,:]
    upper = torch.matmul(Q, upper)
    lower = cloud[3,:]
    cloud2 = torch.vstack([upper, lower])
    print(torch.allclose(embed.forward(cloud, 9), embed.forward( cloud2[:, perm], 9)))

if __name__ == "__main__":
    main()
