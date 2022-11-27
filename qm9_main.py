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


num = 29*28
if torch.cuda.is_available():
    print('cuda')
    device = torch.device('cuda')
else:
    print('cpu')
    device = torch.device('cpu')

seed = 42

class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)


class Embed():
    def __init__(self, device='cuda', sparse=False, density = 1, delta = 1.) -> None:
        self.d = 3
        self.sparse=sparse
        self.density=density    
        self.delta = delta
        self.n = 29
        self.device = torch.device(device)
        self.rng = default_rng()
        self.embed_dim = 2*3*self.n + 1
        self.num = self.n*(self.n-1)
        self.scald = self.n
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
    def w(self):
        if self.sparse:
            S = random(self.embed_dim, 3, density=self.density, random_state=self.rng, data_rvs=self.rvs)
            w = 0.01*torch.tensor(S.A, dtype=torch.float32).reshape(1, self.embed_dim, 3).to(self.device)
        elif not self.sparse:
            torch.manual_seed(self.seed)
            w = 0.01*torch.randn(1, self.embed_dim, 3).to(self.device)
        w = w.repeat(self.num, 1, 1).to(self.device)
        w = torch.reshape(w, (self.num*self.embed_dim, 3, 1)).to(self.device)
        self.w = w
    def W(self):
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
        torch.manual_seed(seed)
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
        cloud_big = self.cloud.expand(self.num, 3, self.n ).to(self.device)
        big = torch.bmm( cloud_big, ug_mat ).to(self.device)
        scalars = big[:,3,:2].clone().to(self.device)#decompose to lower 4th row and point cloud
        scalars = scalars.reshape(self.num, 2, 1).to(self.device)
        rest_scalars, indices = torch.sort(big[:,3,2:].clone(), dim=1)
        rest_scalars = rest_scalars.reshape(self.num, (self.n-2), 1).to(self.device)
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
        unsorted = torch.bmm(projections, self.w).to(self.device)
        sorted, indices = torch.sort(unsorted, dim=1)
        dot_prod = torch.bmm( self.W, sorted).to(self.device)
        dot_prod = dot_prod.reshape((self.num, self.embed_dim, 1)).to(self.device)
        gram = gram.reshape((self.num, 9 , 1)).to(self.device)
        self.append = torch.cat(( self.scalars, self.rest_scalars,gram,dot_prod), dim=1).reshape(1, self.num, self.embed_dim + 9 + self.scald).to(self.device)
    def w_2(self):
        if sparse:
        S = random(embed_dim, embed_dim + 9 + scald, density=density, random_state=rng, data_rvs=rvs)
        w_2 = .01*torch.tensor(S.A, dtype=torch.float32).reshape(embed_dim, embed_dim + 9 + scald, 1).to(device)
    if not sparse:
        torch.manual_seed(42)
        w_2 = torch.randn(embed_dim, embed_dim + 9 + scald, 1 ).to(device)*0.01

    def W_2(self):
        if sparse:
        S = random(embed_dim, num, density=density, random_state=rng, data_rvs=rvs)
        W_2 = .01*torch.tensor(S.A, dtype=torch.float32).reshape(embed_dim, 1, num).to(device)
    if not sparse:
        torch.manual_seed(42)
        W_2 = torch.randn(embed_dim, 1, num ).to(device)*0.01
    def forward(self, cloud, n):
        self.n = n
        cloud = self.process_cloud(cloud)


def forward(cloud, sparse=False,density=1,delta = 1., n=8):
    d_cloud = cloud[:3,:].clone().to(device)
    lower = cloud[3,:].clone().to(device)
    lower = F.normalize(lower, dim=0).to(device) #normalize z values
    mult = torch.ones(n, n).to(device)
    diff = torch.div(1,torch.clamp(torch.tensor(n), min=1))*torch.matmul(d_cloud, mult).to(device)
    val = torch.sub(d_cloud, diff).clone().to(device)
    cloud = torch.vstack((val,lower)).to(device)
    torch.manual_seed(seed)
    rng = default_rng()
    embed_dim = 2*3*n + 1
    num = n*(n-1)
    scald = n
    if sparse:
        rvs = stats.bernoulli(.5).rvs
        S = random(embed_dim, 3, density=density, random_state=rng, data_rvs=rvs)
        w = .01*torch.tensor(S.A, dtype=torch.float32).reshape(1,embed_dim,3).to(device)
    if not sparse:
        torch.manual_seed(seed)
        w = 0.01*torch.randn(1, embed_dim, 3).to(device)
    w= w.repeat(num, 1, 1).to(device)
    w = torch.reshape(w, (num*embed_dim, 3, 1)).to(device)
    if sparse:
        S = random( embed_dim, 27, density=density, random_state=rng, data_rvs=rvs)
        W = .01*torch.tensor(S.A, dtype=torch.float32).reshape(1, embed_dim, 27).to(device)
    if not sparse:
        torch.manual_seed(seed)
        W = torch.randn(1, embed_dim, 27).to(device)*0.01
    W= W.repeat(num, 1, 1).to(device)
    W = torch.reshape(W, (num*embed_dim, 1, (n-2))).to(device)
    torch.manual_seed(seed)
    all_idx_list = torch.arange(n).to(device)
    perms = torch.tensor(list(it.permutations(all_idx_list, r=2))).to(device)
    perms_one_hot = one_hot(perms, num_classes=n).to(float32).to(device)
    kaboom = torch.bmm(perms_one_hot.transpose(1,2), perms_one_hot)
    eyes = torch.reshape(F.one_hot(torch.arange(num*n) % n), (num,n,n)).to(device)
    comp = torch.sub(eyes, kaboom).to(device)
    comp_indices = torch.reshape(torch.nonzero(comp, as_tuple=True)[1], (num,(n-2))).to(device) # list of index values that aren't chosen 
    comp_mat = F.one_hot( comp_indices, num_classes=n ).to(float32).squeeze().to(device)
    comp_mat = torch.transpose(comp_mat,1,2).to(device)
    ug_mat = torch.cat((torch.transpose(perms_one_hot,1,2), comp_mat), dim=2).to(device) # some games with the transpose
    cloud_big = cloud.expand(num, 3, n ).to(device)
    big = torch.bmm( cloud_big, ug_mat ).to(device)
    scalars = big[:,3,:2].clone().to(device)#decompose to lower 4th row and point cloud
    scalars = scalars.reshape(num, 2, 1).to(device)
    rest_scalars, indices = torch.sort(big[:,3,2:].clone(), dim=1)
    rest_scalars = rest_scalars.reshape(num, (n-2), 1).to(device)
    big = big[:,:3,:].to(device)
    a = big[:, :, 0].to(device)
    b = big[:, :, 1].to(device)
    c = torch.cross(a, b).reshape(num, 3,1).to(device) #torch.linalg.cross not deprecated
    sovec = torch.cat([c, big],dim= 2).to(device) # this is all the point clouds arranged such that taking their uuper gram matrix seals the deal
    #######################################################################
    ##################dist matrix##########################################
    #######################################################################
    gram_part = torch.transpose(sovec[:,:,:3].clone(), 1,2).repeat(1,(n+1),1).to(device)
    onew = 3*torch.ones((n+1), dtype=torch.int32).to(device)
    gram_mul = torch.repeat_interleave(sovec.clone().transpose(1,2), onew, dim=1).to(device)
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(gram_mul, gram_part).reshape(num,(n+1),3).transpose(1,2).to(device)
    #######################################################################
    ################# add norms to "diagonal"##############################
    #######################################################################
    actual_gram = torch.bmm( torch.transpose(sovec[:,:,:3].clone(), 1,2), sovec[:,:,:3].clone()).to(device)
    a = torch.diagonal(actual_gram, dim1=-2, dim2=-1).to(device)
    a = torch.diag_embed(a)#creates vector of zero valued matrices with diagonal entries corresponding to norms of gram matrix part
    add_candidate = output[:,:,:3].clone().to(device)
    addition = (add_candidate + a).to(device)
    output[:,:,:3] = addition.to(device)
    ug_so = torch.exp(torch.div((-1)*output.clone(),delta)).clone().to(device)
    gram, projections = torch.split( ug_so, [3, (n-2)] , dim=2 )
    ones = torch.mul(torch.ones(num),embed_dim).to(dtype=torch.int32).to(device) #just to pass to repeat interleave, means nothing
    projections = torch.repeat_interleave(projections, ones, dim=0).to(device)
    projections = projections.transpose(1,2).to(device)
    unsorted = torch.bmm(projections, w).to(device)
    sorted, indices = torch.sort(unsorted, dim=1)
    dot_prod = torch.bmm( W, sorted).to(device)
    dot_prod = dot_prod.reshape((num, embed_dim, 1)).to(device)
    gram = gram.reshape((num, 9 , 1)).to(device)
    append = torch.cat(( scalars, rest_scalars,gram,dot_prod), dim=1).reshape(1, num, embed_dim + 9 + scald).to(device)
    if sparse:
        S = random(embed_dim, embed_dim + 9 + scald, density=density, random_state=rng, data_rvs=rvs)
        w_2 = .01*torch.tensor(S.A, dtype=torch.float32).reshape(embed_dim, embed_dim + 9 + scald, 1).to(device)
    if not sparse:
        torch.manual_seed(42)
        w_2 = torch.randn(embed_dim, embed_dim + 9 + scald, 1 ).to(device)*0.01
    torch.manual_seed(42)
    if sparse:
        S = random(embed_dim, num, density=density, random_state=rng, data_rvs=rvs)
        W_2 = .01*torch.tensor(S.A, dtype=torch.float32).reshape(embed_dim, 1, num).to(device)
    if not sparse:
        torch.manual_seed(42)
        W_2 = torch.randn(embed_dim, 1, num ).to(device)*0.01
    embed_2 = append.expand( ( embed_dim, num, embed_dim + 9 + scald ) )
    first = torch.bmm(embed_2, w_2).to(device)
    first_sorted, indices = torch.sort(first, dim = 1)
    final = torch.bmm( W_2 , first_sorted).squeeze().to(device)
    return final
