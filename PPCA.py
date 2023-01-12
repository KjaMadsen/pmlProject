import numpy as np
import torch

class PPCA():

    def __init__(self, data, M):
        if len(data.shape) > 2:
            self.data = data.flatten(-1, data.shape[1:])
        else:
            self.data = data

        self.eigen = torch.eig(self.data.T)
        self.eigenvec = self.eigen.eigenvectors
        self.eigenval = self.eigen.eigenval
        self.cov = torch.eig(self.data)
        self.dim = self.eigenvec.shape[1]
        self.M = M
        self.sigma2 = 1/(self.dim-M)*torch.sum(self.eigenval[M:])
        self.W = self.eigenvec[:,:M] @ (torch.diag(self.eigenval[:M])-self.sigma2*torch.eye(M)**0.5) 
        self.mu = torch.mean(self.data, 0)

        M = self.W.T @ self.W + self.sigma2 + torch.eye(self.M)
        x_mu = self.data.T - self.mu.repeat(N,1).T
        temp = torch.inverse(M)@self.W.T
        z = temp@x_mu
    
    def plot():
        return