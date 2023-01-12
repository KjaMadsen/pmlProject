class PPCA():

    def __init__(self, dataset, L):
        self.dataset = dataset
        data = self.dataset.data
        if len(data.shape) > 2:
            self.data = data.reshape((60000, 28*28))
        else:
            self.data = data
        self.cov = torch.cov(self.data.T)
        self.eigen = torch.linalg.eig(self.cov)
        self.eigenvec = self.eigen.eigenvectors
        self.eigenval = self.eigen.eigenvalues
      
        self.dim = self.eigenvec.shape[1]
        self.L = L
        self.sigma2 = 1/(self.dim-L)*torch.sum(self.eigenval[L:])
        self.W = self.eigenvec[:,:L] @ (torch.diag(self.eigenval[:L])-self.sigma2*torch.eye(L)**0.5) 
        self.mu = torch.mean(self.data.type(torch.double), 0)
    
        M = self.W.T @ self.W + self.sigma2 + torch.eye(L)
        self.x_mu = self.data.T - self.mu.repeat(self.data.shape[0],1).T
        temp = torch.inverse(M)@self.W.T
        self.z = temp.type(torch.double)@self.x_mu
    
    def plot(self, n=0):
        return
    
    
    def generate_sample(self):
        prior = torch.distributions.Normal(torch.zeros((self.L, self.L)), 1)
        z_hat = prior.sample().type(torch.double)
       
        posterior = torch.distributions.Normal((self.W.type(torch.double)@z_hat).T + self.mu, self.sigma2.type(torch.double))
        print(posterior.loc.shape)
        
        return posterior.sample()
       
      