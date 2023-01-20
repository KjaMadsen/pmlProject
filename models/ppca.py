import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class PPCA():

    def __init__(self, dataset, L):
        self.dataset = dataset
        data = self.dataset.data
        #PCA entire dataset
        self.L = L
        '''
        if len(data.shape) > 2:
            self.data = data.reshape((60000, 28*28))
        else:
            self.data = data
        self.cov = torch.cov(self.data.T)
        self.eigen = torch.linalg.eig(self.cov)
        self.eigenvec = self.eigen.eigenvectors
        self.eigenval = self.eigen.eigenvalues
      
        self.dim = self.eigenvec.shape[1]
        
        self.sigma2 = 1/(self.dim-L)*torch.sum(self.eigenval[L:])
        self.W = self.eigenvec[:,:L] @ (torch.diag(self.eigenval[:L])-self.sigma2*torch.eye(L)**0.5) 
        self.mu = torch.mean(self.data.type(torch.double), 0)
    
        M = self.W.T @ self.W + self.sigma2 + torch.eye(L)
        self.x_mu = self.data.T - self.mu.repeat(self.data.shape[0],1).T
        temp = torch.inverse(M)@self.W.T
        self.z = temp.type(torch.double)@self.x_mu
        '''

    def sample(self, save = True):
        samples = []
        for s in self.compute_pca_loop():
            digit, sample = s
            samples.append(sample)
            if save:
                save_image(sample.view(28, 28).cpu(),
                    'results/PPCA/sample_' + str(digit) + '.png')
        return samples
        
    def compute_pca_loop(self):
        samples = []
        for i in range(10):
            idx = self.dataset.train_labels == i
            data = self.dataset.train_data[idx]
            z, W, x_mu, sigma2 = self.compute_pca(data)
            sample = self.generate_sample(z, W, x_mu, sigma2)
            samples.append((i,sample))
        return samples
    
    def compute_pca(self, data):
        u = data.shape[0]
        data = data.reshape((u, 28*28))
        cov = torch.cov(data.T)
        eigen = torch.linalg.eig(cov)
        eigenvec = eigen.eigenvectors
        eigenval = eigen.eigenvalues
      
        dim = eigenvec.shape[1]
        sigma2 = 1/(dim-self.L)*torch.sum(eigenval[self.L:])
        W = eigenvec[:,:self.L] @ (torch.diag(eigenval[:self.L])-sigma2*torch.eye(self.L)**0.5) 
        mu = torch.mean(data.type(torch.double), 0)
        M = W.T @ W + sigma2 * torch.eye(self.L)
        x_mu = data.T - mu.repeat(data.shape[0],1).T
        temp = torch.inverse(M)@W.T
        z = temp.type(torch.double)@mu
        return z, W.type(torch.double), mu, sigma2

    def generate_sample(self, z, W, mu, sigma2):
        sigma = torch.sqrt(sigma2.type(torch.double))
        return W@z + mu + sigma*torch.randn_like(mu)
       

if __name__=="__main__":
    cuda = torch.cuda.is_available()
    batch_size = 1
    torch.manual_seed(1) # args.seed

    device = torch.device("cuda" if cuda else "cpu") # args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {} # args.cuda

    # Get train and test data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    ppca = PPCA(train_loader.dataset, 5)
    samples = ppca.sample()
   