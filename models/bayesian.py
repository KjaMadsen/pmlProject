import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from tqdm import tqdm

class BayesianLinear(nn.Module):
    #inspiration https://github.com/Harry24k/bayesian-neural-network-pytorch
    __constants__ = ['in_features', 'out_features', 'prior']
    in_features: int
    out_features: int
    prior : torch.distributions
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, prior, bias: bool = True,
                 device=None, dtype=torch.float32) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_log_std = prior.stddev.item()
        self.prior = prior
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))
  
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_std = nn.Parameter(torch.Tensor(out_features))
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.reset_parameters()

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sample = (self.weight_mu + torch.exp(self.weight_std) * torch.rand_like(self.weight_std)).to(self.device)
        bias_sample = (self.bias_mu + torch.exp(self.bias_std) * torch.rand_like(self.bias_std)).to(self.device)
        return F.linear(input, sample, bias_sample)
    
    def KL_loss(self):
        weight_posterior = torch.distributions.Normal(self.weight_mu, self.weight_std)
        bias_posterior = torch.distributions.Normal(self.bias_mu, self.bias_std)
        KLD = torch.distributions.kl_divergence(weight_posterior, self.prior)
        KLD_bias = torch.distributions.kl_divergence(bias_posterior, self.prior)
        return (KLD.sum()+KLD_bias.sum()) / (torch.numel(KLD)+torch.numel(KLD_bias))
    
    def reset_parameters(self) -> None:
        std = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-std,std)
        self.weight_std.data.fill_(self.prior_log_std)
        self.bias_mu.data.uniform_(-std, std)
        self.bias_std.data.fill_(self.prior_log_std)

        


class BayesianVAE(nn.Module):
    def __init__(self, prior):
        super(BayesianVAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = BayesianLinear(784, 400, prior=prior)
        self.fc1a = BayesianLinear(400, 100, prior=prior)
        self.fc21 = BayesianLinear(100, 2, prior=prior)
        self.fc22 = BayesianLinear(100, 2, prior=prior)
        self.fc3 = BayesianLinear(2, 100, prior=prior)
        self.fc3a = BayesianLinear(100, 400, prior=prior)
        self.fc4 = BayesianLinear(400, 784, prior=prior)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc1a(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def KL_loss(self):
        kl = 0
        n = 0
        for m in self.children():
            kl += m.KL_loss().sum()
            n += 1
        return kl

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc3a(h3))
        return torch.sigmoid(self.fc4(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec = self.decode(z)
        return rec, mu, logvar

if __name__=="__main__":
    
    cuda = torch.cuda.is_available()
    batch_size = 64
    num_epochs = 10

    torch.manual_seed(1) # args.seed

    device = torch.device("cuda" if cuda else "cpu") # args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {} # args.cuda

    # Get train and test data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=1, shuffle=True, **kwargs)


    prior = torch.distributions.Normal(torch.tensor([0]), torch.tensor([1]))
    model = BayesianVAE(prior)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    reconstrction_loss = nn.CrossEntropyLoss()
    for epoch in range(1, num_epochs+1):
        model.train() # so that everything has gradients and we can do backprop and so on...
        train_loss = 0
        tqdm_ = tqdm(train_loader)
        for data,_ in tqdm_:
            data = data.view(-1, 28*28)
            data = data.to(device)
            optimizer.zero_grad() # "reset" gradients to 0 for text iteration
            recon_batch, mu, logvar = model(data)
            loss = model.KL_loss() + reconstrction_loss(recon_batch, data)
            loss.backward() # calc gradients
            train_loss += loss.item()
            optimizer.step() # backpropagation
            tqdm_.set_description(f"loss: {loss:.4f}")

        print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))



        model.eval()
        test_loss = 0
        with torch.no_grad(): # no_grad turns of gradients...
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += model.KL_loss().item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


