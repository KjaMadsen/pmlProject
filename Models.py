import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# original VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc1a = nn.Linear(400, 100)
        self.fc21 = nn.Linear(100, 2) # Latent space of 2D
        self.fc22 = nn.Linear(100, 2) # Latent space of 2D
        self.fc3 = nn.Linear(2, 100) # Latent space of 2D
        self.fc3a = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc1a(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc3a(h3))
        return torch.sigmoid(self.fc4(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



class convVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels = 32, hiddenDim=64, kernel_size = 3):
        super(convVAE, self).__init__()
        w = 28+1+1-2*kernel_size
        features=torch.tensor([out_channels,w,w])
        self.num_features = torch.prod(features, dim=0).item()
        self.ft_shape = features
        out_channels1 = 16
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(out_channels1, out_channels=out_channels, kernel_size=kernel_size)
  
        
        self.enFc1 = nn.Linear(self.num_features, hiddenDim)
        self.enFc2 = nn.Linear(self.num_features, hiddenDim)

        self.deFc1 = nn.Linear(hiddenDim, self.num_features)
        self.deConv1 = nn.ConvTranspose2d(out_channels, out_channels1, kernel_size=kernel_size)
        self.deConv2 = nn.ConvTranspose2d(out_channels1, in_channels, kernel_size=kernel_size)


    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        
        h3 = h2.view(-1, self.num_features)
        return self.enFc1(h3), self.enFc2(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.deFc1(z))
        h5 = F.relu(self.deConv1(h4.view(-1, self.ft_shape[0], self.ft_shape[1], self.ft_shape[2])))
        return torch.sigmoid(self.deConv2(h5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class denoisingDiffusion:
    def __init__(self, model : nn.Module, T=1000, device='cpu'):
        self.T = T
        start = 1e-5; end = 1e-2
        self.betas = torch.linspace(start, end, T, device=device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.device = device
        self.model = model

    def sample_q(self, x0, t, noise): #this and 'loss()' makes algorithm 1

        a_t = self.extract(torch.sqrt(self.alphas_bar), t, x0)
        one_minus_a_t = self.extract(torch.sqrt(1-self.alphas_bar), t, x0)
        return (a_t * x0  + one_minus_a_t * noise)

    def loss(self, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=x0.device, dtype=torch.long)
        
        noise=torch.rand_like(x0)
        x_t = self.sample_q(x0, t, noise)
        eps_theta = self.model(x_t, t)

        return F.mse_loss(noise, eps_theta)

    def sample_p(self, x_t, t):
        alpha = self.extract(self.alphas, t, x_t)
        alpha_bar = self.extract(self.alphas_bar, t, x_t)
        params = self.model(x_t, t)
        coef = self.betas / (self.extract(torch.sqrt(1-self.alphas_bar), t, x_t))
    
        print(params.shape)
        print(coef.shape)
        print((coef * params))
        mean = 1 / torch.sqrt(alpha) * (x_t - coef * params)
        var = self.extract(self.betas, t)
        eps = torch.randn_like(x_t, device=x_t.device)
        return mean + torch.sqrt(var) * eps

    def sample_p_loop(self, shape):
        x_t = torch.randn(shape, device=self.device)
        Xs = [x_t]
        for t in reversed(range(self.T)):
            x_t = self.sample_p(x_t, x_t.new_full((x_t.shape[0],), t, dtype=torch.long))
            Xs.append(x_t)
        
        return Xs
    
    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)


class deNoise(nn.Module):
    def __init__(self, in_channels=1, out_channels = 32, hiddenDim=64, kernel_size = 3):
        super(deNoise, self).__init__()
        
        w = 28+1+1-2*kernel_size
        features=torch.tensor([out_channels,w,w])
        self.num_features = torch.prod(features, dim=0).item()
        self.ft_shape = features
        out_channels1 = 16

        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size)
        self.conv2 = nn.Conv2d(out_channels1, out_channels, kernel_size)
        self.fc1 = nn.Linear(self.num_features, hiddenDim)
        
        self.fc2 = nn.Linear(hiddenDim, self.num_features)
        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels1, kernel_size)
        self.conv4 = nn.ConvTranspose2d(out_channels1, in_channels, kernel_size)
        
    def forward(self, x, t):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.fc1(x.view(-1, self.num_features)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.conv3(x.view(-1, self.ft_shape[0], self.ft_shape[1], self.ft_shape[2])))
        x = torch.sigmoid(self.conv4(x))
        return x

class BayesianLinear(nn.Module):
    #inspiration https://github.com/Harry24k/bayesian-neural-network-pytorch
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
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
  
        self.reset_parameters()

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sample = self.weight_mu + torch.exp(self.weight_std) * torch.rand_like(self.weight_std)
        bias_sample = self.bias_mu + torch.exp(self.bias_std) * torch.rand_like(self.bias_std)
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
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        rec = self.decode(z)
        return rec, mu, logvar




