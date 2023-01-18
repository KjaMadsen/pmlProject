import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels = 32, hiddenDim=2, kernel_size = 3):
        super(ConvVAE, self).__init__()
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