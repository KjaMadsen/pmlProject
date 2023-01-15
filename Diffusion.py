#https://nn.labml.ai/diffusion/ddpm/index.html
#https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=Rj17psVw7Shg
# https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=3s 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

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
        coef = (1-alpha) / (1-alpha_bar)**0.5
        mean = 1 / torch.sqrt(alpha) * (x_t - coef * params)
        var = self.extract(self.betas, t, x_t)
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
    
if __name__=="__main__":
    cuda = torch.cuda.is_available()
    batch_size = 128
    log_interval = 10
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
        batch_size=batch_size, shuffle=True, **kwargs)


    
    eps_model = deNoise()
    eps_model.to(device)
    diff = denoisingDiffusion(eps_model, device=device)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=1e-3)
    eps_model.train() # so that everything has gradients and we can do backprop and so on...
    train_loss = 0
    for epoch in range(1, num_epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad() # "reset" gradients to 0 for text iteration
            
            loss = diff.loss(data)
            loss.backward() # calc gradients
            train_loss += loss.item()
            optimizer.step() # backpropagation

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        if True:
            with torch.no_grad():
                sample = torch.randn(28, 28).to(device)
                sample = sample.unsqueeze(0)
                t = torch.arange(diff.T, 0, step=-1, device=device)
                sample = diff.sample_p(sample, t)
                save_image(sample.cpu(),
                            'results/diffusion/sample_' + str(epoch) + '.png')