#Implemnted using inspiration from these sources
#https://arxiv.org/pdf/2006.11239.pdf (original paper)
#https://nn.labml.ai/diffusion/ddpm/index.html
#https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb
#https://github.com/cloneofsimo/minDiffusion/blob/00a3c8066d25a34d3d472ef6c87c7b6b238ea444/superminddpm.py#L83

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

class denoisingDiffusion(nn.Module):
    def __init__(self, model : nn.Module, T=1000, device='cpu'):
        super(denoisingDiffusion, self).__init__()
        self.T = T
        start = 1e-4; end = 0.02
        self.betas = torch.linspace(start, end, T+1, device=device)
        self.alphas = 1 - self.betas
        #self.alphas_bar = torch.cumprod(self.alphas, dim=0) #Not implemented for apple silicon
        alphas_log = torch.log(self.alphas)
        self.alphas_bar = torch.cumsum(alphas_log, dim=0).exp()
        self.device = device
        self.model = model

    def sample_q(self, x0, t, noise): #this and 'loss()' makes algorithm 1
        a_t = self.extract(torch.sqrt(self.alphas_bar), t, x0)
        one_minus_a_t = self.extract(torch.sqrt(1-self.alphas_bar), t, x0)
        return (a_t * x0  + one_minus_a_t * noise)

    def forward(self, x0): #algorithm 1
        batch_size = x0.shape[0]
        t = torch.randint(1, self.T, (batch_size,), device=x0.device, dtype=torch.long)
        
        noise=torch.randn_like(x0)
        x_t = self.sample_q(x0, t, noise)
        eps_theta = self.model(x_t, t)

        return F.mse_loss(noise, eps_theta)

    def sample_p(self, n=1): #alogrithm 2
        x_t = torch.randn(n, *(1,28,28)).to(self.device)
        for t in range(self.T, 0, -1):
            noise = torch.randn_like(x_t) if t > 1 else 0
            alpha = self.alphas[t]
            alpha_bar = self.alphas_bar[t]
            eps_theta = self.model(x_t, t)
            coef = (1-alpha) / (1-alpha_bar)**0.5
            mean = 1 / torch.sqrt(alpha) * (x_t - coef * eps_theta)
            var = self.betas[t]**0.5
            x_t = mean + var*noise
        return x_t
    
    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)
    
    
class deNoiseBIG(nn.Module):
    def __init__(self):
        super(deNoiseBIG, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 7, padding=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
    def forward(self, x, t):
        return self.network(x)


class deNoise(nn.Module):
    def __init__(self):
        super(deNoise, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        
    def forward(self, x, t):
        return self.network(x)
    
if __name__=="__main__":
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    batch_size = 128
    num_epochs = 10

    torch.manual_seed(1) # args.seed
    if mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if cuda else "cpu") # args.cuda
    print("Using device ", device)
    kwargs = {'num_workers': 5, 'pin_memory': True} if cuda else {} # args.cuda

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
    train_loss = 0
    
    for epoch in range(1, num_epochs):
        tqdm_ = tqdm(train_loader)
        for data, _ in tqdm_:
            diff.train() # so that everything has gradients and we can do backprop and so on...
            data = data.to(device)
            optimizer.zero_grad() # "reset" gradients to 0 for text iteration
            
            loss = diff(data)
            loss.backward() # calc gradients
            train_loss += loss.item()
            optimizer.step() # backpropagation
            tqdm_.set_description(f"loss: {loss:.4f}")
        
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        
        diff.eval()
        with torch.no_grad():
            sample = diff.sample_p(n=64)
            save_image(sample.view(64, 1, 28, 28).cpu(),
                        'results/diffusion/sample_' + str(epoch) + '.png')
        torch.save(diff.state_dict(), "weights/diff.pth")