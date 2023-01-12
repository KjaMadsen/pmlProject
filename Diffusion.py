#https://nn.labml.ai/diffusion/ddpm/index.html
#https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=Rj17psVw7Shg
# https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=3s 
import torch
import torch.nn as nn
import torch.functional as F


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