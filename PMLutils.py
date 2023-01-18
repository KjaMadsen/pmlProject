import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import ExplainedVariance, MeanAbsoluteError

def test_suite(real, fake):
    if real.shape != fake.shape:
        fake = fake.reshape(-1,1,28,28)
    assert real.shape == fake.shape
    msi = MSSSIM(kernel_size=(3,3), betas=(0.0448, 0.03))
    m = msi(fake, real).item()
    psnr = PSNR()
    p = psnr(fake, real).item()
    explained_variance = ExplainedVariance() 
    c = explained_variance(fake, real).item()
    mae = MeanAbsoluteError()
    a = mae(fake, real).item()
    return {"MSSSIM": m, "PSNR": p, "ExpV": c, "MAE": a}

# Reconstruction + KL divergence losses summed over all elements and batch
def ELBO_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.reshape(*recon_x.size()), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD # -ELBO


def train_VAE(model, optimizer, dataloader, device):
    model.train() # so that everything has gradients and we can do backprop and so on...
    tqdm_ = tqdm(dataloader)
    for data,_ in tqdm_:
        data = data.to(device)
        optimizer.zero_grad() # "reset" gradients to 0 for text iteration
        recon_batch, mu, logvar = model(data)
        loss = ELBO_loss_function(recon_batch, data, mu, logvar)
        loss.backward() # calc gradients
        optimizer.step() # backpropagation
        tqdm_.set_description(f"loss: {loss:.4f}")
    


def test_VAE(model, device, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad(): # no_grad turns of gradients...
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += ELBO_loss_function(recon_batch, data, mu, logvar).item()
    

    test_loss /= len(dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss

def save_imgs(model, num_imgs, z, location, weights = None, device='cpu'):
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    if not os.path.exists(location):
        os.makedirs(location)
    with torch.no_grad():
        for i in range(num_imgs):
            sample = torch.rand(1, 2)
            sample = model.decode(sample).cpu().numpy()
            np.save(location+f"/sample_{i}", sample.reshape((28,28)))

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = np.load(directory+"/"+filename)
        images.append(torch.from_numpy(np.array(img)))
    return torch.stack(images).unsqueeze(1)

