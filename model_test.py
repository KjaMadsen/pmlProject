import torch
import numpy as np
import torch.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from Models import *

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



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD # -ELBO


def KL_loss(p, q):
    return torch.distributions.kl.kl_divergence(p, q)

def train(epoch, model, optimizer, loss_function):
    model.train() # so that everything has gradients and we can do backprop and so on...
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad() # "reset" gradients to 0 for text iteration
        recon_batch, mu, logvar = model(data)
        loss = model.KL_loss()
        loss.backward() # calc gradients
        train_loss += loss.item()
        optimizer.step() # backpropagation

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad(): # no_grad turns of gradients...
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += model.KL_loss().item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__=="__main__":
    prior = torch.distributions.Normal(torch.tensor([0]), torch.tensor([.1]))
    model = BayesianVAE(prior).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"trainable parameters: {params:,}")

    for epoch in range(1, num_epochs + 1):
        train(epoch, model, optimizer, KL_loss)
        #torch.save(model.state_dict(), "weights/VAE.pth")
        test(epoch, model, KL_loss)
        with torch.no_grad():
            sample = torch.randn(64, 2).to(device) # 20 -> 2
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                    'figures/BayesVAE/sample_' + str(epoch) + '.png')