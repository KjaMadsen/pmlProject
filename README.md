# Probabilistic Machine Learning
This repo was part of an exam in the Probabilistic Machine Learning course at DIKU

# Exercises And Description

## A Density modeling
During one of the exercise sessions, we implemented a basic variational autoencoder (VAE) on MNIST data. In the first part of the project we will consider extensions and alternative models to fit the same data, and compare them to the basic VAE that you did in class. Note that a sketch of a solution to the VAE exercise from class is available in Absalon under Files->Solutions.
### A.1 Implement a convolutional VAE
The first task is to extend the original model to use convolutional layers rather than linear layers. Compare the quality of the model to the original. Are the models significantly different?
### A.2
Briefly describe how the VAE was implemented, and how parameters were estimated.
Describe the performance of the model compared to the original. You can do this visually, based on the quality of generated images, but try to also come up with a way to do this quantitatively.

Implement one or more alternative density models on the same data (e.g. Mixture model, A discrete latent variable model, Probabilistic PCA, Bayesian VAE, Diffusion model)
• Argue for your choice of models and describe essential properties and advantages/disadvantages of these models (e.g. tractability of likelihood)
• For each model, assess whether the model is better or worse than the original VAE (preferrably both by visual inspection and by quantifying the goodness of fit).

## B Function fitting
In this part, you will use Gaussian Processes for function optimization. Use Pyro to implement a fully Bayesian GP and then use it to find a local minimum of the function
f (x) = sin(20x) + 2 cos(14x) − 2 sin(6x) on the interval x ∈ [−1, 1]
### B.1 Fitting a GP with Pyro
Your task is to implement a full Bayesian GP modeling ap- proach where you first use MCMC sampling to obtain the posterior hyperparameters θ of the GP and then use these samples to provide mean and variance estimates for observations at new points. As dataset D, use the five input-label pairs (xi, yi) with xi = −1,−1/2,0,1/2,1 and y-values yi = f(xi) using the function above. Follow the steps below:
1. Use NUTS to sample from the posterior p(θ|D).
2. Check the quality of the MCMC sampling using diagnostics (hint: use Arviz). Use the diagnostics to choose the hyperparameters of the sampling (such as the number of warmup samples).
3. Use the obtained MCMC samples from the posterior to obtain estimates of mean m(x∗) and variance v(x∗) of p(y∗|x∗, D) at a point x∗ ∈ [−1, 1]
Deliverables:
• A scatter plot on log-log-scale of N = 500 samples from p(θ|D).
• An analysis of your obtained sample quality.
• A plot visualizing p(f∗|x∗,D) for x∗ ∈ [−1,1]. For this, overlay a plot f(x), a scatter plot of D, and plots of m(x∗) as well as m(x∗)±2pv(x∗). Add proper labels and axis descriptions.
For your deliverables, use the following GP parameters:
• Kernel: Gaussian RBM kernel with parameters lengthscale σl2 (kernel width)
and variance σs2.
• Priors: use a LogNormal prior:
σl2 ∼ LogNormal(−1, 1) σv2 ∼ LogNormal(0, 2)
• Fixed GP noise variance 10−4.



Data: Initial dataset D. Candidate set X∗ = {x∗1, . . . x∗l }, number of iterations T
for k = 1,...,T do
Sample f∗ ∼ p(f∗|X∗,D);
p = arg mini fi∗;
Add (x∗p,f(x∗p)) to the dataset D

Algorithm 1: Bayesian Optimization 


Build a Bayesian Optimization loop to find a local optimum of f. For this, implement Algorithm 1. In each iteration, sample a function observed on X∗ from the posterior predictive of the GP using the dataset. Then you locate the minimum of the sampled function on X∗ and evaluate f at this position. Finally, add the new position/value pair to D. Repeat until a target number of iterations is reached. For the GP use the same prior parameters as in the previous task and you should also ensure sufficient convergence of the MCMC chain. For X∗ use 200 evenly spaced points in [−1, 1]

Hint 1: To obtain a sample f∗ only a single sample from p(θ|D) is needed.
Hint 2: For numerical stability, f∗ should not be sampled noise-free.
Hint 3: if it appears that Pyro ignores your setting of a kernel parameter, consider cleaning up old variables with pyro.clear param store().
 
