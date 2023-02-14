import torch

x = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.tensor([2.2]), torch.tensor([0.4, 0.2]))
print(x.rsample((4,)).shape)