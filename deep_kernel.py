import torch
from tqdm import tqdm
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel



class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(train_x, train_y, device_idx=0):
    device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')

    likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True).cuda()
    model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes).cuda()
    
    model.to(device)
    likelihood.to(device)
    
    training_iterations = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    
    for i in tqdm(range(training_iterations)):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x.to(device))
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y.to(device))
        loss.backward()
        optimizer.step()
    
    return model, likelihood