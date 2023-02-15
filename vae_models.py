import torch
import torch.nn.functional as F
import torch.nn as nn


class VanillaVAE(nn.Module):
    def __init__(self, initial_feature_dim=102, hidden_dim=64, num_features=16, num_cols=None):
        super(VanillaVAE, self).__init__()
        self.num_features = num_features
        self.enc1 = nn.Linear(initial_feature_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, num_features * 2)

        self.dec1 = nn.Linear(num_features, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, initial_feature_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        return sample

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.num_features)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        x = F.relu(self.dec1(z))
        recon = torch.sigmoid(self.dec2(x))
        return recon

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, x, num_samples, device):
        _, mu, logvar = self.encode(x)
        z = torch.randn(num_samples, self.num_features) * torch.exp(0.5 * logvar) + mu
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def vae_loss(self, bce_loss, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce_loss + kl


class RelaxedBernoulliVAE(nn.Module):
    def __init__(self, initial_feature_dim=102, hidden_dim=64, num_features=16, num_cols=None):
        super(RelaxedBernoulliVAE, self).__init__()
        self.num_features = num_features
        self.enc1 = nn.Linear(initial_feature_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, num_features)

        self.dec1 = nn.Linear(num_features, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, initial_feature_dim)

    def reparameterize(self, mu_logit, **kwargs):
        eps = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(kwargs['tau'], logits=mu_logit)
        mu = eps.probs
        z = eps.rsample()
        if kwargs['hard']:
            z_ = 0.5 * (torch.sign(z) + 1)
            z = z + (z_ - z).detach()
        return z, mu

    def encode(self, x, **kwargs):
        x = F.relu(self.enc1(x))
        mu_logit = self.enc2(x)
        z, mu = self.reparameterize(mu_logit, **kwargs)
        return z, mu, mu_logit

    def decode(self, z):
        x = F.relu(self.dec1(z))
        recon = torch.sigmoid(self.dec2(x))
        return recon

    
    def forward(self, x, **kwargs):
        z, mu, _ = self.encode(x, **kwargs)
        recon = self.decode(z)
        return recon, mu

    def sample(self, x, num_samples, device, **kwargs):
        _, _, mu_logit = self.encode(x, **kwargs)
        z = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(kwargs['tau'], logits=mu_logit).rsample((num_samples,))
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def vae_loss(self, bce_loss, mu, prior=0.5, eps=1e-10):
        t1 = mu * ((mu + eps)/prior).log()
        t2 = (1-mu) * ((1 - mu + eps) / (1-prior)).log()
        kl = torch.sum(t1 + t2, dim=-1).sum()
        return bce_loss + kl


class MixedGumbelVae(nn.Module):
    def __init__(self, initial_feature_dim=102, hidden_dim=64, num_features=16, num_cols=4):
        super(MixedGumbelVae, self).__init__()
        self.num_features = num_features
        self.num_cols = num_cols
        self.initial_feature_dim = initial_feature_dim
        self.enc1 = nn.Linear(initial_feature_dim, hidden_dim)
        self.encG2 = nn.Linear(hidden_dim, num_features * (initial_feature_dim - num_cols))
        self.encN2 = nn.Linear(hidden_dim, num_features * 2)

        self.decG1 = nn.Linear(num_features * (initial_feature_dim-num_cols), hidden_dim)
        self.decG2 = nn.Linear(hidden_dim, (initial_feature_dim-num_cols))
        self.decN1 = nn.Linear(num_features, hidden_dim)
        self.decN2 = nn.Linear(hidden_dim, num_cols)

    def reparameterize(self, mu_N, logvar_N, x_G, device, **kwargs):
        # Gaussian
        std_N = torch.exp(0.5 * logvar_N)
        eps_N = torch.randn_like(std_N)
        z_N = mu_N + eps_N * std_N
        z_N = z_N.to(device)
        # Gumbel
        logits = x_G.view(-1, self.num_features, (self.initial_feature_dim - self.num_cols))
        z_G = self.gumbel_softmax(logits, kwargs['tau'], kwargs['hard']).to(device)
        return z_N, z_G, logits

    def encode(self, x, device, **kwargs):
        # Encode first num_cols items using Gaussian, and rest using Gumbel
        x = F.relu(self.enc1(x))
        x_G = self.encG2(x)
        x_N = self.encN2(x).view(-1, 2, self.num_features)
        mu_N, logvar_N = x_N[:, 0, :], x_N[:, 1, :]
        z_N, z_G, logits = self.reparameterize(mu_N, logvar_N, x_G, device, **kwargs)
        return z_N, mu_N, logvar_N, z_G, logits

    def decode(self, z_N, z_G):
        x_N = F.relu(self.decN1(z_N))
        x_N = torch.sigmoid(self.decN2(x_N))
        x_G = F.relu(self.decG1(z_G))
        x_G = torch.sigmoid(self.decG2(x_G))
        recon = torch.cat((x_N, x_G), dim=1)
        return recon

    def forward(self, x, device, **kwargs):
        z_N, mu_N, logvar_N, z_G, logits = self.encode(x, device, **kwargs)
        recon = self.decode(z_N, z_G)
        logits_ = F.softmax(logits, dim=-1).reshape(*z_G.size())
        return recon, mu_N, logvar_N, logits_

    def vae_loss(self, bce_loss, mu_N, logvar_N, logits_, eps=1e-20):
        kl_N = -0.5 * torch.sum(1 + logvar_N - mu_N.pow(2) - logvar_N.exp())
        log_ratio = torch.log(logits_ * (self.initial_feature_dim - self.num_cols) + eps)
        kl_G = torch.sum(logits_ * log_ratio, dim=-1).mean()
        return bce_loss + kl_N + kl_G



    def sample_gumbel_dist(self, shape, device, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)


    def gumbel_softmax(self, logits, tau, device, hard=False):
        y = logits.to(device) + self.sample_gumbel_dist(logits.size(), device)
        y = F.softmax(y / tau, dim=-1)
        if not hard:
            return y.view(-1, self.num_features*(self.initial_feature_dim - self.num_cols))
        
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = y + (y_hard - y).detach()
        return y_hard.view(-1, self.num_features * (self.initial_feature_dim - self.num_cols))

    def sample(self, num_samples, device, **kwargs):
        pass 

    


