import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace, StudentT, MultivariateNormal, Chi2
from torch.nn import Parameter
from tqdm import tqdm

from module import OmniglotEncoder, OmniglotDecoder, SAB, CausalEncoder, CausalDecoder


class GMVAE(nn.Module):  # b old
    def __init__(self,
                 input_shape=[1, 28, 28],
                 unsupervised_em_iters=5,
                 semisupervised_em_iters=5,
                 fix_pi=False,
                 hidden_size=64,
                 component_size=20,
                 latent_size=64,
                 train_mc_sample_size=10,
                 test_mc_sample_size=10,
                 linear=False,
                 eval_iter=15,
                 laplace=False,
                 gamma=1):
        super().__init__()

        self.gamma1 = 1 / math.sqrt(gamma)
        self.eval_iter = eval_iter
        self.input_shape = input_shape
        self.unsupervised_em_iters = unsupervised_em_iters
        self.semisupervised_em_iters = semisupervised_em_iters
        self.fix_pi = fix_pi
        self.hidden_size = hidden_size
        self.last_hidden_size = 2 * 2 * hidden_size
        self.component_size = component_size
        self.latent_size = latent_size
        self.train_mc_sample_size = train_mc_sample_size
        self.test_mc_sample_size = test_mc_sample_size
        self.h_old = np.inf
        self.laplace = laplace
        self.linear = linear

        self.I = torch.eye(latent_size).cuda()

        self.encoder = OmniglotEncoder(
            hidden_size=hidden_size
        )
        self.q_z_given_H = nn.Sequential(
            SAB(
                dim_in=self.last_hidden_size,
                dim_out=self.last_hidden_size,
                num_heads=4,
                ln=False
            ),
            SAB(
                dim_in=self.last_hidden_size,
                dim_out=self.last_hidden_size,
                num_heads=4,
                ln=False
            ),
            nn.Linear(self.last_hidden_size, 2 * self.hidden_size)
        )

        self.causal = CausalEncoder(self.hidden_size, linear=self.linear)

        self.proj = nn.Sequential(
            nn.Linear(2 * latent_size, self.last_hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.last_hidden_size, self.last_hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.last_hidden_size, self.last_hidden_size),
            nn.ELU(inplace=True),
        )

        self.decoder = OmniglotDecoder(
            hidden_size=hidden_size
        )

        self.rec_criterion = nn.BCELoss(reduction='sum')

        self.register_buffer('log_norm_constant', torch.tensor(-0.5 * np.log(2 * np.pi)))
        self.register_buffer('log_2', torch.tensor(np.log(2)))
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def semi_gaussian_log_prob(self, x, mean):
        a = (x - mean).pow(2)
        log_p = -0.5 * a
        return log_p.sum(dim=-1)

    def laplace_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            loglambda = torch.zeros_like(mean)
        else:
            loglambda = 0.5 * (logvar - self.log_2)
        a = (x - mean).abs()
        log_p = -a / loglambda.exp() - loglambda - self.log_2
        return log_p.sum(dim=-1)

    def reparametrize(self, mean, logvar, S=1, laplace=False):
        mean = mean.unsqueeze(2).repeat(1, 1, S, 1)
        logvar = logvar.unsqueeze(2).repeat(1, 1, S, 1)
        if laplace:
            lambda_ = (0.5 * logvar).exp() * 0.5 * math.sqrt(2)
            eps = Laplace(0.0, 1.0).sample(mean.shape).to(mean.device)
            return eps.mul(lambda_).add(mean)

        else:
            std = logvar.mul(0.5).exp()
            eps = torch.randn_like(mean)
            return eps.mul(std).add(mean)

    def confouneder(self, X, v=6, std_k=0.1, sigma=0.1):
        # batch_size, sample_size * mc_sample_size, latent_size
        tau = torch.diag_embed(std_k * X.std([1]) * math.sqrt((v - 2) / v))
        s = torch.randn_like(X.unsqueeze(-1).repeat(1, 1, 1, X.shape[-1])) * sigma
        s = s - torch.diag_embed(torch.diagonal(s, dim1=-1, dim2=-2) - 1)
        l = torch.tril(s)
        # sigma = l.bmm(l.transpose(1, 2))
        # t = StudentT(v, loc=0, scale=s).sample(self).to(tau.device)
        student_t = MultivariateNormal(torch.zeros_like(l[..., 0]), scale_tril=l).sample() / torch.sqrt(
            Chi2(v).sample(l.shape[:3]).to(l.device) / v)
        return student_t.bmm(tau) * 0

    def kl_loss(self, mean, logvar, laplace=False):
        if laplace:
            mean = mean.abs()
            logvar -= np.log(2)
            # return (-logvar - 1 + mean / logvar.exp() + (-mean / logvar.exp()).exp()).sum()
            return (-logvar - 1 + mean + (-mean / logvar.exp() + logvar).exp()).sum(-1)
        else:  # gaussian
            return 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum(-1)

    def kl_loss_semi(self, mean=None, logvar=None, laplace=False):
        if logvar is None:
            if laplace:
                mean = mean.abs()
                return ((-mean).exp() + mean - 1).sum(-1)
            else:  # gaussian
                return 0.5 * (mean ** 2).sum(-1)
        if mean is None:
            if laplace:
                loglambda = (2 * logvar - np.log(2))
                return (-loglambda - 1 + loglambda.exp()).sum(-1)
            else:  # gaussian
                return 0.5 * (-logvar - 1 + logvar.exp()).sum(-1)

    def h_a(self, A):
        h = torch.trace(torch.matrix_exp(A * A)) - self.latent_size

        # x = torch.eye(self.latent_size).cuda() + torch.div(A, self.latent_size)
        # expm_A = torch.matrix_power(x, self.latent_size)
        # h = torch.trace(expm_A) - self.latent_size

        return h

    def get_unsupervised_params(self, X, psi):
        batch_size, sample_size = X.shape[0], X.shape[1]
        pi, mean, trace = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1),
        ) + torch.log(
            pi[:, None, :].repeat(1, sample_size, 1)
        ) + self.gaussian_log_prob(
            # batch_size, component_size, latent_size
            self.causal(mean),
            # batch_size, component_size, latent_size
            torch.log(trace)[:, :, None].repeat(1, 1, self.latent_size)
        ).unsqueeze(1)  # batch_size, component_size ->  batch_size, sample_size, component_size

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        N = torch.where(N.eq(0.0), 1e-22 + torch.zeros_like(N), N)

        # batch_size, component_size
        trace = self.gamma1 * torch.ones_like(N)

        # batch_size, component_size, latent_size, latent_size
        std_causal = self.causal(self.I[None, None, :, :] * trace[:, :, None, None])

        inverse = torch.inverse(
            self.I[None, None, :, :] + std_causal.matmul(std_causal.transpose(-1, -2))
        )

        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        # denominator = N[:, :, None].repeat(1, 1, self.latent_size)
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X,
        ) / denominator

        mean = mean.unsqueeze(2).matmul(inverse).squeeze(2)

        return pi, mean, trace

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        # inverse: batch_size, component_size, latent_size, latent_size
        unsupervised_sample_size = unsupervised_X.shape[1]
        batch_size = unsupervised_X.shape[0]
        pi, mean, logvar, trace = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1),
            # logvar[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)  # todo add by me
        ) + torch.log(
            pi[:, None, :].repeat(1, unsupervised_sample_size, 1)
        ) + self.gaussian_log_prob(
            # batch_size, component_size, latent_size
            self.causal(mean),
            # batch_size, component_size, latent_size
            torch.log(trace)[:, :, None].repeat(1, 1, self.latent_size)
        ).unsqueeze(1)  # batch_size, component_size ->  batch_size, unsupervised_sample_size, component_size

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + \
                      unsupervised_N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size
        trace = self.gamma1 * torch.ones_like(supervised_N)
        # batch_size, component_size, latent_size, latent_size
        std = torch.diag_embed((0.5 * logvar).exp()) * trace[:, :, None, None]
        # batch_size, component_size, latent_size, latent_size
        std_causal = self.causal(std.view(batch_size, -1, self.latent_size)).view(std.shape)
        inverse = torch.inverse(
            self.I[None, None, :, :] + std_causal.matmul(std_causal.transpose(-1, -2)))

        # batch_size, component_size, latent_size
        supervised_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_X
        )

        # batch_size, component_size, latent_size
        unsupervised_mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            unsupervised_X
        )

        # supervised_mean = supervised_mean.bmm(inverseI)
        # unsupervised_mean = unsupervised_mean.unsqueeze(2).matmul(inverse_sigma).squeeze(2)

        mean = (supervised_mean + unsupervised_mean) / denominator
        mean = mean.unsqueeze(2).matmul(inverse).squeeze(2)

        supervised_X2 = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_X.pow(2.0)
        )

        supervised_X_mean = mean * torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_X
        )

        # batch_size, component_size, latent_size
        supervised_mean2 = supervised_N[:, :, None].repeat(1, 1, self.latent_size) * mean.pow(2.0)

        supervised_var = supervised_X2 - 2 * supervised_X_mean + supervised_mean2

        unsupervised_X2 = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            unsupervised_X.pow(2.0)
        )

        unsupervised_X_mean = mean * torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            unsupervised_X
        )

        # batch_size, component_size, latent_size
        unsupervised_mean2 = unsupervised_N[:, :, None].repeat(1, 1, self.latent_size) * mean.pow(2.0)

        unsupervised_var = unsupervised_X2 - 2 * unsupervised_X_mean + unsupervised_mean2

        var = (supervised_var + unsupervised_var) / denominator
        logvar = torch.log(var)

        return pi, mean, logvar, trace

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)

        # batch_size, component_size
        initial_trace = torch.ones_like(initial_pi)

        # latent_size, latent_size
        sig_cas = self.causal(self.I)
        # latent_size, latent_size
        sig_cas_2 = sig_cas.mm(sig_cas.T)

        # batch_size, component_size, latent_size, latent_size
        inverse = torch.inverse(
            self.I[None, None, :, :] + 1 / initial_trace[:, :, None, None] * sig_cas_2[None, None, :, :])

        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0).unsqueeze(2).matmul(inverse).squeeze(2)
        psi = (initial_pi, initial_mean, initial_trace)

        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                # inverse=inverse,  # for reduce computation
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size, sample_size = unsupervised_z.shape[0], unsupervised_z.shape[1]
        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        component_size = initial_pi.shape[1]
        # batch_size, component_size
        initial_trace = torch.ones_like(initial_pi)

        # latent_size, latent_size
        sig_cas = self.causal(self.I)
        # latent_size, latent_size
        sig_cas_2 = sig_cas.mm(sig_cas.T)

        # batch_size, component_size, latent_size, latent_size
        inverse = torch.inverse(
            self.I[None, None, :, :] + 1 / initial_trace[:, :, None, None] * sig_cas_2[None, None, :, :])

        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)

        initial_mean = initial_mean.unsqueeze(2).matmul(inverse).squeeze(2)

        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        psi = (initial_pi, initial_mean, initial_logvar, initial_trace)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                # inverse=[inverse, inverse_sig],
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_log_prob(self, q, mean, logvar, laplace=False):
        if not laplace:
            log_q = self.gaussian_log_prob(
                # batch_size, sample_size, mc_sample_size, latent_size
                q,
                mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
                logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
            )  # batch_size, sample_size, mc_sample_size
        else:
            log_q = self.laplace_log_prob(
                # batch_size, sample_size, mc_sample_size, latent_size
                q,
                mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
                logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
            )  # batch_size, sample_size, mc_sample_size
        return log_q

    def forward(self, X, h_lambda, h_c, a_lambda):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_z_mean: batch_size, sample_size, hidden_size
        q_z_mean, q_z_logvar = self.q_z_given_H(H).split(self.latent_size, dim=-1)
        # q_z: batch_size, sample_size, train_mc_sample_size, latent_size
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size, laplace=False)
        # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
        q_z = q_z.view(batch_size, -1, self.latent_size)

        # u = self.confouneder(q_z)
        # q_z: batch_size, sample_size*train_mc_sample_size, latent_size

        q_eps = self.causal(q_z - q_z.mean(1, keepdim=True))

        A = self.causal.get_w()
        # q_z = q_z - q_eps

        # H_rec = self.proj(q_z.view(-1, self.latent_size))
        H_rec = self.proj(torch.cat([q_z, q_eps], -1).view(-1, 2 * self.latent_size))

        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec,
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        # epi_loss = self.kl_loss_semi(q_eps)

        # mu_k： batch_size, component_size, latent_size
        # pi_k： batch_size, component_size
        with torch.no_grad():
            pi_k, mu_k, trace_k = self.get_unsupervised_prior(q_z)
        q_eps_mu_k = self.causal(mu_k)
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z.view(batch_size, sample_size, self.train_mc_sample_size, self.latent_size)[:, :, :, None, :].repeat(
                1, 1, 1, self.component_size, 1),
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
        ) + torch.log(
            pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1)
        ) + self.gaussian_log_prob(
            # batch_size, component_size, latent_size
            # self.causal(mu_k).detach(),
            # q_eps_mu_k.detach(),
            q_eps_mu_k,
            # batch_size, component_size, latent_size
            torch.log(trace_k)[:, :, None].repeat(1, 1, self.latent_size)
        ).unsqueeze(1).unsqueeze(
            2)  # batch_size, component_size ->  batch_size,  sample_size, mc_sample_size, component_size

        # batch_size, sample_size, mc_sample_size
        log_p_z = torch.logsumexp(log_likelihoods, dim=-1)

        log_q_z = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z.view(batch_size, sample_size, self.train_mc_sample_size, self.latent_size),
            q_z_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )

        # log_q_eps = 0
        # batch_size, sample_size, self.train_mc_sample_size, self.latent_size
        # q_eps = torch.cat([q_eps, q_eps_mu_k], 1)
        log_p_eps = self.gaussian_log_prob(
            q_eps,
            torch.zeros_like(q_eps),
        )

        kl_loss = log_q_z.mean() - log_p_z.mean() - log_p_eps.mean()
        # kl_loss = log_q_z.mean() - log_p_z.mean()
        # print(log_q_z.mean(), log_p_epi.mean())
        # h_augment
        h = self.h_a(A)
        h_augment = h_lambda * h + 0.5 * h_c * h * h + a_lambda * A.abs().sum()
        # print(self.h_lambda, self.A.abs().sum())
        return X_rec, rec_loss, kl_loss, h_augment, h

    def prediction(self, X_tr, y_tr, X_te):
        batch_size, tr_sample_size = X_tr.shape[:2]
        te_sample_size = X_te.shape[1]

        # batch_size, tr_sample_size+te_sample_size, 1, 28, 28
        X = torch.cat([X_tr, X_te], dim=1)

        with torch.no_grad():
            # encode
            H = self.encoder(
                X.view(-1, *self.input_shape)
            ).view(batch_size, tr_sample_size + te_sample_size, self.last_hidden_size)

            # q_z_mean: batch_size, sample_size, hidden_size
            q_z_mean, q_z_logvar = self.q_z_given_H(H).split(self.latent_size, dim=-1)
            # q_z: batch_size, sample_size, train_mc_sample_size, latent_size
            q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size, laplace=False)
            # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
            # q_z = q_z - q_z.view(batch_size, -1, self.latent_size).mean(1, keepdim=True).unsqueeze(1)
            # q_z = self.causal.mask(q_z.view(batch_size, -1, self.latent_size)).view(q_z.shape)

            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(tr_sample_size + te_sample_size), :, :].view(
                batch_size, te_sample_size * self.test_mc_sample_size, self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(
                batch_size, tr_sample_size * self.test_mc_sample_size, self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()
            pi_k, mu_k, sigma_k, trace = self.get_semisupervised_prior(
                unsupervised_z=unsupervised_z,
                supervised_z=supervised_z,
                y=y
            )

            # batch_size, component_size

            # batch_size, te_sample_size, mc_sample_size, component_size
            log_likelihoods = self.gaussian_log_prob(
                # batch_size, te_sample_size, mc_sample_size, component_size, latent_size
                unsupervised_z.view(
                    batch_size, te_sample_size, self.test_mc_sample_size, self.latent_size
                )[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
                mu_k[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1),
                sigma_k[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1)
            ) + torch.log(
                pi_k[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1)
            ) + self.gaussian_log_prob(
                # batch_size, component_size, latent_size
                self.causal(mu_k),
                # batch_size, component_size, latent_size
                torch.log(trace)[:, :, None].repeat(1, 1, self.latent_size)
            ).unsqueeze(1).unsqueeze(
                2)  # batch_size, component_size ->  batch_size, te_sample_size, mc_sample_size, component_size

            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred

    def tr_prediction(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_z
        # q_z_mean: batch_size, sample_size, hidden_size
        q_z_mean, q_z_logvar = self.q_z_given_H(H).split(self.latent_size, dim=-1)
        # q_z: batch_size, sample_size, train_mc_sample_size, latent_size
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size, laplace=False)
        q_z = q_z.view(batch_size, -1, self.latent_size)
        with torch.no_grad():
            pi_k, mu_k, trace_k = self.get_unsupervised_prior(q_z)
        q_eps_mu_k = self.causal(mu_k)
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z.view(batch_size, sample_size, self.train_mc_sample_size, self.latent_size)[:, :, :, None, :].repeat(
                1, 1, 1, self.component_size, 1),
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
        ) + torch.log(
            pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1)
        ) + self.gaussian_log_prob(
            # batch_size, component_size, latent_size
            # self.causal(mu_k).detach(),
            # q_eps_mu_k.detach(),
            q_eps_mu_k,
            # batch_size, component_size, latent_size
            torch.log(trace_k)[:, :, None].repeat(1, 1, self.latent_size)
        ).unsqueeze(1).unsqueeze(
            2)  # batch_size, component_size ->  batch_size,  sample_size, mc_sample_size, component_size

        # batch_size, sample_size, mc_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )
        # batch_size, sample_size
        y_pred = posteriors.mean(dim=-2).argmax(dim=-1)

        return y_pred

    def te_prediction(self, X_tr, y_tr, X_te):
        return self.prediction(X_tr, y_tr, X_te)

    def generate(self, X_tr, y_tr, X_te, seed=None):
        batch_size, tr_sample_size = X_tr.shape[:2]
        te_sample_size = X_te.shape[1]

        # batch_size, tr_sample_size+te_sample_size, 1, 28, 28
        X = torch.cat([X_tr, X_te], dim=1)

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, tr_sample_size + te_sample_size, self.last_hidden_size)
        # q_z_mean: batch_size, sample_size, hidden_size
        q_z_mean, q_z_logvar = self.q_z_given_H(H).split(self.latent_size, dim=-1)
        # q_z: batch_size, sample_size, train_mc_sample_size, latent_size
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size, laplace=False)
        # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
        # q_z = q_z - q_z.view(batch_size, -1, self.latent_size).mean(1, keepdim=True).unsqueeze(1)
        # q_z = self.causal.mask(q_z.view(batch_size, -1, self.latent_size)).view(q_z.shape)

        # batch_size, te_sample_size*mc_sample_size, latent_size
        unsupervised_z = q_z[:, tr_sample_size:(tr_sample_size + te_sample_size), :, :].view(
            batch_size, te_sample_size * self.test_mc_sample_size, self.latent_size)

        # batch_size, tr_sample_size*mc_sample_size, latent_size
        supervised_z = q_z[:, :tr_sample_size, :, :].view(
            batch_size, tr_sample_size * self.test_mc_sample_size, self.latent_size)

        # batch_size, tr_sample_size*mc_sample_size
        y = y_tr.view(
            batch_size * tr_sample_size
        )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
        # batch_size, tr_sample_size*mc_sample_size, component_size
        y = F.one_hot(y, self.component_size).float()
        pi_k, mu_k, sigma_k, trace = self.get_semisupervised_prior(
            unsupervised_z=unsupervised_z,
            supervised_z=supervised_z,
            y=y
        )
        idx = 0

        # sample_size, last_hidden_size
        z_to = unsupervised_z.view(batch_size, te_sample_size, self.test_mc_sample_size, self.latent_size)[idx, :, idx,
               :]

        H_rec = self.proj(
            torch.cat([z_to, self.causal(z_to)], -1)
        )
        Recon_X = self.decoder(
            H_rec
        )

        # component_size, latent_size
        eps = self.causal(mu_k[idx, :, :])
        H_component = self.proj(torch.cat([mu_k[idx, :, :], eps], -1))
        Component_X = self.decoder(
            H_component
        )

        True_X = X_te[idx]

        # counterfactual
        A = self.causal.get_w()
        indices = torch.nonzero(self.causal.get_w() > 1.2e-3)

        # causes = [1, 59, 44, 33, 34, 11, 39]
        causes = [0, 58]
        if seed is not None:
            np.random.seed(15)

        def removelist(x, y):
            return [item for item in x if item not in y]

        # causes = np.random.choice(removelist(range(self.latent_size), causes), size=len(causes), replace=False)
        print(causes)

        z_to_intervention = z_to

        for i in range(0, z_to.shape[0]):
            for j in range(self.latent_size):
                if j in causes:
                    z_to_intervention[i][j] = 0

        z_ = z_to_intervention - self.causal(z_to_intervention)
        H_rec_2 = self.proj(
            torch.cat([z_, self.causal(z_)], -1)
        )
        Recon_X_2 = self.decoder(
            H_rec_2
        )
        return True_X, Recon_X, Component_X, Recon_X_2
