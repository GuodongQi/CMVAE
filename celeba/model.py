import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace

from module import SAB, CausalEncoder


class GMVAE(nn.Module):
    def __init__(
            self,
            input_shape=512,
            unsupervised_em_iters=5,
            semisupervised_em_iters=5,
            fix_pi=False,
            component_size=20,
            latent_size=64,
            train_mc_sample_size=10,
            test_mc_sample_size=10,
            linear=False,
            laplace=False,
            gamma=1
    ):
        super(GMVAE, self).__init__()

        self.gamma1 = 1 / math.sqrt(gamma)
        self.laplace = laplace
        self.linear = linear
        self.input_shape = input_shape
        self.unsupervised_em_iters = unsupervised_em_iters
        self.semisupervised_em_iters = semisupervised_em_iters
        self.fix_pi = fix_pi
        self.component_size = component_size
        self.latent_size = latent_size
        self.train_mc_sample_size = train_mc_sample_size
        self.test_mc_sample_size = test_mc_sample_size
        self.I = torch.eye(latent_size).cuda()

        self.q_z_given_H = nn.Sequential(
            SAB(
                dim_in=self.input_shape,
                dim_out=self.input_shape,
                num_heads=4,
                ln=False
            ),
            SAB(
                dim_in=self.input_shape,
                dim_out=self.input_shape,
                num_heads=4,
                ln=False
            ),
            nn.Linear(self.input_shape, 2 * self.latent_size)
        )

        self.causal = CausalEncoder(self.latent_size, linear=self.linear)

        self.proj = nn.Sequential(
            nn.Linear(2 * latent_size, 2 * self.input_shape),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.input_shape, 2 * self.input_shape),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.input_shape, self.input_shape),
            nn.ReLU(inplace=True),
        )

        self.register_buffer('log_norm_constant', torch.tensor(-0.5 * np.log(2 * np.pi)))
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

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

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

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

    def forward(self, H, h_lambda, h_c, a_lambda):
        batch_size, sample_size = H.shape[:2]

        # q_z
        # batch_size, sample_size, latent_size
        # batch_size, sample_size, latent_size
        # batch_size, sample_size, mc_sample_size, latent_size
        # q_z_mean: batch_size, sample_size, hidden_size
        q_z_mean, q_z_logvar = self.q_z_given_H(H).split(self.latent_size, dim=-1)
        # q_z: batch_size, sample_size, train_mc_sample_size, latent_size
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size, laplace=False)
        # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
        q_z = q_z.view(batch_size, -1, self.latent_size)

        q_eps = self.causal(q_z)

        A = self.causal.get_w()
        # q_z = q_z - q_eps

        ## decode ##
        # batch_size*sample_size*mc_sample_size, latent_size
        H_rec = self.proj(torch.cat([q_z, q_eps], -1).view(-1, 2 * self.latent_size))
        # batch_size*sample_size*mc_sample_size, input_shape
        H_rec = H_rec.view(batch_size, sample_size, self.train_mc_sample_size, self.input_shape)

        ## rec loss ##
        H = H[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        rec_loss = F.mse_loss(H_rec, H, reduction='sum') / (batch_size * sample_size * self.train_mc_sample_size)

        ## kl loss ##
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
            q_eps_mu_k,
            # batch_size, component_size, latent_size
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
        # h_augment = h_c * h * h + a_lambda * A.abs().sum()
        # print(rec_loss, log_q_z.mean(), log_p_z.mean(), log_p_eps.mean(), h_augment)
        return H_rec, rec_loss, kl_loss, h_augment, h

    def prediction(self, H_tr, y_tr, H_te):
        batch_size, tr_sample_size = H_tr.shape[:2]
        te_sample_size = H_te.shape[1]

        # batch_size, tr_sample_size+te_sample_size, 256
        H = torch.cat([H_tr, H_te], dim=1)

        with torch.no_grad():
            # q_z
            # q_z_mean: batch_size, sample_size, hidden_size
            q_z_mean, q_z_logvar = self.q_z_given_H(H).split(self.latent_size, dim=-1)
            # q_z: batch_size, sample_size, train_mc_sample_size, latent_size
            q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size, laplace=False)
            # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
            # q_z = q_z - q_z.view(batch_size, -1, self.latent_size).mean(1, keepdim=True).unsqueeze(1)
            # q_z = self.causal.mask(q_z.view(batch_size, -1, self.latent_size)).view(q_z.shape)
            # self.causal.mean = q_z.view(batch_size, -1, self.latent_size).mean(1, keepdim=True)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(tr_sample_size + te_sample_size), :, :].view(batch_size,
                                                                                                 te_sample_size * self.test_mc_sample_size,
                                                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(batch_size * tr_sample_size)[:, None].repeat(1, self.test_mc_sample_size).view(batch_size,
                                                                                                         tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

            # p_z
            p_z_given_psi = self.get_semisupervised_prior(
                unsupervised_z=unsupervised_z,
                supervised_z=supervised_z,
                y=y
            )

            # batch_size, component_size
            # batch_size, component_size, latent_size
            # batch_size, component_size, latent_size
            p_y_given_psi_pi, p_z_given_y_psi_mean, p_z_given_y_psi_logvar, trace = p_z_given_psi

            # batch_size, te_sample_size, mc_sample_size, component_size
            log_likelihoods = self.gaussian_log_prob(
                # batch_size, te_sample_size, mc_sample_size, component_size, latent_size
                unsupervised_z.view(
                    batch_size, te_sample_size, self.test_mc_sample_size, self.latent_size
                )[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
                p_z_given_y_psi_mean[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1),
                p_z_given_y_psi_logvar[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1)
            ) + torch.log(
                p_y_given_psi_pi[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1)
            ) + self.gaussian_log_prob(
                # batch_size, component_size, latent_size
                self.causal(p_z_given_y_psi_mean),
                # batch_size, component_size, latent_size
                torch.log(trace)[:, :, None].repeat(1, 1, self.latent_size)
            ).unsqueeze(1).unsqueeze(
                2)  # batch_size, component_size ->  batch_size, te_sample_size, mc_sample_size, component_size

            # batch_size, sample_size, mc_sample_size, component_size
            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
            )
            # batch_size, sample_size
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred
