class GMVAE_(nn.Module):
    def __init__(
            self,
            input_shape=[1, 28, 28],
            unsupervised_em_iters=5,
            semisupervised_em_iters=5,
            fix_pi=False,
            hidden_size=64,
            component_size=20,
            latent_size=64,
            train_mc_sample_size=10,
            test_mc_sample_size=10
    ):
        super(GMVAE, self).__init__()

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

        self.encoder = OmniglotEncoder(
            hidden_size=hidden_size
        )

        self.q_z_given_x_net = nn.Sequential(
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

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
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
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

    def reparametrize(self, mean, logvar, S=1):
        mean = mean.unsqueeze(1).repeat(1, S, 1)
        logvar = logvar.unsqueeze(1).repeat(1, S, 1)
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

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1,
                                                                                                                  self.latent_size)

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
        mean = (supervised_mean + unsupervised_mean) / denominator

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

        return pi, mean, logvar

    def get_posterior(self, H, mc_sample_size=10):
        batch_size, sample_size = H.shape[:2]

        ## q z ##
        # batch_size, sample_size, latent_size
        q_z_given_x_mean, q_z_given_x_logvar = self.q_z_given_x_net(H).split(self.latent_size, dim=-1)
        # batch_size, sample_size, mc_sample_size, latent_size
        q_z_given_x = self.reparametrize(
            mean=q_z_given_x_mean.view(batch_size * sample_size, self.latent_size),
            logvar=q_z_given_x_logvar.view(batch_size * sample_size, self.latent_size),
            S=mc_sample_size
        ).view(batch_size, sample_size, mc_sample_size, self.latent_size)

        return q_z_given_x_mean, q_z_given_x_logvar, q_z_given_x

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0)

        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_z
        # batch_size, sample_size, latent_size
        # batch_size, sample_size, latent_size
        # batch_size, sample_size, mc_sample_size, latent_size
        q_z_given_x_mean, q_z_given_x_logvar, q_z_given_x = self.get_posterior(H, self.train_mc_sample_size)

        # p_z
        all_z = q_z_given_x.view(batch_size, -1, self.latent_size)
        p_z_given_psi = self.get_unsupervised_prior(z=all_z)
        # batch_size, component_size
        # batch_size, component_size, latent_size
        p_y_given_psi_pi, p_z_given_y_psi_mean = p_z_given_psi

        ## decode ##
        # batch_size*sample_size*mc_sample_size, latent_size
        H_rec = self.proj(q_z_given_x.view(-1, self.latent_size))
        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        ## kl loss ##
        # batch_size, sample_size, mc_sample_size
        log_qz = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z_given_x,
            q_z_given_x_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_given_x_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )

        # batch_size, sample_size, mc_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z_given_x[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            p_z_given_y_psi_mean[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(p_y_given_psi_pi[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))
        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        # kl_loss = torch.mean(log_qz.mean(dim=-1) - log_pz.mean(dim=-1))
        kl_loss = torch.mean(log_qz - log_pz)

        return X_rec, rec_loss, kl_loss, 0, 0

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

            # q_z
            # batch_size, tr_sample_size+te_sample_size, mc_sample_size, latent_size
            _, _, q_z_given_x = self.get_posterior(H, self.test_mc_sample_size)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z_given_x[:, tr_sample_size:(tr_sample_size + te_sample_size), :, :].view(batch_size,
                                                                                                         te_sample_size * self.test_mc_sample_size,
                                                                                                         self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z_given_x[:, :tr_sample_size, :, :].view(batch_size,
                                                                      tr_sample_size * self.test_mc_sample_size,
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
            p_y_given_psi_pi, p_z_given_y_psi_mean, p_z_given_y_psi_logvar = p_z_given_psi

            # batch_size, te_sample_size, mc_sample_size, component_size
            log_likelihoods = self.gaussian_log_prob(
                # batch_size, te_sample_size, mc_sample_size, component_size, latent_size
                unsupervised_z.view(batch_size, te_sample_size, self.test_mc_sample_size, self.latent_size)[:, :, :,
                None,
                :].repeat(1, 1, 1, self.component_size, 1),
                p_z_given_y_psi_mean[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1),
                p_z_given_y_psi_logvar[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1)
            ) + torch.log(p_y_given_psi_pi[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

            # batch_size, sample_size, mc_sample_size, component_size
            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
            )
            # batch_size, sample_size
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred












class GMVAE_old(nn.Module):  # b old
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
                 h_lambda=0.1,
                 h_c=0.1,
                 eval_iter=15,
                 reg_sigma_k=5e-6):
        super().__init__()

        self.eval_iter = eval_iter
        self.h_c = h_c
        self.h_lambda = h_lambda
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
        self.reg_sigma_k = reg_sigma_k

        self.encoder = OmniglotEncoder(
            hidden_size=hidden_size
        )

        self.q_eps_given_x_net = nn.Sequential(
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

        self.q_z_given_epi = CausalEncoder(self.latent_size)

        # self.deep_gmm = nn.Sequential(
        #     nn.Linear(self.hidden_size, 2 * self.hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2 * self.hidden_size, self.hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.hidden_size, self.component_size),
        # )
        self.deep_gmm = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.hidden_size, self.component_size)
        )

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
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

        self.A = Parameter(torch.zeros([self.latent_size, self.latent_size]).cuda(), requires_grad=True)
        self.register_buffer('log_norm_constant', torch.tensor(-0.5 * np.log(2 * np.pi)))
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def reparametrize(self, mean, logvar, S=1, laplace=False):
        mean = mean.unsqueeze(2).repeat(1, 1, S, 1)
        logvar = logvar.unsqueeze(2).repeat(1, 1, S, 1)
        if laplace:
            std = logvar.exp()
            eps = Laplace(0.0, 1.0).sample(mean.shape).to(mean.device)
        else:
            std = logvar.mul(0.5).exp()
            eps = torch.randn_like(mean)
        return eps.mul(std).add(mean)

    def confouneder(self, X, v=6, std_k=0.5, sigma=0.5):
        # batch_size, sample_size * mc_sample_size, latent_size
        tau = torch.diag_embed(std_k * X.std([1]) * math.sqrt((v - 2) / v))
        s = torch.randn_like(X.unsqueeze(-1).repeat(1, 1, 1, X.shape[-1])) * sigma
        s = s - torch.diag_embed(torch.diagonal(s, dim1=-1, dim2=-2) - 1)
        l = torch.tril(s)
        # sigma = l.bmm(l.transpose(1, 2))
        # t = StudentT(v, loc=0, scale=s).sample(self).to(tau.device)
        student_t = MultivariateNormal(torch.zeros_like(l[..., 0]), scale_tril=l).sample() / torch.sqrt(
            Chi2(v).sample(l.shape[:3]).to(l.device) / v)
        return student_t.bmm(tau)

    def kl_loss(self, mean, logvar, laplace=False):
        if laplace:
            mean = mean.abs()
            # return (-logvar - 1 + mean / logvar.exp() + (-mean / logvar.exp()).exp()).sum()
            return (-logvar - 1 + mean + (-mean / logvar.exp() + logvar).exp()).sum(-1)
        else:  # gaussian
            return 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum(-1)

    def h_augment(self):
        h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.latent_size
        h_augment = self.h_lambda * h + 0.5 * self.h_c * h * h
        # adjust hyper params
        self.h_lambda += self.h_c * h.item()
        if h.item() > self.h_old * 0.25 and self.h_c < 1e12:
            self.h_c *= 10
        self.h_old = h.item()
        return h_augment

    def get_posterior(self, q_z):
        # q_z: batch_size, N, latent_size
        pi_ik = F.softmax(self.deep_gmm(q_z), dim=-1)  # batch_size, N, k
        pi_k = pi_ik.sum(1) / q_z.shape[1]  # batch_size, k
        mu_k = pi_ik.transpose(1, 2).bmm(q_z) / (pi_ik.sum(1).unsqueeze(-1))  # batch_size, k, latent_size
        sigma_k = (pi_ik.unsqueeze(-1) * (q_z.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2)).sum(1) / (
            pi_ik.sum(1).unsqueeze(-1))  # batch_size, k, latent_size
        return pi_k, mu_k, sigma_k

    def get_semisupervised_prior(self, deep_gmm, unsupervised_z, supervised_z, y):
        # *_z: batch_size, N, latent_size
        # y: batch_size, N, k

        pi_ik = F.softmax(deep_gmm(unsupervised_z), dim=-1)  # batch_size, N, k
        pi_k = pi_ik.sum(1) / unsupervised_z.shape[1]  # batch_size, k
        mu_k = (pi_ik.transpose(1, 2).bmm(unsupervised_z) + y.transpose(1, 2).bmm(supervised_z)) / (
                pi_ik.sum(1).unsqueeze(-1) + y.sum(1).unsqueeze(-1))  # batch_size, k, latent_size
        sigma_k = ((pi_ik.unsqueeze(-1) * (unsupervised_z.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2)).sum(1) +
                   (y.unsqueeze(-1) * (supervised_z.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2)).sum(1)) / (
                          pi_ik.sum(1).unsqueeze(-1) + y.sum(1).unsqueeze(-1))  # batch_size, k, latent_size
        return pi_k, mu_k, sigma_k

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        H_m = H.mean(-1, keepdim=True)

        # batch_size, sample_size, latent_size
        # batch_size, sample_size, latent_size
        # batch_size, sample_size, mc_sample_size, latent_size

        # q_eps: batch_size, latent_size
        q_eps_mean, q_eps_logvar = self.q_eps_given_x_net(H - H_m).split(self.latent_size, dim=-1)
        q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, True).squeeze(2)

        # latent confouneder
        f = self.confouneder(q_eps)

        # q_z: batch_size, sample_size, mc_sample_size, latent_size
        q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps + f, self.A).split(self.latent_size, dim=-1)
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

        # p_z, deep_gmm, batch_size, mc_sample_size, k
        pi_k, mu_k, sigma_k = self.get_posterior(q_z.view(batch_size, -1, self.latent_size))

        H_rec = self.proj(q_z.view(-1, self.latent_size))
        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec + H_m
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        log_qz = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z,
            q_z_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )  # batch_size, sample_size, mc_sample_size

        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
            sigma_k.log()[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))
        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        # q_z_ = q_z.view(batch_size, -1, self.latent_size)
        # log_pz = ((-0.5 * (q_z_.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2) / sigma_k.unsqueeze(1)).sum(-1).exp() / (
        #         2 * np.pi * sigma_k.prod(-1).unsqueeze(1)).sqrt() * pi_k.unsqueeze(1)).add(1e-14).sum(-1).log()

        kl_loss = log_qz.mean() - log_pz.mean() + 0 * self.kl_loss(q_eps_mean, q_eps_logvar, True).mean()
        # print(log_qz.mean(), log_pz.mean(), sigma_k.exp().min())

        # h_augment
        h_augment = self.h_augment()
        reg_sigma_k = (1 / sigma_k).sum() * self.reg_sigma_k
        return X_rec, rec_loss, kl_loss, h_augment, reg_sigma_k

    def prediction(self, X_tr, y_tr, X_te):
        batch_size, tr_sample_size = X_tr.shape[:2]
        te_sample_size = X_te.shape[1]

        # batch_size, tr_sample_size+te_sample_size, 1, 28, 28
        X = torch.cat([X_tr, X_te], dim=1)
        deep_gmm = copy.deepcopy(self.deep_gmm)
        deep_gmm.train()
        deep_gmm.requires_grad_()
        # print(list(deep_gmm.parameters()))
        adam = torch.optim.Adam(deep_gmm.parameters(), lr=1e-4)

        with torch.no_grad():
            # encode
            H = self.encoder(
                X.view(-1, *self.input_shape)
            ).view(batch_size, tr_sample_size + te_sample_size, self.last_hidden_size)

            H = H - H.mean(-1, keepdim=True)
            # q_eps: batch_size, latent_size
            q_eps_mean, q_eps_logvar = self.q_eps_given_x_net(H).split(self.latent_size, dim=-1)
            q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, True).squeeze(2)

            # latent confouneder
            f = self.confouneder(q_eps)

            # q_z: batch_size, sample_size, mc_sample_size, latent_size
            q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps + f, self.A).split(self.latent_size, dim=-1)
            q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(
                    tr_sample_size + te_sample_size), :, :].view(batch_size, te_sample_size * self.test_mc_sample_size,
                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

        for eval_iter in range(self.eval_iter):
            # p_z
            pi_k, mu_k, sigma_k = self.get_semisupervised_prior(
                deep_gmm=deep_gmm,
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
                sigma_k.log()[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1)
            ) + torch.log(pi_k[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))
            log_pz = torch.logsumexp(log_likelihoods, dim=-1)
            # batch_size, sample_size, mc_sample_size, component_size
            loss = -log_pz.mean()
            adam.zero_grad()
            loss.backward()
            adam.step()

        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
        y_te_pred = -posteriors.mean(dim=-2).argmax(dim=-1)

        return y_te_pred

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
                 h_lambda=0.1,
                 h_c=0.1,
                 eval_iter=15,
                 laplace=False):
        super().__init__()

        self.eval_iter = eval_iter
        self.h_c = h_c
        self.h_lambda = h_lambda
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
        self.A = Parameter(torch.zeros([self.latent_size, self.latent_size]).cuda(), requires_grad=True)

        self.encoder = OmniglotEncoder(
            hidden_size=hidden_size
        )
        self.q_eps_given_x = nn.Sequential(
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

        self.q_z_given_epi = CausalDecoder(self.latent_size)

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
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
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def reparametrize(self, mean, logvar, S=1, laplace=False):
        mean = mean.unsqueeze(2).repeat(1, 1, S, 1)
        logvar = logvar.unsqueeze(2).repeat(1, 1, S, 1)
        if laplace:
            std = logvar.exp()
            eps = Laplace(0.0, 1.0).sample(mean.shape).to(mean.device)
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
        return student_t.bmm(tau)

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1,
                                                                                                                  self.latent_size)

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
        mean = (supervised_mean + unsupervised_mean) / denominator

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

        return pi, mean, logvar

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0)

        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def kl_loss(self, mean, logvar, laplace=False):
        if laplace:
            mean = mean.abs()
            logvar -= np.log(2)
            # return (-logvar - 1 + mean / logvar.exp() + (-mean / logvar.exp()).exp()).sum()
            return (-logvar - 1 + mean + (-mean / logvar.exp() + logvar).exp()).sum(-1)
        else:  # gaussian
            return 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum(-1)

    def h_augment(self):
        h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.latent_size
        h_augment = self.h_lambda * h + 0.5 * self.h_c * h * h
        # adjust hyper params
        self.h_lambda += self.h_c * h.item()
        if h.item() > self.h_old * 0.25 and self.h_c < 1e12:
            self.h_c *= 2
        self.h_old = h.item()
        # print(self.A.sum())
        return h_augment

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_eps: batch_size, sample_size,latent_size
        q_eps_mean, q_eps_logvar = self.q_eps_given_x(H).split(self.latent_size, dim=-1)
        # q_eps_logvar = torch.zeros_like(q_eps_mean)

        # latent confouneder
        u = self.confouneder(q_eps_logvar)

        # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
        # q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps, self.A, u).split(self.latent_size, dim=-1)
        q_z_mean = self.q_z_given_epi.cal_mean(q_eps_mean, self.A, u)
        q_z_logvar = self.q_z_given_epi.cal_logvar(q_eps_logvar, self.A, u)

        # q_eps_mean = self.q_eps_given_z(qz, self.A, u)
        # q_z: batch_size, sample_size, mc_sample_size, latent_size
        # q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps_mean, self.A, u).split(self.latent_size, dim=-1)
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

        # p_z, deep_gmm, batch_size, mc_sample_size, k
        pi_k, mu_k = self.get_unsupervised_prior(q_z.view(batch_size, -1, self.latent_size))

        H_rec = self.proj(q_z.view(-1, self.latent_size))

        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        log_qz = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z,
            q_z_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )  # batch_size, sample_size, mc_sample_size

        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
            # sigma_k.log()[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))

        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        # q_z_ = q_z.view(batch_size, -1, self.latent_size)
        # log_pz = ((-0.5 * (q_z_.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2) / sigma_k.unsqueeze(1)).sum(-1).exp() / (
        #         2 * np.pi * sigma_k.prod(-1).unsqueeze(1)).sqrt() * pi_k.unsqueeze(1)).add(1e-14).sum(-1).log()

        kl_loss = log_qz.mean() - log_pz.mean() + 0. * self.kl_loss(q_eps_mean, q_eps_logvar, self.laplace).mean()
        # kl_loss = log_qz.mean() - log_pz.mean()
        # print(log_qz.mean(), log_pz.mean(), sigma_k.exp().min())

        # h_augment
        h_augment = self.h_augment()
        # reg_sigma_k = (1 / sigma_k).sum() * self.reg_sigma_k
        # return X_rec, rec_loss, kl_loss, h_augment, reg_sigma_k
        return X_rec, rec_loss, kl_loss, h_augment

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

            # q_eps: batch_size, sample_size,latent_size
            q_eps_mean, q_eps_logvar = self.q_eps_given_x(H).split(self.latent_size, dim=-1)
            # q_eps_logvar = torch.zeros_like(q_eps_mean)

            # latent confouneder
            u = self.confouneder(q_eps_logvar)

            # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
            # q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps, self.A, u).split(self.latent_size, dim=-1)
            q_z_mean = self.q_z_given_epi.cal_mean(q_eps_mean, self.A, u)
            q_z_logvar = self.q_z_given_epi.cal_logvar(q_eps_logvar, self.A, u)

            # q_z: batch_size, sample_size, mc_sample_size, latent_size
            q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(
                    tr_sample_size + te_sample_size), :, :].view(batch_size, te_sample_size * self.test_mc_sample_size,
                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

            pi_k, mu_k, sigma_k = self.get_semisupervised_prior(
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
            ) + torch.log(pi_k[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred

class GMVAE_(nn.Module):  # b old
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
                 h_lambda=0.1,
                 h_c=0.1,
                 eval_iter=15,
                 laplace=False):
        super().__init__()

        self.eval_iter = eval_iter
        self.h_c = h_c
        self.h_lambda = h_lambda
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
        self.A = Parameter(torch.zeros([self.latent_size, self.latent_size]).cuda(), requires_grad=True)

        self.encoder = OmniglotEncoder(
            hidden_size=hidden_size
        )
        self.q_eps_given_x = nn.Sequential(
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
            nn.Linear(self.last_hidden_size, self.hidden_size)
        )

        self.q_z_given_epi = CausalDecoder(self.latent_size)

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
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
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def reparametrize(self, mean, logvar, S=1, laplace=False):
        mean = mean.unsqueeze(2).repeat(1, 1, S, 1)
        logvar = logvar.unsqueeze(2).repeat(1, 1, S, 1)
        if laplace:
            std = logvar.exp()
            eps = Laplace(0.0, 1.0).sample(mean.shape).to(mean.device)
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
        return student_t.bmm(tau)

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1,
                                                                                                                  self.latent_size)

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
        mean = (supervised_mean + unsupervised_mean) / denominator

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

        return pi, mean, logvar

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0)

        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def kl_loss(self, mean, logvar, laplace=False):
        if laplace:
            mean = mean.abs()
            # return (-logvar - 1 + mean / logvar.exp() + (-mean / logvar.exp()).exp()).sum()
            return (-logvar - 1 + mean + (-mean / logvar.exp() + logvar).exp()).sum(-1)
        else:  # gaussian
            return 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum(-1)

    def h_augment(self):
        h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.latent_size
        h_augment = self.h_lambda * h + 0.5 * self.h_c * h * h
        # adjust hyper params
        self.h_lambda += self.h_c * h.item()
        if h.item() > self.h_old * 0.25 and self.h_c < 1e12:
            self.h_c *= 10
        self.h_old = h.item()
        # print(self.A.sum())
        return h_augment

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        H_m = H.mean(1, keepdim=True)
        # q_eps: batch_size, sample_size,latent_size
        q_eps_mean = self.q_eps_given_x(H - H_m)
        q_eps_logvar = torch.zeros_like(q_eps_mean)

        # latent confouneder
        u = self.confouneder(q_eps_logvar)

        # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
        q_eps = q_eps_mean
        q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps, self.A, u).split(self.latent_size, dim=-1)

        # q_eps_mean = self.q_eps_given_z(qz, self.A, u)
        # q_z: batch_size, sample_size, mc_sample_size, latent_size
        # q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps_mean, self.A, u).split(self.latent_size, dim=-1)
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

        # p_z, deep_gmm, batch_size, mc_sample_size, k
        pi_k, mu_k = self.get_unsupervised_prior(q_z.view(batch_size, -1, self.latent_size))

        H_rec = self.proj(q_z.view(-1, self.latent_size))

        H_m = H_m[:, :, None, :].repeat(
            1, sample_size, self.train_mc_sample_size, 1
        ).view(-1, self.last_hidden_size)

        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec + H_m
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        log_qz = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z,
            q_z_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )  # batch_size, sample_size, mc_sample_size

        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
            # sigma_k.log()[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))

        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        # q_z_ = q_z.view(batch_size, -1, self.latent_size)
        # log_pz = ((-0.5 * (q_z_.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2) / sigma_k.unsqueeze(1)).sum(-1).exp() / (
        #         2 * np.pi * sigma_k.prod(-1).unsqueeze(1)).sqrt() * pi_k.unsqueeze(1)).add(1e-14).sum(-1).log()

        kl_loss = log_qz.mean() - log_pz.mean() + self.kl_loss(q_eps_mean, q_eps_logvar, self.laplace).mean()
        # kl_loss = log_qz.mean() - log_pz.mean()
        # print(log_qz.mean(), log_pz.mean(), sigma_k.exp().min())

        # h_augment
        h_augment = self.h_augment()
        # reg_sigma_k = (1 / sigma_k).sum() * self.reg_sigma_k
        # return X_rec, rec_loss, kl_loss, h_augment, reg_sigma_k
        return X_rec, rec_loss, kl_loss, h_augment

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

            # q_eps: batch_size, sample_size,latent_size
            q_eps_mean = self.q_eps_given_x(H - H.mean(1, keepdim=True))
            q_eps_logvar = torch.zeros_like(q_eps_mean)

            # latent confouneder
            # u = torch.zeros_like(q_eps_logvar)
            u = self.confouneder(q_eps_logvar)

            # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
            q_eps = q_eps_mean
            q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps, self.A, u).split(self.latent_size, dim=-1)

            # q_z: batch_size, sample_size, mc_sample_size, latent_size
            q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(
                    tr_sample_size + te_sample_size), :, :].view(batch_size, te_sample_size * self.test_mc_sample_size,
                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

            pi_k, mu_k, sigma_k = self.get_semisupervised_prior(
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
            ) + torch.log(pi_k[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred


class GMVAE_(nn.Module):  # b old
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
                 h_lambda=0.1,
                 h_c=0.1,
                 eval_iter=15,
                 laplace=False):
        super().__init__()

        self.eval_iter = eval_iter
        self.h_c = h_c
        self.h_lambda = h_lambda
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
        self.A = Parameter(torch.zeros([self.latent_size, self.latent_size]).cuda(), requires_grad=True)

        self.encoder = OmniglotEncoder(
            hidden_size=hidden_size
        )
        self.q_eps_given_x = nn.Sequential(
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
            nn.Linear(self.last_hidden_size, self.hidden_size)
        )

        self.q_z_given_epi = CausalDecoder(self.latent_size)

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
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
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def reparametrize(self, mean, logvar, S=1, laplace=False):
        mean = mean.unsqueeze(2).repeat(1, 1, S, 1)
        logvar = logvar.unsqueeze(2).repeat(1, 1, S, 1)
        if laplace:
            std = logvar.exp()
            eps = Laplace(0.0, 1.0).sample(mean.shape).to(mean.device)
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
        return student_t.bmm(tau)

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1,
                                                                                                                  self.latent_size)

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
        mean = (supervised_mean + unsupervised_mean) / denominator

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

        return pi, mean, logvar

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0)

        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def kl_loss(self, mean, logvar, laplace=False):
        if laplace:
            mean = mean.abs()
            # return (-logvar - 1 + mean / logvar.exp() + (-mean / logvar.exp()).exp()).sum()
            return (-logvar - 1 + mean + (-mean / logvar.exp() + logvar).exp()).sum(-1)
        else:  # gaussian
            return 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum(-1)

    def h_augment(self):
        h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.latent_size
        h_augment = self.h_lambda * h + 0.5 * self.h_c * h * h
        # adjust hyper params
        self.h_lambda += self.h_c * h.item()
        if h.item() > self.h_old * 0.25 and self.h_c < 1e12:
            self.h_c *= 10
        self.h_old = h.item()
        # print(self.A.sum())
        return h_augment

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        H_m = H.mean(1, keepdim=True)
        # q_eps: batch_size, sample_size,latent_size
        q_eps_mean = self.q_eps_given_x(H - H_m)
        q_eps_logvar = torch.zeros_like(q_eps_mean)

        # latent confouneder
        u = self.confouneder(q_eps_logvar)

        # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
        q_eps = q_eps_mean
        q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps, self.A, u).split(self.latent_size, dim=-1)

        # q_eps_mean = self.q_eps_given_z(qz, self.A, u)
        # q_z: batch_size, sample_size, mc_sample_size, latent_size
        # q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps_mean, self.A, u).split(self.latent_size, dim=-1)
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

        # p_z, deep_gmm, batch_size, mc_sample_size, k
        pi_k, mu_k = self.get_unsupervised_prior(q_z.view(batch_size, -1, self.latent_size))

        H_rec = self.proj(q_z.view(-1, self.latent_size))

        H_m = H_m[:, :, None, :].repeat(
            1, sample_size, self.train_mc_sample_size, 1
        ).view(-1, self.last_hidden_size)

        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec + H_m
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        log_qz = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z,
            q_z_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )  # batch_size, sample_size, mc_sample_size

        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
            # sigma_k.log()[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))

        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        # q_z_ = q_z.view(batch_size, -1, self.latent_size)
        # log_pz = ((-0.5 * (q_z_.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2) / sigma_k.unsqueeze(1)).sum(-1).exp() / (
        #         2 * np.pi * sigma_k.prod(-1).unsqueeze(1)).sqrt() * pi_k.unsqueeze(1)).add(1e-14).sum(-1).log()

        kl_loss = log_qz.mean() - log_pz.mean() + self.kl_loss(q_eps_mean, q_eps_logvar, self.laplace).mean()
        # kl_loss = log_qz.mean() - log_pz.mean()
        # print(log_qz.mean(), log_pz.mean(), sigma_k.exp().min())

        # h_augment
        h_augment = self.h_augment()
        # reg_sigma_k = (1 / sigma_k).sum() * self.reg_sigma_k
        # return X_rec, rec_loss, kl_loss, h_augment, reg_sigma_k
        return X_rec, rec_loss, kl_loss, h_augment

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

            # q_eps: batch_size, sample_size,latent_size
            q_eps_mean = self.q_eps_given_x(H - H.mean(1, keepdim=True))
            q_eps_logvar = torch.zeros_like(q_eps_mean)

            # latent confouneder
            # u = torch.zeros_like(q_eps_logvar)
            u = self.confouneder(q_eps_logvar)

            # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
            q_eps = q_eps_mean
            q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps, self.A, u).split(self.latent_size, dim=-1)

            # q_z: batch_size, sample_size, mc_sample_size, latent_size
            q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(
                    tr_sample_size + te_sample_size), :, :].view(batch_size, te_sample_size * self.test_mc_sample_size,
                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

            pi_k, mu_k, sigma_k = self.get_semisupervised_prior(
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
            ) + torch.log(pi_k[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred


import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace, StudentT, MultivariateNormal, Chi2
from torch.nn import Parameter

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
                 h_lambda=0.1,
                 h_c=0.1,
                 eval_iter=15,
                 laplace=True):
        super().__init__()

        self.eval_iter = eval_iter
        self.h_c = h_c
        self.h_lambda = h_lambda
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
        self.A = Parameter(torch.zeros([self.latent_size, self.latent_size]).cuda(), requires_grad=True)

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
            nn.Linear(self.last_hidden_size, self.hidden_size)
        )

        self.q_eps_given_z = CausalEncoder(self.latent_size)
        self.q_z_given_eps = CausalDecoder(self.latent_size)

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
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
        self.register_buffer('uniform_pi', torch.ones(self.component_size) / self.component_size)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def reparametrize(self, mean, logvar, S=1, laplace=False):
        mean = mean.unsqueeze(2).repeat(1, 1, S, 1)
        logvar = logvar.unsqueeze(2).repeat(1, 1, S, 1)
        if laplace:
            std = logvar.exp()
            eps = Laplace(0.0, 1.0).sample(mean.shape).to(mean.device)
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

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1,
                                                                                                                  self.latent_size)

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
        mean = (supervised_mean + unsupervised_mean) / denominator

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

        return pi, mean, logvar

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0)

        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

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

    def h_augment(self):
        h = torch.trace(torch.matrix_exp(self.A * self.A)) - self.latent_size
        h_augment = self.h_lambda * h + 0.5 * self.h_c * h * h
        # adjust hyper params
        self.h_lambda += self.h_c * h.item()
        if h.item() > self.h_old * 0.25 and self.h_c < 1e12:
            self.h_c *= 10
        self.h_old = h.item()
        # print(self.A.sum())
        return h_augment

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_eps: batch_size, sample_size, latent_size
        q_zh = self.q_z_given_H(H)

        # latent confouneder
        u = self.confouneder(q_zh)

        q_eps_mean = self.q_eps_given_z(q_zh, self.A, u)

        # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
        q_z_mean, q_z_logvar = self.q_z_given_eps(q_eps_mean, self.A, u).split(self.latent_size, dim=-1)

        # q_eps_mean = self.q_eps_given_z(qz, self.A, u)
        # q_z: batch_size, sample_size, mc_sample_size, latent_size
        # q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps_mean, self.A, u).split(self.latent_size, dim=-1)
        q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

        # p_z, deep_gmm, batch_size, mc_sample_size, k
        pi_k, mu_k = self.get_unsupervised_prior(q_z.view(batch_size, -1, self.latent_size))

        H_rec = self.proj(q_z.view(-1, self.latent_size))

        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        log_qz = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, latent_size
            q_z,
            q_z_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )  # batch_size, sample_size, mc_sample_size

        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
            # sigma_k.log()[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))

        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        # q_z_ = q_z.view(batch_size, -1, self.latent_size)
        # log_pz = ((-0.5 * (q_z_.unsqueeze(2) - mu_k.unsqueeze(1)).pow(2) / sigma_k.unsqueeze(1)).sum(-1).exp() / (
        #         2 * np.pi * sigma_k.prod(-1).unsqueeze(1)).sqrt() * pi_k.unsqueeze(1)).add(1e-14).sum(-1).log()

        kl_loss = 0 * log_qz.mean() - log_pz.mean() + self.kl_loss_semi(mean=q_eps_mean,
                                                                        laplace=self.laplace).mean()
        # kl_loss = log_qz.mean() - log_pz.mean()
        # print(log_qz.mean(), log_pz.mean(), sigma_k.exp().min())

        # h_augment
        h_augment = self.h_augment()
        reg_A = torch.sum(self.A.abs()) * 0
        # reg_sigma_k = (1 / sigma_k).sum() * self.reg_sigma_k
        # return X_rec, rec_loss, kl_loss, h_augment, reg_sigma_k
        return X_rec, rec_loss, kl_loss, h_augment + 5e-3 * reg_A

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

            q_zh = self.q_z_given_H(H)

            # latent confouneder
            u = self.confouneder(q_zh)

            q_eps_mean = self.q_eps_given_z(q_zh, self.A, u)

            # q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, 1, self.laplace).squeeze(2)
            q_z_mean, q_z_logvar = self.q_z_given_eps(q_eps_mean, self.A, u).split(self.latent_size, dim=-1)

            # q_eps_mean = self.q_eps_given_z(qz, self.A, u)
            # q_z: batch_size, sample_size, mc_sample_size, latent_size
            # q_z_mean, q_z_logvar = self.q_z_given_epi(q_eps_mean, self.A, u).split(self.latent_size, dim=-1)
            q_z = self.reparametrize(q_z_mean, q_z_logvar, self.train_mc_sample_size)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(
                    tr_sample_size + te_sample_size), :, :].view(batch_size, te_sample_size * self.test_mc_sample_size,
                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

            pi_k, mu_k, sigma_k = self.get_semisupervised_prior(
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
            ) + torch.log(pi_k[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred



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
                 h_lambda=1,
                 h_c=1,
                 eval_iter=15,
                 laplace=False):
        super().__init__()

        self.eval_iter = eval_iter
        self.h_c = h_c
        self.h_lambda = h_lambda
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
        self.A = Parameter(torch.tril(torch.randn([self.latent_size, self.latent_size]), diagonal=-1).cuda(),
                           requires_grad=True)

        self.encoder = OmniglotEncoder(
            hidden_size=hidden_size
        )
        self.q_eps_given_H = nn.Sequential(
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

        self.q_z_given_eps = CausalDecoder(self.hidden_size)

        self.proj = nn.Sequential(
            nn.Linear(latent_size, self.last_hidden_size),
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

    def h_augment(self, A, h_lambda, h_c, h_old):
        h = torch.trace(torch.matrix_exp(A * A)) - self.latent_size
        h_augment = h_lambda * h + 0.5 * h_c * h * h
        # adjust hyper params
        h_lambda += h_c * h.item()
        if h.item() > h_old * 0.25 and h_c < 1e12:
            h_c *= 10
        h_old = h.item()
        # print(self.A.sum())
        return h_augment, h_lambda, h_c, h_old

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)

        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        supervised_N = torch.sum(y, dim=1)
        # batch_size, component_size
        unsupervised_N = torch.sum(posteriors, dim=1)
        # batch_size, component_size, latent_size
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1,
                                                                                                                  self.latent_size)

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
        mean = (supervised_mean + unsupervised_mean) / denominator

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

        return pi, mean, logvar

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0)

        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_eqi_poster(self, q, mean, logvar, laplace=False):
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

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_eps_mean: batch_size, sample_size, hidden_size
        q_eps_mean, q_eps_logvar = self.q_eps_given_H(H).split(self.latent_size, dim=-1)
        # q_eps: batch_size, sample_size, train_mc_sample_size, latent_size
        q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, self.train_mc_sample_size, self.laplace)
        # q_eps: batch_size, sample_size*train_mc_sample_size, latent_size

        q_eps = q_eps.view(batch_size, -1, self.latent_size)
        u = self.confouneder(q_eps)

        # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
        q_z = self.q_z_given_eps(q_eps + u, self.A)

        H_rec = self.proj(q_z.view(-1, self.latent_size))

        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        pi_k, mu_k = self.get_unsupervised_prior(q_z)
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z.view(batch_size, sample_size, self.train_mc_sample_size, self.latent_size)[:, :, :, None, :].repeat(
                1, 1, 1, self.component_size, 1),
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1),
            # sigma_k.log()[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))

        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        log_q_epi = self.get_eqi_poster(
            q_eps.view(batch_size, sample_size, self.train_mc_sample_size, self.latent_size),
            q_eps_mean,
            q_eps_logvar,
            self.laplace)

        kl_loss = log_q_epi.mean() - log_pz.mean()
        # h_augment
        h_augment, self.h_lambda, self.h_c, self.h_old = self.h_augment(self.A, self.h_lambda, self.h_c, self.h_old)
        # print(self.h_lambda, self.A.abs().sum())
        return X_rec, rec_loss, kl_loss, h_augment

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

            # q_eps_mean: batch_size, sample_size, hidden_size
            q_eps_mean, q_eps_logvar = self.q_eps_given_H(H).split(self.latent_size, dim=-1)
            # q_eps: batch_size, sample_size, train_mc_sample_size, latent_size
            q_eps = self.reparametrize(q_eps_mean, q_eps_logvar, self.train_mc_sample_size, self.laplace)
            # q_eps: batch_size, sample_size*train_mc_sample_size, latent_size
            q_eps = q_eps.view(batch_size, -1, self.latent_size)
            u = self.confouneder(q_eps)

            # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
            q_z = self.q_z_given_eps(q_eps + u, self.A).view(
                batch_size, tr_sample_size + te_sample_size, self.test_mc_sample_size, self.latent_size)

            ## p z ##
            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(
                    tr_sample_size + te_sample_size), :, :].view(batch_size, te_sample_size * self.test_mc_sample_size,
                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

            pi_k, mu_k, sigma_k = self.get_semisupervised_prior(
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
            ) + torch.log(pi_k[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred




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

                 eval_iter=15,
                 laplace=False):
        super().__init__()

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

        self.q_epi_given_z = CausalEncoder(self.hidden_size)
        self.p_z_given_epi = CausalDecoder(self.hidden_size)

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

    def get_unsupervised_params(self, X, inverse, A, psi):
        sample_size = X.shape[1]
        pi, mean = psi

        # batch_size, sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, component_size, latent_size
            X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(
            pi[:, None, :].repeat(1, sample_size, 1)
        ) + self.gaussian_log_prob(
            # batch_size, component_size, latent_size
            mean.bmm((self.I - A).unsqueeze(0).repeat(mean.shape[0], 1, 1)),
            # batch_size, component_size, latent_size
            torch.zeros_like(mean),
        ).unsqueeze(1)  # batch_size, component_size ->  batch_size, sample_size, component_size

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
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
            X.bmm(inverse)
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, inverse, A, psi):
        # inverse: batch_size, component_size, latent_size, latent_size
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1),
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(
            pi[:, None, :].repeat(1, unsupervised_sample_size, 1)
        ) + self.gaussian_log_prob(
            # batch_size, component_size, latent_size
            mean.bmm((self.I - A).unsqueeze(0).repeat(mean.shape[0], 1, 1)),
            # batch_size, component_size, latent_size
            torch.zeros_like(mean),
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
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1,
                                                                                                                  self.latent_size)

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

        return pi, mean, logvar

    def get_unsupervised_prior(self, z, A):
        batch_size, sample_size = z.shape[0], z.shape[1]
        IminusA2 = (self.I - A).mm(self.I - A.t())
        # batch_size, latent_size, latent_size
        inverse = torch.inverse(self.I + IminusA2).unsqueeze(0).repeat(batch_size, 1, 1)
        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
        # batch_size, component_size, latent_size
        initial_mean = torch.stack(initial_mean, dim=0).bmm(inverse)

        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z,
                inverse=inverse,  # for reduce computation
                A=A,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y, A):
        batch_size = unsupervised_z.shape[0]
        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        component_size = initial_pi.shape[1]
        IminusA2 = (self.I - A).mm(self.I - A.t())
        inverse = torch.inverse(self.I + IminusA2).unsqueeze(0).repeat(batch_size, 1, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
            # batch_size, component_size, sample_size
            y.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            supervised_z.bmm(inverse)
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
        # batch_size, component_size, latent_size
        initial_logvar = torch.zeros_like(initial_mean)

        IminusA2 = (self.I - A).mm(self.I - A.t()).unsqueeze(0).unsqueeze(0).repeat(batch_size, component_size, 1, 1)

        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            # batch_size, component_size, latent_size, latent_size
            inverse = torch.inverse(self.I.unsqueeze(0).unsqueeze(0) + IminusA2.matmul(torch.diag_embed(psi[2].exp())))
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                inverse=inverse,
                A=A,
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
        u = 0
        # q_z: batch_size, sample_size*train_mc_sample_size, latent_size
        q_epi = self.q_epi_given_z(q_z - q_z.mean(1, keepdim=True), u)
        A = self.q_epi_given_z.get_w()

        # H_rec = self.proj(q_z.view(-1, self.latent_size))
        H_rec = self.proj(torch.cat([q_z, q_epi], -1).view(-1, 2 * self.latent_size))

        # batch_size*sample_size*mc_sample_size, 1, 28, 28
        X_rec = self.decoder(
            H_rec,
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(
            X_rec,
            X[:, :, None, :, :, :].repeat(1, 1, self.train_mc_sample_size, 1, 1, 1)
        ) / (batch_size * sample_size * self.train_mc_sample_size)

        # epi_loss = self.kl_loss_semi(q_epi)

        # mu_k： batch_size, component_size, latent_size
        # pi_k： batch_size, component_size
        pi_k, mu_k = self.get_unsupervised_prior(q_z, A.detach())
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            q_z.view(batch_size, sample_size, self.train_mc_sample_size, self.latent_size)[:, :, :, None, :].repeat(
                1, 1, 1, self.component_size, 1),
            # batch_size, sample_size, mc_sample_size, component_size, latent_size
            mu_k[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(
            pi_k[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1)
        ) + self.gaussian_log_prob(
            # batch_size, component_size, latent_size
            mu_k.bmm((self.I - A.detach()).unsqueeze(0).repeat(mu_k.shape[0], 1, 1)),
            # batch_size, component_size, latent_size
            torch.zeros_like(mu_k),
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
        q_epi = q_epi.view(batch_size, sample_size, self.train_mc_sample_size, self.latent_size)

        log_p_eps = self.gaussian_log_prob(
            q_epi,
            torch.zeros_like(q_epi),
        )

        kl_loss = log_q_z.mean() - log_p_z.mean() - log_p_eps.mean()
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
            # q_z = q_z.view(batch_size, -1, self.latent_size)

            A = self.q_epi_given_z.get_w()

            # batch_size, te_sample_size*mc_sample_size, latent_size
            unsupervised_z = q_z[:, tr_sample_size:(
                    tr_sample_size + te_sample_size), :, :].view(batch_size, te_sample_size * self.test_mc_sample_size,
                                                                 self.latent_size)
            # batch_size, tr_sample_size*mc_sample_size, latent_size
            supervised_z = q_z[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size * self.test_mc_sample_size,
                                                              self.latent_size)

            # batch_size, tr_sample_size*mc_sample_size
            y = y_tr.view(
                batch_size * tr_sample_size
            )[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size * self.test_mc_sample_size)
            # batch_size, tr_sample_size*mc_sample_size, component_size
            y = F.one_hot(y, self.component_size).float()

            pi_k, mu_k, sigma_k = self.get_semisupervised_prior(
                unsupervised_z=unsupervised_z,
                supervised_z=supervised_z,
                y=y,
                A=A,
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
                mu_k.bmm((self.I - A).unsqueeze(0).repeat(mu_k.shape[0], 1, 1)),
                # batch_size, component_size, latent_size
                torch.zeros_like(mu_k),
            ).unsqueeze(1).unsqueeze(
                2)  # batch_size, component_size ->  batch_size, te_sample_size, mc_sample_size, component_size

            posteriors = torch.exp(
                log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True))
            y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)

            return y_te_pred
