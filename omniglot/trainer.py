import os
from collections import OrderedDict

from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from model import GMVAE
from data import Data
import wandb


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data = Data(args.data_dir)

        dataset = TensorDataset(torch.from_numpy(self.data.x_mtr).float())
        # sampler = RandomSampler(dataset, replacement=True, )
        self.trloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size * args.sample_size,
            shuffle=True,
            # num_workers=4,
            # pin_memory=True,
            drop_last=True
        )

        self.input_shape = [1, 28, 28]

        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=args.unsupervised_em_iters,
            semisupervised_em_iters=args.semisupervised_em_iters,
            fix_pi=args.fix_pi,
            hidden_size=args.hidden_size,
            component_size=args.way,
            latent_size=args.latent_size,
            train_mc_sample_size=args.train_mc_sample_size,
            test_mc_sample_size=args.test_mc_sample_size
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        epoches = self.args.train_iters // self.args.freq_iters

        self.writer = SummaryWriter(
            log_dir=os.path.join(args.save_dir, "tb_log")
        )
        wandb.watch(self.model, log="all")

        self.tr_image_save_dir = os.path.join(self.args.save_dir, 'tr_samples')
        os.makedirs(self.tr_image_save_dir, exist_ok=True)
        self.te_image_save_dir = os.path.join(self.args.save_dir, 'te_samples')
        os.makedirs(self.te_image_save_dir, exist_ok=True)

    def train(self):
        global_epoch = 0
        global_step = 0
        best_1shot = 0.0
        best_5shot = 0.0
        best_elbo_loss = np.inf
        iterator = iter(self.trloader)
        h_tol, h_c_max = 1e-8, 1e11
        h_lambda, h_c, a_lambda, h_old = 0.1, 0.1, 1e-4, np.inf
        log_dict = {}

        while global_epoch * self.args.freq_iters < self.args.train_iters:
            with tqdm(total=self.args.freq_iters, position=0) as pbar:
                for _ in range(self.args.freq_iters):
                    self.model.train()
                    self.model.zero_grad()
                    try:
                        X = next(iterator)[0]
                    except StopIteration:
                        iterator = iter(self.trloader)
                        X = next(iterator)[0]

                    X = X.to(self.args.device).float()
                    X = X.view(self.args.batch_size, self.args.sample_size, *self.input_shape)

                    X_rec, rec_loss, kl_loss, h_augment, h = self.model(X, h_lambda, h_c, a_lambda)
                    loss = rec_loss + kl_loss + h_augment

                    loss.backward()
                    self.optimizer.step()

                    postfix = OrderedDict(
                        {'rec': '{0:.4f}'.format(rec_loss),
                         'kld': '{0:.4f}'.format(kl_loss),
                         'h_aug': '{0:.4f}'.format(h_augment),
                         }
                    )
                    pbar.set_postfix(**postfix)
                    self.writer.add_scalars(
                        'train',
                        {'rec': rec_loss, 'kld': kl_loss, 'h_aug': h_augment, },
                        global_step
                    )

                    pbar.update(1)
                    global_step += 1

                    if self.args.debug:
                        break

            # self.scheduler.step()
            log_dict.update({'causal/h_lambda': h_lambda, 'causal/h_c': h_c, 'causal/h': h,
                             'train/rec': rec_loss, 'train/kld': kl_loss, 'train/h_aug': h_augment,
                             'lr/lr': self.optimizer.param_groups[0]['lr'],
                             'epoch': global_epoch
                             }, )

            f_batch = 32

            log_dict.update({"image/X|X_rec": [wandb.Image(im) for im in
                                               torch.cat([X[0, :f_batch], X_rec[0, :f_batch, 0]], -1)]})

            # todo: remember to delete test=True
            mean_1shot, conf_1shot = self.eval(shot=1, test=False)
            mean_5shot, conf_5shot = self.eval(shot=5, test=False)

            log_dict.update({'test/1shot-acc-mean': mean_1shot, 'test/5shot-acc-mean': mean_5shot,
                             'test/1shot-acc-conf': conf_1shot, 'test/5shot-acc-conf': conf_5shot,
                             })

            if best_1shot < mean_1shot:
                best_1shot = mean_1shot
                state = {
                    'state_dict': self.model.state_dict(),
                    'accuracy': mean_1shot,
                    'epoch': global_epoch,
                }
                wandb.run.summary["best_1shot"] = mean_1shot
                wandb.run.summary["best_1shot_epoch"] = global_epoch

                torch.save(state, os.path.join(self.args.save_dir, '1shot_best_%s.pth' % global_step))

            print(
                "1shot {0}-th EPOCH Val Accuracy: {1:.4f}, BEST Accuracy: {2:.4f}".format(
                    global_epoch, mean_1shot, best_1shot))

            if best_5shot < mean_5shot:
                best_5shot = mean_5shot
                state = {
                    'state_dict': self.model.state_dict(),
                    'accuracy': mean_5shot,
                    'epoch': global_epoch,
                }
                wandb.run.summary["best_5shot"] = mean_5shot
                wandb.run.summary["best_5shot_epoch"] = global_epoch

                torch.save(state, os.path.join(self.args.save_dir, '5shot_best_%s.pth' % global_step))

            print(
                "5shot {0}-th EPOCH Val Accuracy: {1:.4f}, BEST Accuracy: {2:.4f}".format(
                    global_epoch, mean_5shot, best_5shot, ))

            wandb.log(log_dict)

            global_epoch += 1
            # if h.item() > h_tol:
            #     if h.item() > h_old * 0.25:
            #         if h_c < h_c_max:
            #             h_c *= 10
            #
            #     h_augment += h_c * h.item()
            # h_old = h.item()

        del self.model

        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=self.args.unsupervised_em_iters,
            semisupervised_em_iters=self.args.semisupervised_em_iters,
            fix_pi=self.args.fix_pi,
            hidden_size=self.args.hidden_size,
            component_size=self.args.way,
            latent_size=self.args.latent_size,
            train_mc_sample_size=self.args.train_mc_sample_size,
            test_mc_sample_size=self.args.test_mc_sample_size
        ).to(self.args.device)

        state_dict = torch.load(os.path.join(self.args.save_dir, '1shot_best_1000.pth'))['state_dict']
        self.model.load_state_dict(state_dict)
        mean_1shot, conf_1shot = self.eval(shot=1, test=True)
        print("1shot Final Test Accuracy: {0:.4f} Confidence Interval: {1:.4f}".format(mean_1shot, conf_1shot))

        del self.model

        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=self.args.unsupervised_em_iters,
            semisupervised_em_iters=self.args.semisupervised_em_iters,
            fix_pi=self.args.fix_pi,

            hidden_size=self.args.hidden_size,
            component_size=self.args.way,
            latent_size=self.args.latent_size,
            train_mc_sample_size=self.args.train_mc_sample_size,
            test_mc_sample_size=self.args.test_mc_sample_size
        ).to(self.args.device)

        state_dict = torch.load(os.path.join(self.args.save_dir, '5shot_best_1000.pth'))['state_dict']
        self.model.load_state_dict(state_dict)
        mean_5shot, conf_5shot = self.eval(shot=5, test=True)
        print("5shot Final Test Accuracy: {0:.4f} Confidence Interval: {1:.4f}".format(mean_5shot, conf_5shot))

    def eval(self, shot, test=False):

        self.model.eval()
        all_accuracies = np.array([])
        for _ in tqdm(range(self.args.eval_episodes // self.args.batch_size)):
            X_tr, y_tr, X_te, y_te = self.data.generate_test_episode(
                way=self.args.way,
                shot=shot,
                query=self.args.query,
                n_episodes=self.args.batch_size,
                test=test
            )
            X_tr = torch.from_numpy(X_tr).to(self.args.device).float()
            y_tr = torch.from_numpy(y_tr).to(self.args.device)
            X_te = torch.from_numpy(X_te).to(self.args.device).float()
            y_te = torch.from_numpy(y_te).to(self.args.device)

            y_te_pred = self.model.prediction(X_tr, y_tr, X_te)
            accuracies = torch.mean(torch.eq(y_te_pred, y_te).float(), dim=-1).cpu().numpy()
            all_accuracies = np.concatenate([all_accuracies, accuracies], axis=0)

        all_accuracies = all_accuracies[:self.args.eval_episodes]
        return np.mean(all_accuracies), 1.96 * np.std(all_accuracies) / float(np.sqrt(self.args.eval_episodes))

    def reload_weights(self):
        del self.model

        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=self.args.unsupervised_em_iters,
            semisupervised_em_iters=self.args.semisupervised_em_iters,
            fix_pi=self.args.fix_pi,

            hidden_size=self.args.hidden_size,
            component_size=self.args.way,
            latent_size=self.args.latent_size,
            train_mc_sample_size=self.args.train_mc_sample_size,
            test_mc_sample_size=self.args.test_mc_sample_size
        ).to(self.args.device)

        state_dict = torch.load(os.path.join(self.args.save_dir, '5shot_best.pth'))['state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        iterator = iter(self.trloader)

        # save train samples
        try:
            X = next(iterator)[0]
        except StopIteration:
            iterator = iter(self.trloader)
            X = next(iterator)[0]

        X = X.to(self.args.device).float()
        X = X.view(self.args.batch_size, self.args.sample_size, *self.input_shape)
        y = self.model.tr_prediction(X)
        True_X, y = X[0], y[0]
        self.save_tr_img(True_X, y)

        # save test samples
        X_tr, y_tr, X_te, y_te = self.data.generate_test_episode(
            way=self.args.way,
            shot=5,
            query=15,
            n_episodes=self.args.batch_size,
            test="test"
        )
        X_tr = torch.from_numpy(X_tr).to(self.args.device).float()
        y_tr = torch.from_numpy(y_tr).to(self.args.device)
        X_te = torch.from_numpy(X_te).to(self.args.device).float()
        y_te = torch.from_numpy(y_te).to(self.args.device)
        y_te_pred = self.model.prediction(X_tr, y_tr, X_te)[0]

        True_X_5shot, Recon_X_5shot, Component_X_5shot, inter_X_5shot = self.model.generate(X_tr, y_tr, X_te, seed=10086)

        self.save_te_img(
            shot=5,
            True_X=1 - True_X_5shot,
            Recon_X=1 - Recon_X_5shot,
            Component_X=1 - Component_X_5shot,
            inter_X_5shot=1 - inter_X_5shot,
            y_te_pred=y_te_pred
        )

    def save_tr_img(self, True_X, y):

        for i in range(self.args.way):
            if len(True_X[y == i]) == 0:
                True_images = torchvision.utils.make_grid(torch.zeros((1, 1, 28, 28)), nrow=1, padding=2, pad_value=255)
                torchvision.utils.save_image(
                    True_images,
                    os.path.join(self.tr_image_save_dir, '{0}th_True.png'.format(i + 1))
                )
            else:
                True_images = torchvision.utils.make_grid(True_X[y == i], nrow=20, padding=2, pad_value=255)
                torchvision.utils.save_image(
                    True_images,
                    os.path.join(self.tr_image_save_dir, '{0}th_True.png'.format(i + 1))
                )

    def save_te_img(self, shot, True_X, Recon_X, Component_X, inter_X_5shot, y_te_pred):

        res_true = []
        res_rec = []
        res_inter = []
        for i in range(5):
            res_true.append(True_X[y_te_pred == i][:5])
            res_rec.append(Recon_X[y_te_pred == i][:5])
            res_inter.append(inter_X_5shot[y_te_pred == i][:5])
        res_true = torch.cat(res_true, 0)
        res_rec = torch.cat(res_rec, 0)
        res_inter = torch.cat(res_inter, 0)

        True_images = torchvision.utils.make_grid(res_true, nrow=5, padding=2, pad_value=255)
        torchvision.utils.save_image(
            True_images,
            os.path.join(self.te_image_save_dir, '{0}shot_True.png'.format(shot))
        )
        Recon_images = torchvision.utils.make_grid(res_rec, nrow=5, padding=2, pad_value=255)
        torchvision.utils.save_image(
            Recon_images,
            os.path.join(self.te_image_save_dir, '{0}shot_Recon.png'.format(shot))
        )

        Inter_images = torchvision.utils.make_grid(res_inter, nrow=5, padding=2, pad_value=255)
        torchvision.utils.save_image(
            Inter_images,
            os.path.join(self.te_image_save_dir, '{0}shot_Inter.png'.format(shot))
        )
