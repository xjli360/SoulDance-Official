# Copyright (c) 2023 Chuan Guo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_m2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *
from MMR.src.model.losses import InfoNCE_with_filtering

from einops import rearrange, repeat

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model, eval_wrapper):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=0.7, threshold_selfsim=0.8
        )

        self.mmr = eval_wrapper

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        code_idx, _ = self.vq_model.encode(motion)  # torch.Size([64, 75*3, 6])
        m_lens = m_lens // 4
        m_lens = m_lens * 3 if self.opt.hrvq else m_lens
        # [64, 256]
        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        cls_loss, _pred, _acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens)
        
        infonce_loss = torch.tensor(0.0, requires_grad=True)
        if self.opt.mmr_loss:
            if self.opt.dataset_name == 'finedance':
                m_len_max = 37 # 150 //4
            else:
                m_len_max = 75
            nb_code = self.opt.nb_code
            code_dim = 512
            if self.opt.hrvq:
                vq_code_all = torch.arange(0, nb_code, dtype=code_idx.dtype).unsqueeze(0).expand(1, -1).unsqueeze(-1).repeat(1,3,1).to(self.device)
            else:
                vq_code_all = torch.arange(0, nb_code, dtype=code_idx.dtype).unsqueeze(0).expand(1, -1).unsqueeze(-1).to(self.device)

            vq_lantent_all = self.vq_model.quantizer.get_codebook_entry(vq_code_all).squeeze(0)[:code_dim,:]
            soft_score = gumbel_softmax_sample(_pred.reshape(-1,  m_len_max, nb_code),temperature=0.01)
            vq_lantent = soft_score @ vq_lantent_all.T
            _pre_motions = self.vq_model.decoder(vq_lantent.reshape(vq_lantent.shape[0], -1, m_len_max))
            m_latents = self.mmr.get_mmr_motion_embeddings(_pre_motions[...,:263])
            infonce_loss = self.contrastive_loss_fn(conds, m_latents)
        
        total_loss = cls_loss * 1.0 + infonce_loss * 0.5

        return total_loss, infonce_loss, _acc

    def update(self, batch_data):
        loss, infonce_losss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), infonce_losss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_acc = 0.
        best_fid = 10000

        while epoch < self.opt.max_epoch:
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, infonce_loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['infonce_loss'] += infonce_loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            val_infonce_loss = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, infonce_loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_infonce_loss.append(infonce_loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, infonce loss:{np.mean(val_infonce_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)
            self.logger.add_scalar('Val/infonce_loss', np.mean(val_infonce_loss), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_mmr, writer = evaluation_mask_transformer(self.opt, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model, eval_wrapper):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=0.7, threshold_selfsim=0.8
        )

        self.mmr = eval_wrapper

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            # self.l1_criterion = torch.nn.SmoothL1Loss()


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        m_lens = m_lens * 3 if self.opt.hrvq else m_lens
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        cls_loss, _pred, _acc = self.res_transformer(code_idx, conds, m_lens)
        
        infonce_loss = torch.tensor(0.0, requires_grad=True)
        if self.opt.mmr_loss:
            if self.opt.dataset_name == 'finedance':
                m_len_max = 37 # 150 //4
            else:
                m_len_max = 75
            nb_code = self.opt.nb_code
            code_dim = 512
            # print(self.opt.nb_code)
            if self.opt.hrvq:
                vq_code_all = torch.arange(0, nb_code, dtype=code_idx.dtype).unsqueeze(0).expand(1, -1).unsqueeze(-1).repeat(1,3,1).to(self.device)
            else:
                vq_code_all = torch.arange(0, nb_code, dtype=code_idx.dtype).unsqueeze(0).expand(1, -1).unsqueeze(-1).to(self.device)

            vq_lantent_all = self.vq_model.quantizer.get_codebook_entry(vq_code_all).squeeze(0)[:code_dim,:]
            soft_score = gumbel_softmax_sample(_pred.reshape(-1,  m_len_max, nb_code),temperature=0.01)
            vq_lantent = soft_score @ vq_lantent_all.T
            _pre_motions = self.vq_model.decoder(vq_lantent.reshape(vq_lantent.shape[0], -1, m_len_max))

            m_latents = self.mmr.get_mmr_motion_embeddings(_pre_motions[...,:263])
            infonce_loss = self.contrastive_loss_fn(conds, m_latents)
        
        total_loss = cls_loss * 1.0 + infonce_loss * 0.5

        return total_loss, infonce_loss, _acc

    def update(self, batch_data):
        loss, infonce_losss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), infonce_losss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_acc = 0.
        best_fid = 10000

        while epoch < self.opt.max_epoch:
            self.res_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                # print("Bath idx: {}".format(i))

                loss, infonce_loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['infonce_loss'] += infonce_loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            val_infonce_loss = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, infonce_loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_infonce_loss.append(infonce_loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, infonce loss:{np.mean(val_infonce_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)
            self.logger.add_scalar('Val/infonce_loss', np.mean(val_infonce_loss), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                # self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_mmr_score,writer = evaluation_res_transformer(
            self.opt, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch, best_fid,eval_wrapper=eval_wrapper, plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
        )