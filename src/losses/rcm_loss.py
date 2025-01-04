from loguru import logger

import torch
import torch.nn as nn


class RCMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['rcm']['loss']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']
        self.criterion = nn.CrossEntropyLoss().cuda()

    def compute_coarse_loss_s2d(self, data, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        conf = data['attn']  # [b, num_kpt, hw*hw+1, L]
        conf_gt = data['conf_matrix_gt'].unsqueeze(-1).repeat(1, 1, 1, conf.shape[-1])
        pos_mask, neg_mask = conf_gt == 1, conf_gt.sum(-2) == 0

        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w

        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        conf = torch.clamp(conf, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        pos_conf = conf[:, :, :-1, :][pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()

        neg_conf = conf[:, :, -1, :][neg_mask]
        loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()

        loss_pos, loss_neg = c_pos_w * loss_pos.mean(), c_neg_w * loss_neg.mean()
        loss = loss_pos + loss_neg
        return loss, loss_pos, loss_neg

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        # norm_gt = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1)
        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        # inverse_std = 1. / torch.clamp(std, min=1e-10) / torch.clamp(norm_gt, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def compute_scale_loss(self, scale_gt, scale):
        scale = torch.clamp(scale, 1e-6, 1-1e-6)
        gamma = self.loss_config['focal_gamma']
        gt_mask = scale_gt == 1
        conf = scale[gt_mask]
        loss_scale = - torch.pow(1 - conf, gamma) * conf.log()
        loss_scale = loss_scale.mean()
        return loss_scale

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = None

        # 1. coarse-level loss
        loss_c, loss_pos, loss_neg = self.compute_coarse_loss_s2d(data)
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu(),
                             "loss_c_pos": loss_pos.clone().detach().cpu(),
                             "loss_c_neg": loss_neg.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        loss_s = self.compute_scale_loss(data['spv_scale'], data['est_scale'])
        loss += loss_s * self.loss_config['scale_weight']
        loss_scalars.update({"loss_s": loss_s.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
