import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
import math
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from src.rcm.utils.utils import Assign, normalize_keypoints, KeypointEncoder
from .attention import SelfAttn, CrossAttn, CrossAttn_d2s, CrossLin_s2d
from timm.models.layers import trunc_normal_
from .extractor_lite import Extractor
from src.rcm.utils.supervision import compute_supervision_coarse_scale
from src.rcm.utils.utils import switch_back4


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, num_layers, flash=False):
        super().__init__()
        self.num_layers = num_layers
        self.self_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.self_layers.append(SelfAttn(feature_dim, 4, flash=flash))
            self.cross_layers.append(CrossAttn(feature_dim, 4, flash=flash))

    def forward(self, data, desc0, desc1):
        mask0 = data['mask0'] if 'mask0' in data else None
        mask1 = data['mask1'].flatten(1, 2) if 'mask1' in data else None
        desc0_l, desc1_l = [], []
        desc0, desc1 = desc0.transpose(1, 2), desc1.transpose(1, 2)
        for i in range(self.num_layers):
            desc0, desc1 = self.self_layers[i](desc0, desc1, mask0, mask1)
            desc0, desc1 = self.cross_layers[i](desc0, desc1, mask0, mask1)
            desc0_l.append(desc0.transpose(1, 2))
            desc1_l.append(desc1.transpose(1, 2))
        data.update({
            'desc0_l': desc0_l,
            'desc1_l': desc1_l,
        })
        return desc0.transpose(1, 2), desc1.transpose(1, 2)


class AttentionalGNN_fine(nn.Module):
    def __init__(self, feature_dim: int, num_layers: list, flash=False, window=5):
        super().__init__()
        self.num_layers = num_layers
        self.layers_sparse = nn.ModuleList([
            CrossAttn_d2s(feature_dim, 4, flash=flash)
            for _ in range(num_layers)])
        self.layers_dense = nn.ModuleList([
            CrossLin_s2d(feature_dim, window)
            for _ in range(num_layers)])

    def forward(self, desc0, desc1):
        desc0, desc1 = desc0.transpose(1, 2), desc1.transpose(1, 2)
        for i in range(self.num_layers):
            desc0, desc1 = self.layers_sparse[i](desc0, desc1), self.layers_dense[i](desc1, desc0)
        return desc0.transpose(1, 2), desc1.transpose(1, 2)


class RCM(nn.Module):
    default_config = {
        'weights_path': None,
        'keypoint_encoder': [32, 64, 128, 256],
        'match_threshold': 0.2,
        'learnable_temp': True,
        'num_layers_c': 5,
        'num_layers_f': 2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.train_coarse_percent = self.config['match_coarse']['train_coarse_percent']
        self.train_pad_num_gt_min = self.config['match_coarse']['train_pad_num_gt_min']
        self.d_model_c = self.config['coarse']['d_model']
        self.d_model_f = self.config['fine']['d_model']

        self.extractor = Extractor(config)
        self.kenc = KeypointEncoder(self.d_model_c, self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(self.d_model_c, self.config['num_layers_c'], flash=False)

        self.gnn_fine = AttentionalGNN_fine(self.d_model_f, self.config['num_layers_f'], flash=False, window=self.config['fine_window_size'])

        self.down_proj = nn.Linear(self.d_model_c, self.d_model_f, bias=True)
        self.merge_feat = nn.Linear(2 * self.d_model_f, self.d_model_f, bias=True)

        self.dustbin = nn.Parameter(torch.zeros(1, self.d_model_c, 1))
        trunc_normal_(self.dustbin, std=0.2)
        self.register_parameter('dustbin', self.dustbin)

        self.assign = Assign(dim=256, learnable_temp=self.config['learnable_temp'])

    def coarse_match(self, data):
        desc0, desc1 = data['descriptors0'], rearrange(data['descriptors1'], 'b c h w -> b c (h w)')
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'].unsqueeze(1), data['scores1'].unsqueeze(1)
        b, c, hw = desc1.shape

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])

        # Keypoint MLP encoder.
        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(data, desc0, desc1)
        if self.mode == 'test':
            dustbin = self.dustbin.repeat(desc1.size(0), 1, 1)
            desc1 = torch.cat([desc1, dustbin], -1)
            pad_mask_bin = torch.cat([data['mask1'].reshape(b, hw), torch.tensor(1, dtype=torch.bool, device=desc1.device).view(1, 1).repeat(desc1.size(0), 1)], -1)  if 'mask1' in data else None
            attn = self.assign(desc0, desc1, data['mask0'], pad_mask_bin, self.mode)

            data.update({
                'desc0': desc0,
                'desc1': desc1[:, :, :-1],
                'attn': attn
            })
            # predict coarse matches from conf_matrix
            self.get_coarse_matches(attn[:, :, :-1], data)
        else:
            desc0, desc1 = torch.stack(data['desc0_l'], -1), torch.stack(data['desc1_l'], -1)
            dustbin = self.dustbin.unsqueeze(-1).repeat(desc1.size(0), 1, 1, desc1.shape[-1])
            desc1 = torch.cat([desc1, dustbin], -2)
            pad_mask1_bin = torch.cat([data['mask1'].reshape(b, hw),
                                      torch.tensor(1, dtype=torch.bool, device=desc1.device).view(1, 1).repeat(
                                          desc1.size(0), 1)], -1)  if 'mask1' in data else None
            attn = self.assign(desc0, desc1, data['mask0'], pad_mask1_bin, self.mode)
            data.update({
                'desc0': desc0[..., -1],
                'desc1': desc1[:, :, :-1, -1],
                'attn': attn,
            })
            self.get_coarse_matches(attn[:, :, :-1, -1], data)

    @torch.no_grad()
    def get_coarse_matches(self, scores, data):
        _device = scores.device

        # Get the matches with score above "match_threshold".
        max0 = scores.max(2)
        matches = max0.indices
        conf = max0.values
        valid = conf > self.config['match_threshold']  # to the dustbin and conf bigger than threshold
        b_ids, i_ids = torch.where(valid)
        j_ids = matches[b_ids, i_ids]
        mconf = conf[b_ids, i_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            num_matches_train = int(data['num_candidates_max'] * self.config['match_coarse']['train_coarse_percent'])
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        b_ids_list, i_ids_list, j_ids_list, mconf_list = [], [], [], []
        for b in range(conf.shape[0]):
            b_mask = b_ids == b
            b_ids_, i_ids_, j_ids_, mconf_ = b_ids[b_mask], i_ids[b_mask], j_ids[b_mask], mconf[b_mask]
            b_ids_list.append(b_ids_)
            i_ids_list.append(i_ids_)
            j_ids_list.append(j_ids_)
            mconf_list.append(mconf_)
        b_ids = torch.cat(b_ids_list, 0)
        i_ids = torch.cat(i_ids_list, 0)
        j_ids = torch.cat(j_ids_list, 0)
        mconf = torch.cat(mconf_list, 0)
        # These matches select patches that feed into fine-level network
        data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids})

        # 4. Update with matches in original image resolution
        if 'scale0' in data:
            scale0 = data['scale0'][b_ids]
            scale1 = data['scale1'][b_ids]
            mkpts0_c = data['keypoints0'][b_ids, i_ids] * scale0
            mkpts1_c = data['keypoints1'][b_ids, j_ids] * scale1
        else:
            mkpts0_c = data['keypoints0'][b_ids, i_ids]
            mkpts1_c = data['keypoints1'][b_ids, j_ids]

        # These matches is the current prediction (for visualization)
        data.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0],
        })

    def fine_preprocess(self, data):
        stride = data['hw0_f'][0] // data['hw0_c'][0]
        W1 = self.config['fine_window_size']
        feat_c0, feat_c1 = data['desc0'], data['desc1']
        feat_f0, feat_f1 = data['feat_f0'], data['feat_f1']
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, W1 ** 2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, W1 ** 2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        scale_r2f = data['hw0_i'][0] // data['hw0_f'][0]
        kpts0 = data['keypoints0'][data['b_ids'], data['i_ids']] // scale_r2f
        kpts0_long = kpts0[:, :].round().long()

        # 1. unfold(crop) all local windows
        feat_f0_unfold = feat_f0[data['b_ids'], :, kpts0_long[:, 1], kpts0_long[:, 0]].unsqueeze(1)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W1, W1), stride=stride, padding=W1 // 2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W1 ** 2)  # [b, h_f/stride * w_f/stride, w*w, c]

        # 2. select only the predicted matches
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]  # [n, ww, cf]

        # option: use coarse-level rcm feature as context: concat and linear
        feat_c_win0 = self.down_proj(feat_c0[data['b_ids'], :, data['i_ids']])  # [n, c]
        feat_c_win1 = self.down_proj(feat_c1[data['b_ids'], :, data['j_ids']])  # [n, c]

        feat_f0_unfold = self.merge_feat(torch.cat([feat_f0_unfold, repeat(feat_c_win0, 'n c -> n one c', one=1)], -1)).transpose(1, 2)
        feat_f1_unfold = self.merge_feat(torch.cat([feat_f1_unfold, repeat(feat_c_win1, 'n c -> n ww c', ww=W1 ** 2)], -1)).transpose(1, 2)  # [2n, ww, cf]

        return feat_f0_unfold, feat_f1_unfold

    def fine_match(self, feat_f0, feat_f1, data):
        M, C, WW0 = feat_f0.shape
        _, _, WW1 = feat_f1.shape
        W0 = int(math.sqrt(WW0))
        W1 = int(math.sqrt(WW1))
        scale = data['hw1_i'][0] / data['hw1_f'][0]
        self.M, self.W0, self.WW0, self.W1, self.WW1, self.C, self.scale = M, W0, WW0, W1, WW1, C, scale
        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        feat_f0_picked = feat_f0[:, :, WW0//2]  # center
        sim_matrix = torch.einsum('mc,mcr->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W1, W1)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W1, W1, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized ** 2 * heatmap.view(-1, WW1, 1), dim=1) - coords_normalized ** 2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W1, self.WW1, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]  # all thing behind len(data['mconf']) od padding
        mkpts0_f = mkpts0_f  # / data['scale0'][data['b_ids'][:len(data['mconf'])]]
        mkpts1_f = mkpts1_f  # / data['scale1'][data['b_ids'][:len(data['mconf'])]]
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

    def forward(self, data, mode='test'):
        self.mode = mode
        # for training
        if 'mask0' in data:  # img_padding is True
            [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([data['mask0'], data['mask1']], dim=0).float(),
                                                   scale_factor=0.0625,
                                                   mode='nearest',
                                                   recompute_scale_factor=False).bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        self.extractor(data, mode)

        if mode != 'test':
            compute_supervision_coarse_scale(data, self.config)

        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:],
            'hw0_c': data['descriptors1'].shape[2:],
            'hw0_f': data['feat_f0'].shape[2:], 'hw1_f': data['feat_f1'].shape[2:]
        })

        self.coarse_match(data)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data)

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.gnn_fine(feat_f0_unfold, feat_f1_unfold)

        self.fine_match(feat_f0_unfold, feat_f1_unfold, data)

        switch_back4(data)
