from pathlib import Path
import torch
from torch import nn
from loguru import logger
from kornia.utils import create_meshgrid
from einops import rearrange
import torch.nn.functional as F


def out_bound_mask(pt, img_size):
    h, w = img_size[:, 0].unsqueeze(1), img_size[:, 1].unsqueeze(1)
    return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)


class up_conv(nn.Module):
    def __init__(self, dim_in, dim_out, scale_factor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors_new(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - 1), (h*s - 1)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class Extractor(nn.Module):
    """
    Adopting SuperPoint Detector
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'resolution': [16, 2],
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_avg_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_avg_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool_avg_8 = nn.AvgPool2d(kernel_size=8, stride=8)

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        for p in self.parameters():
            p.requires_grad = False

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.config['descriptor_dim'], kernel_size=1, stride=1, padding=0)

        path = '/home/leo/projects/LoFTR/superglue_models/weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))
        logger.info(f"Load \'{path}\' as pretrained SuperPoint checkpoint")

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        self.conv5a = nn.Conv2d(self.config['descriptor_dim'], 256, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.up1 = up_conv(256, 128)  # 1/16 -> 1/8

        self.conv6a = nn.Conv2d(2 * 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.up2 = up_conv(128, 128)  # 1/8 -> 1/4

        self.conv7a = nn.Conv2d(2 * 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv(128, 64)  # 1/4 -> 1/2

        self.conv8a = nn.Conv2d(2 * 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lin0 = nn.Conv1d(256, 256, 1)
        self.lin1 = nn.Conv1d(128, 128, 1)
        self.lin2 = nn.Conv1d(64, 64, 1)
        self.merge_c = nn.Conv1d(256 + 256 + 128 + 64, 256, 1)

        self.cpe_f = nn.Conv2d(64, 64, 3, 1, 1, groups=64)

        self.convSa = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.convSb = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool_avg = nn.AdaptiveAvgPool2d(output_size=(20, 20))
        self.scale = nn.Sequential(
            nn.Conv2d(400, 64, kernel_size=3, stride=1, padding=1),  # 20
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 10
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # 5
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0),
        )

    def add_random_point(self, keypoints, scores):
        for i, (k, s) in enumerate(zip(keypoints, scores)):
            if len(k) < self.config['max_keypoints']:
                to_add_points = self.config['max_keypoints'] - len(k)
                random_keypoints = torch.stack(
                    [torch.randint(0, self.w * 8, (to_add_points,), dtype=torch.float32, device=k.device),
                     torch.randint(0, self.h * 8, (to_add_points,), dtype=torch.float32, device=k.device)], 1)
                keypoints[i] = torch.cat([keypoints[i], random_keypoints], dim=0)
                scores[i] = torch.cat(
                    [scores[i], torch.zeros(to_add_points, dtype=torch.float32, device=s.device) * 0.1], dim=0)

    def sparse_branch(self, data):
        # Sparse: Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in data['scores0']]
        scores = [s[tuple(k.t())] for s, k in zip(data['scores0'], keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], self.h * 8, self.w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.max_keypoints >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.max_keypoints)
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        scores = list(scores)
        if self.mode == 'train':
            self.add_random_point(keypoints, scores)

        desc1 = torch.stack([sample_descriptors_new(k[None], d[None], 2)[0] for k, d in zip(keypoints, data['d1_0'])], 0)  # 1/2
        desc2 = torch.stack([sample_descriptors_new(k[None], d[None], 4)[0] for k, d in zip(keypoints, data['d2_0'])], 0)  # 1/4
        desc3 = torch.stack([sample_descriptors_new(k[None], d[None], 8)[0] for k, d in zip(keypoints, data['cDa_0'])], 0)  # 1/8 (oringnal)
        desc4 = torch.stack([sample_descriptors_new(k[None], d[None], 16)[0] for k, d in zip(keypoints, data['d4_0'])], 0)  # 1/16

        desc4 = self.lin0(desc4)
        desc2 = self.lin1(desc2)
        desc1 = self.lin2(desc1)
        descriptors = self.merge_c(torch.cat([desc1, desc2, desc3, desc4], 1))  # 2,4,8,16

        keypoints = torch.stack(keypoints, 0)
        mask0 = ~out_bound_mask(keypoints, data['prepad_size0']) if 'prepad_size0' in data else None

        data.update({
            'keypoints0': keypoints,
            'scores0': torch.stack(scores, 0),
            'descriptors0': descriptors,
            'mask0': mask0
        })

    def dense_branch(self, data):
        device = data['image0'].device
        B, _, H, W = data['image0'].shape
        scale = self.config['resolution'][0]
        h_c, w_c = H//scale, W//scale
        keypoints = [rearrange((create_meshgrid(h_c, w_c, False, device) * scale).squeeze(0), 'h w t->(h w) t')] * B  # kpt_xy
        scores = [s[tuple(torch.flip(k.long(), [1]).t())] for s, k in zip(data['scores1'], keypoints)]  # take score using kpt_hw
        scores = list(scores)

        desc1 = torch.nn.functional.normalize(self.pool_avg_8(data['d1_1']), p=2, dim=1).reshape(B, -1, h_c*w_c)
        desc2 = torch.nn.functional.normalize(self.pool_avg_4(data['d2_1']), p=2, dim=1).reshape(B, -1, h_c*w_c)
        desc3 = torch.nn.functional.normalize(self.pool_avg_2(data['cDa_1']), p=2, dim=1).reshape(B, -1, h_c*w_c)
        desc4 = torch.nn.functional.normalize(data['d4_1'], p=2, dim=1).reshape(B, -1, h_c*w_c)

        # Multi-level Feature
        desc4 = self.lin0(desc4)
        desc2 = self.lin1(desc2)
        desc1 = self.lin2(desc1)
        descriptors = self.merge_c(torch.cat([desc1, desc2, desc3, desc4], 1)).reshape(B, -1, h_c, w_c)  # 2,4,8,16

        data.update({
            'keypoints1': torch.stack(keypoints, 0),
            'scores1': torch.stack(scores, 0),
            'descriptors1': descriptors,
        })

    def switch(self, data, switch):
        if 'mask0' in data:
            keys0 = ['image0', 'mask0', 'depth0', 'K0', 'scale0', 'T_0to1', 'feat_f0', 'scores0', 'd1_0', 'd2_0', 'cDa_0', 'd4_0', 'prepad_size0']
            keys1 = ['image1', 'mask1', 'depth1', 'K1', 'scale1', 'T_1to0', 'feat_f1', 'scores1', 'd1_1', 'd2_1', 'cDa_1', 'd4_1', 'prepad_size1']
        else:
            keys0 = ['image0', 'depth0', 'K0', 'T_0to1', 'feat_f0', 'scores0', 'd1_0', 'd2_0', 'cDa_0', 'd4_0']
            keys1 = ['image1', 'depth1', 'K1', 'T_1to0', 'feat_f1', 'scores1', 'd1_1', 'd2_1', 'cDa_1', 'd4_1']
        # switch
        for (key0, key1) in zip(keys0, keys1):
            item0_list, item1_list = [], []
            for b in range(len(data[key0])):
                if switch[b]:
                    item0_list.append(data[key1][b])
                    item1_list.append(data[key0][b])
                else:
                    item0_list.append(data[key0][b])
                    item1_list.append(data[key1][b])
            data.update({
                key0: torch.stack(item0_list, 0),
                key1: torch.stack(item1_list, 0)
            })

    def forward(self, data, mode='test', max_kpts=None):
        """ Compute keypoints, scores, descriptors for image """
        self.max_keypoints = max_kpts if max_kpts is not None else self.config['max_keypoints']
        self.mode = mode
        B = data['image0'].shape[0]
        images = torch.cat([data['image0'], data['image1']], 0)
        # Shared Encoder
        x0 = self.relu(self.conv1a(images))
        x0 = self.relu(self.conv1b(x0))
        x1 = self.pool(x0)  # 1/2
        x1 = self.relu(self.conv2a(x1))
        x1 = self.relu(self.conv2b(x1))
        x2 = self.pool(x1)  # 1/4
        x2 = self.relu(self.conv3a(x2))
        x2 = self.relu(self.conv3b(x2))
        x3 = self.pool(x2)  # 1/8
        x3 = self.relu(self.conv4a(x3))
        x3 = self.relu(self.conv4b(x3))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x3))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        self.h, self.w = h, w
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x3))
        cDa = self.convDb(cDa)

        d4 = self.pool(cDa)  # 1/16
        d4 = self.relu(self.conv5a(d4))
        d4 = self.relu(self.conv5b(d4))

        d3 = self.up1(d4)  # 1/8
        d3 = self.relu(self.conv6a(torch.cat([x3, d3], 1)))
        d3 = self.relu(self.conv6b(d3))

        d2 = self.up2(d3)  # 1/4
        d2 = self.relu(self.conv7a(torch.cat([x2, d2], 1)))
        d2 = self.relu(self.conv7b(d2))

        d1 = self.up3(d2)  # 1/2
        d1 = self.relu(self.conv8a(torch.cat([x1, d1], 1)))
        d1 = self.conv8b(d1)

        feat_f = d1 + self.cpe_f(d1)
        feat_f0, feat_f1 = feat_f.split(B)

        (scores0, scores1), (cDa_0, cDa_1) = scores.split(B), cDa.split(B)
        (d1_0, d1_1), (d2_0, d2_1), (d4_0, d4_1) = d1.split(B), d2.split(B), d4.split(B)

        data.update({
            'scores0': scores0,
            'scores1': scores1,
            'd1_0': d1_0,
            'd1_1': d1_1,
            'd2_0': d2_0,
            'd2_1': d2_1,
            'cDa_0': cDa_0,
            'cDa_1': cDa_1,
            'd4_0': d4_0,
            'd4_1': d4_1,
            'feat_f0': feat_f0,
            'feat_f1': feat_f1
        })

        scale_feat = self.relu(self.convSa(cDa.detach()))
        scale_feat = self.convSb(scale_feat)
        scale_feat = self.pool_avg(scale_feat)
        scale_feat0, scale_feat1 = scale_feat.split(B)
        corr = torch.einsum('b c h w,b c H W->b h w H W', scale_feat0, scale_feat1)  # [b, 30, 30, 30, 30]
        corr = rearrange(corr, 'b h w H W -> b (H W) h w')  # [b, 30, 30, 900]
        scale = self.scale(corr).squeeze(3).squeeze(2)  # [b, 2]
        scale = F.softmax(scale, dim=1)

        switch = scale[:, 0] > 1 / 2

        data.update({'switch': switch,
                     'est_scale': scale})

        self.switch(data, switch)

        self.sparse_branch(data)

        self.dense_branch(data)
