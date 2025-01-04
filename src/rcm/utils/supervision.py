from math import log
from loguru import logger
import matplotlib.pyplot as plt
import torch
from einops import repeat, rearrange
from kornia.utils import create_meshgrid
import numpy as np
from .geometry import warp_kpts
from .common import make_matching_plot, make_warp_plot, plot_image_pair_scale
import cv2
import time
from src.utils.plotting import kp_color
##############  ↓  Coarse-Level supervision  ↓  ##############

INF = 1E9


def get_scale_dist(keypoints1, keypoints2, sample_num, clamp_ratio=3.):
    sample_num = sample_num if keypoints1.shape[0] > sample_num else keypoints1.shape[0]
    random_idx = torch.randint(0, keypoints1.shape[0], (sample_num,))
    mkpts0, mkpts1 = keypoints1[random_idx], keypoints2[random_idx]
    dist0, dist1 = torch_cdist(mkpts0, mkpts0), torch_cdist(mkpts1, mkpts1)
    dist0, dist1 = torch.triu(dist0), torch.triu(dist1)
    dist_mean0 = dist0.sum() / ((sample_num ** 2 - sample_num) / 2)
    dist_mean1 = dist1.sum() / ((sample_num ** 2 - sample_num) / 2)
    scale = dist_mean0 / (dist_mean1 + 1e-4)
    scale = torch.clamp(scale, 1/clamp_ratio, clamp_ratio)
    return scale


def torch_cdist(keypoints1, keypoints2):
    diff = (keypoints1[:, None, :] - keypoints2[None, :, :]) ** 2
    summed = diff.sum(-1)
    distance = torch.sqrt(summed)
    return distance


def torch_cdist_b(keypoints1, keypoints2):
    diff = (keypoints1[:, :, None, :] - keypoints2[:, None, :, :]) ** 2
    summed = diff.sum(-1)
    distance = torch.sqrt(summed)
    return distance


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


def groundtruth_viz(data, batch_idx, path, dpi):
    save_path = path + str(batch_idx)
    b_ids, i_ids, j_ids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']
    mkpts0 = data['keypoints0'][b_ids, i_ids].cpu().numpy()
    mkpts1 = data['keypoints1'][b_ids, j_ids].cpu().numpy()
    image0, image1 = (data['image0']*255.).squeeze().cpu().numpy(), (data['image1']*255.).squeeze().cpu().numpy()
    kpts0, kpts1 = data['keypoints0'].squeeze().cpu().numpy(), data['keypoints1'].squeeze().cpu().numpy()
    color = np.zeros((mkpts1.shape[0], 4))
    color[:, 2] = 1
    color[:, 3] = 1
    make_matching_plot(
        image0, image1,
        kpts0, kpts1, mkpts0, mkpts1, color, save_path, ['#GT matches: {}'.format(len(b_ids))], True, False, False, 'Matches', [], dpi=dpi)


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """

    # 1. make kpt mask
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    m = data['keypoints0'].shape[1]
    kpt0_r = data['keypoints0']
    grid_pt1_r = data['keypoints1']
    scale0 = data['scale0'][:, None] if 'scale0' in data else 1
    scale1 = data['scale1'][:, None] if 'scale0' in data else 1
    kpt0_i = scale0 * kpt0_r

    # 2. warp grids w/o masking
    # create kpts in meshgrid and resize them to image resolution
    scale = config['LOFTR']['RESOLUTION'][0]
    w1, h1 = W1 // scale, H1 // scale
    n = w1 * h1

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    # if 'mask0' in data:
    #     grid_pt1_r = mask_pts_at_padded_regions(grid_pt1_r, data['mask1'])

    grid_pt1_i = scale1 * grid_pt1_r

    valid_mask, w_kpt0to1_i = warp_kpts(kpt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    w_kpt0to1_r = w_kpt0to1_i // scale1

    # projection error
    pro_error = torch_cdist_b(w_kpt0to1_r, grid_pt1_r)
    # pro_error[~valid_mask] = INF
    if 'mask0' in data:
        pro_error[~data['mask0'].flatten(1,2)] = INF
        pad_mask = data['mask1'].flatten(1,2).unsqueeze(1).repeat(1, pro_error.size(1), 1)
        pro_error[~pad_mask] = INF
    # find corr index
    min_index = torch.argmin(pro_error, dim=2)

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    mask_inlier = torch.take_along_dim(pro_error, indices=min_index[:, :, None], dim=-1).squeeze(2) < (scale/2)*(2**0.5)
    correct_0to1 = mask_inlier  # ignore the top-left corner
    correct_0to1[out_bound_mask(w_kpt0to1_r, H1, W1)] = False

    # 4. construct a gt match_matrix
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = min_index[b_ids, i_ids]

    conf_matrix_gt = torch.zeros(N, m, n, device=device)
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1

    data.update({'conf_matrix_gt': conf_matrix_gt,})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids,
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'num_candidates_max': b_ids.shape[0],
        'spv_w_pt0_i': w_kpt0to1_i,
        'spv_pt1_i': grid_pt1_i
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


def compute_supervision_coarse_scale(data, config):
    # 1. make kpt mask
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    grid_pt0_r = data['keypoints0']
    grid_pt1_r = data['keypoints1']
    scale = config['resolution'][0]
    scale0 = data['scale0'][:, None] if 'scale0' in data else 1
    scale1 = data['scale1'][:, None] if 'scale0' in data else 1
    grid_pt0_i = scale0 * grid_pt0_r
    grid_pt1_i = scale1 * grid_pt1_r
    m, n = data['keypoints0'].shape[1], data['keypoints1'].shape[1]

    # 2. warp grids w/o masking
    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    valid_mask, w_pt0to1_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    _, w_pt1to0_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0to1_r = w_pt0to1_i / scale1
    w_pt1to0_r = w_pt1to0_i / scale0

    # reprojection error
    dis_mat0 = torch_cdist_b(grid_pt0_r, w_pt1to0_r)
    dis_mat1 = torch_cdist_b(grid_pt1_r, w_pt0to1_r)
    repro_error = torch.maximum(dis_mat0, dis_mat1.transpose(1, 2))  # n1*n2

    # projection error
    pro_error = dis_mat1.transpose(1, 2)

    pro_error[~valid_mask] = INF
    if 'mask0' in data:
        pro_error[~data['mask0']] = INF
        pad_mask = data['mask1'].flatten(1, 2).unsqueeze(1).repeat(1, pro_error.size(1), 1)
        pro_error[~pad_mask] = INF
    # find corr index
    min_index = torch.argmin(pro_error, dim=2)

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    mask_inlier = torch.take_along_dim(pro_error, indices=min_index[:, :, None], dim=-1).squeeze(2) < (scale / 2) * (2 ** 0.5)
    correct_0to1 = mask_inlier  # ignore the top-left corner
    correct_0to1[out_bound_mask(w_pt0to1_r, H1, W1)] = False

    # 4. construct a gt match_matrix
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = min_index[b_ids, i_ids]

    conf_matrix_gt = torch.zeros(N, m, n, device=device)
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1

    data.update({'conf_matrix_gt': conf_matrix_gt, })

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids,
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'num_candidates_max': b_ids.shape[0],
        'spv_w_pt0_i': w_pt0to1_i,
        'spv_pt1_i': grid_pt1_i
    })

    # b = 2
    # d = mask_inlier[b].sum()
    # b_mask = b_ids == b
    # mkpts0 = data['keypoints0'][b_ids[b_mask], i_ids[b_mask]].cpu().numpy()
    # mkpts1 = data['keypoints1'][b_ids[b_mask], j_ids[b_mask]].cpu().numpy()
    # image0, image1 = (data['image0']*255.)[b, 0].cpu().numpy(), (data['image1']*255.)[b, 0].cpu().numpy()
    # kpts0, kpts1 = data['keypoints0'][b].cpu().numpy(), grid_pt1_r[b].cpu().numpy()
    # color = np.zeros((mkpts1.shape[0], 4))
    # color[:, 1] = 1
    # color[:, 3] = 0.3
    # match_viz_path = '/home/leo/projects/LoFTR/debug_viz/rcm_gt2.png'
    # make_matching_plot(
    #     image0, image1,
    #     mkpts0, mkpts1, color, match_viz_path, kpts0, kpts1, ['matches: {}'.format(d)], True, False, False, 'Matches', [], dpi=300)
    #
    # kpts0, kpts1 = (grid_pt1_i/scale1)[b].cpu().numpy(), (w_pt0to1_i/scale1)[b, data['mask0'][b]].cpu().numpy()
    # kpts = (grid_pt0_i/scale0)[b, data['mask0'][b]].cpu().numpy()
    # color = kp_color(kpts[:, 1], kpts[:, 0], image0.shape[:2]) / 255.
    # match_viz_path = '/home/leo/projects/LoFTR/debug_viz/rcm_gt_warp.png'
    # make_warp_plot(image0, image1, kpts, kpts0, kpts1, match_viz_path, color)

    # find corr index
    nn_sort1 = torch.argmin(repro_error, dim=2)
    nn_sort2 = torch.argmin(repro_error, dim=1)
    mask_mutual = torch.gather(nn_sort2, 1, nn_sort1) == torch.arange(grid_pt0_r.shape[1], device=device).view(1,
                                                                                                               -1).repeat(
        N, 1)
    mask_inlier = torch.take_along_dim(repro_error, indices=nn_sort1[:, :, None], dim=-1).squeeze(2) < 5
    correct_0to1 = mask_mutual & mask_inlier

    # 4. construct a gt conf_matrix
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nn_sort1[b_ids, i_ids]

    scale_gt = []
    for b in range(grid_pt0_r.shape[0]):
        b_mask = b_ids == b
        if b_mask.sum() > 2:
            if data['switch'][b]:
                mkpts1 = grid_pt0_r[b_ids[b_mask], i_ids[b_mask]]
                mkpts0 = grid_pt1_r[b_ids[b_mask], j_ids[b_mask]]
            else:
                mkpts0 = grid_pt0_r[b_ids[b_mask], i_ids[b_mask]]
                mkpts1 = grid_pt1_r[b_ids[b_mask], j_ids[b_mask]]
            scale_gt_dist = get_scale_dist(mkpts0, mkpts1, sample_num=500)
        else:
            scale_gt_dist = torch.ones((), device=b_ids.device)
        scale_gt.append(scale_gt_dist)
    scale_gt = torch.stack(scale_gt, 0).unsqueeze(1).repeat(1, 2)
    scale_gt[:, 0] = torch.where(scale_gt[:, 0] < 1, 1, 0)
    scale_gt[:, 1] = 1 - scale_gt[:, 0]
    data.update({
        'spv_scale': scale_gt
    })


##############  ↓  Fine-Level supervision  ↓  ##############
@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['RCM']['RESOLUTION'][1]

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
    radius = config['RCM']['FINE_WINDOW_SIZE'] // 2

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    # scale = scale / scale_gt[b_ids] * data['scale1'][b_ids] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError

