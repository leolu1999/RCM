import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #
def make_gt_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='k', s=4)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='k', s=4)
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    else:
        return fig


def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    else:
        return fig


def make_kpt_figure(data, b_id=0, dpi=75, path=None):
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['keypoints0'][b_id].cpu().numpy()
    kpts1 = data['keypoints1'][b_id].cpu().numpy()
    # draw image pair
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=1)

    axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=1)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic', switch=None, path=None):
    switch = switch[b_id] if switch is not None else False
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    if switch:
        img0 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        kpts0 = data['mkpts1_f'][b_mask].cpu().numpy()
        kpts1 = data['mkpts0_f'][b_mask].cpu().numpy()

        # for megadepth, we visualize matches on the resized image
        if 'scale0' in data:
            kpts0 = kpts0 / data['scale1'][b_id].cpu().numpy()[[1, 0]]
            kpts1 = kpts1 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
    else:
        img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
        kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

        # for megadepth, we visualize matches on the resized image
        if 'scale0' in data:
            kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
            kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path)
    return figure


def _make_evaluation_figure_coarse(data, b_id, alpha='dynamic', switch=None, path=None, dpi=75):
    switch = switch[b_id] if switch is not None else False
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    if switch:
        img0 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        kpts0 = data['mkpts1_f'][b_mask].cpu().numpy()
        kpts1 = data['mkpts0_f'][b_mask].cpu().numpy()

        # for megadepth, we visualize matches on the resized image
        if 'scale0' in data:
            kpts0 = kpts0 / data['scale1'][b_id].cpu().numpy()[[1, 0]]
            kpts1 = kpts1 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
    else:
        img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
        kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

        # for megadepth, we visualize matches on the resized image
        if 'scale0' in data:
            kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
            kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    correct_mask = data['correct_flag'][b_id].cpu().numpy()
    precision = data['precision'][b_id].cpu().numpy()
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = data['recall'][b_id].cpu().numpy()
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    errs = np.stack([0 if acc else 1 for acc in correct_mask]) if len(correct_mask)>0 else np.empty([])
    color = error_colormap(errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path, dpi=dpi)
    return figure


def _make_evaluation_figure_test(data, b_id, alpha='dynamic', switch=None, path=None, dpi=150):
    switch = switch[b_id] if switch is not None else False
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    # conf_thr = 1e-5
    if switch:
        img0 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        kpts0 = data['mkpts1_f'][b_mask].cpu().numpy()
        kpts1 = data['mkpts0_f'][b_mask].cpu().numpy()

        # for megadepth, we visualize matches on the resized image
        if 'scale0' in data:
            kpts0 = kpts0 / data['scale1'][b_id].cpu().numpy()[[1, 0]]
            kpts1 = kpts1 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
    else:
        img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
        kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

        # for megadepth, we visualize matches on the resized image
        if 'scale0' in data:
            kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
            kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    # img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    # img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    # kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    # kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    #
    # # for megadepth, we visualize matches on the resized image
    # if 'scale0' in data:
    #     kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
    #     kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    R_errs = data['R_errs'][b_id]
    t_errs = data['t_errs'][b_id]

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'R_errs: {R_errs:.1f}',
        f't_errs: {t_errs:.1f}'
    ]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path, dpi=dpi)
    return figure


def _make_gt_figure(data, switch=None, path=None, dpi=75):
    b_id = 0
    switch = switch[b_id] if switch is not None else False
    if switch:
        b_ids, i_ids, j_ids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']
        img0 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        mkpts0 = data['keypoints0'][b_ids, i_ids].cpu().numpy()
        mkpts1 = data['keypoints1'][b_ids, j_ids].cpu().numpy()
        kpts0 = data['keypoints0'].squeeze().cpu().numpy()
        kpts1 = data['keypoints1'][:, data['mask0'].squeeze().view(-1)].squeeze().cpu().numpy()
    else:
        b_ids, i_ids, j_ids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']
        img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
        mkpts0 = data['keypoints0'][b_ids, i_ids].cpu().numpy()
        mkpts1 = data['keypoints1'][b_ids, j_ids].cpu().numpy()
        kpts0 = data['keypoints0'].squeeze().cpu().numpy()
        # kpts0 = data['keypoints0'][:, data['mask0'].squeeze().view(-1)].squeeze().cpu().numpy()
        kpts1 = data['keypoints1'][:, data['mask1'].squeeze().view(-1)].squeeze().cpu().numpy()
    text = [
        f'#GT Matches {len(mkpts0)}',
    ]
    color = np.zeros((mkpts1.shape[0], 4))
    color[:, :] = 1
    # make the figure
    figure = make_gt_figure(img0, img1, mkpts0, mkpts1,
                                  color, kpts0=kpts0, kpts1=kpts1, text=text, path=path, dpi=dpi)
    return figure


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def coord_trans(u, v):
    rad = np.sqrt(np.square(u) + np.square(v))
    u /= (rad+1e-3)
    v /= (rad+1e-3)
    return u, v

def kp_color(u, v, resolution):
    h, w = resolution
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    xx, yy = coord_trans(xx, yy)
    vis = flow_uv_to_colors(xx, yy)

    color = vis[v.astype(np.int32), u.astype(np.int32)]
    return color

def draw_kp(img, kps, colors):
    for i, kp in enumerate(kps):
        img = cv2.circle(img, (int(kp[1]), int(kp[0])), 1, colors[i].tolist(), -1)
    return img


def vis_matches(image0, image1, kp0, kp1):
    lh, lw = image0.shape[:2]
    rh, rw = image1.shape[:2]
    mask1 = np.logical_and.reduce(np.array((kp0[:,1]>=0, kp0[:,1]<lw, kp0[:,0]>=0, kp0[:,0]<lh)))
    mask2 = np.logical_and.reduce(np.array((kp1[:,1]>=0, kp1[:,1]<rw, kp1[:,0]>=0, kp1[:,0]<rh)))

    mask = np.logical_and.reduce(np.array((mask1, mask2)))
    kp0 = kp0[mask]
    kp1 = kp1[mask]

    color = kp_color(kp0[:,1], kp0[:,0], (lh, lw))

    image0 = draw_kp(image0, kp0, color)
    image1 = draw_kp(image1, kp1, color)

    pad_width = 5
    zero_image = np.zeros([lh, pad_width, 3])
    vis = np.concatenate([image0, zero_image, image1], axis=1)

    # kp1[:,1] += lw + pad_width
    # vis = draw_matches(vis, kp0, kp1)

    return vis


# def _make_evaluation_figure_wheel(data, b_id, alpha='dynamic', switch=None, path=None, dpi=150):
#     switch = switch[b_id] if switch is not None else False
#     b_mask = data['m_bids'] == b_id
#     if switch:
#         img0 = (data['imagec_1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
#         img1 = (data['imagec_0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
#         kpts0 = data['mkpts1_f'][b_mask]
#         kpts1 = data['mkpts0_f'][b_mask]
#         # for megadepth, we visualize matches on the resized image
#         if 'scale0' in data:
#             kpts0 = kpts0 / data['scale1'][b_id][[1, 0]]
#             kpts1 = kpts1 / data['scale0'][b_id][[1, 0]]
#     else:
#         img0 = (data['imagec_0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
#         img1 = (data['imagec_1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
#         kpts0 = data['mkpts0_f'][b_mask]
#         kpts1 = data['mkpts1_f'][b_mask]
#         # for megadepth, we visualize matches on the resized image
#         if 'scale0' in data:
#             kpts0 = kpts0 / data['scale0'][b_id][[1, 0]]
#             kpts1 = kpts1 / data['scale1'][b_id][[1, 0]]
#
#     # make the figure
#     kpts_wh_0 = torch.flip(kpts0, [1]).cpu().numpy()
#     kpts_wh_1 = torch.flip(kpts1, [1]).cpu().numpy()
#     figure = vis_matches(img0, img1, kpts_wh_0, kpts_wh_1)
#     cv2.imwrite(path+'.png', figure)
#     return figure


def _make_evaluation_figure_wheel(data, b_id, alpha='dynamic', switch=None, path=None, dpi=150):
    b_mask = data['m_bids'] == b_id

    img0 = (data['imagec_0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['imagec_1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask]
    kpts1 = data['mkpts1_f'][b_mask]
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id][[0, 1]]
        kpts1 = kpts1 / data['scale1'][b_id][[0, 1]]

    # make the figure
    kpts_wh_0 = torch.flip(kpts0, [1]).cpu().numpy()
    kpts_wh_1 = torch.flip(kpts1, [1]).cpu().numpy()
    figure = vis_matches(img0, img1, kpts_wh_0, kpts_wh_1)
    cv2.imwrite(path+'.png', figure)
    return figure


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode='evaluation', switch=None, path=None, dpi=150):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'test', 'wheel', 'coarse']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, switch=switch)
        elif mode == 'test':
            fig = _make_evaluation_figure_test(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, switch=switch, path=path, dpi=dpi)
        elif mode == 'coarse':
            fig = _make_evaluation_figure_coarse(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, switch=switch, path=path, dpi=dpi)
        elif mode == 'wheel':
            fig = _make_evaluation_figure_wheel(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA, switch=switch, path=path, dpi=dpi)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
