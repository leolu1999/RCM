import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous


# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})

    if 'spv_scale' in data:
        gt = data['spv_scale'][:, 0] == 1
        est = data['est_scale'][:, 0] >= 1/2
        scale_errs = gt == est
        data.update({'scale_errs': scale_errs})


def compute_f1(data):
    correct_flag_l = []
    precision_l = []
    recall_l = []
    f1_score_l = []
    for b in range(len(data['conf_matrix_gt'])):
        b_mask = b == data['b_ids']
        correct_flag = data['conf_matrix_gt'][b, data['i_ids'][b_mask], data['j_ids'][b_mask]] == 1
        precision = correct_flag.sum() / len(correct_flag) if len(correct_flag) > 0 else torch.tensor(0., device=correct_flag.device)
        recall = correct_flag.sum() / len(data['spv_b_ids']) if len(data['spv_b_ids']) > 0 else torch.tensor(0., device=correct_flag.device)
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0., device=correct_flag.device)
        correct_flag_l.append(correct_flag)
        precision_l.append(precision)
        recall_l.append(recall)
        f1_score_l.append(f1_score)
    data.update({
        'correct_flag': correct_flag_l,
        'precision': precision_l,
        'recall': recall_l,
        'f1_score': f1_score_l,
                 })


def compute_symmetrical_epipolar_errors_sp(data):
    """
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0']
    pts1 = data['mkpts1']
    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(np.bool_))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


def estimate_lo_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    from .warppers import Camera, Pose
    import poselib
    camera0, camera1 = Camera.from_calibration_matrix(K0).float(), Camera.from_calibration_matrix(K1).float()
    pts0, pts1 = kpts0, kpts1

    M, info = poselib.estimate_relative_pose(
        pts0,
        pts1,
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {
            "max_epipolar_error": thresh,
        },
    )
    success = M is not None and ( ((M.t != [0., 0., 0.]).all()) or ((M.q != [1., 0., 0., 0.]).all()) )
    if success:
        M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t)) # .to(pts0)
        # print(M)
    else:
        M = Pose.from_4x4mat(torch.eye(4).numpy()) # .to(pts0)
        # print(M)

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": torch.tensor(info.pop("inliers")), # .to(pts0),
        **info,
    }
    return estimation


def compute_pose_errors2(data, config):
    """
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    RANSAC = config.TRAINER.POSE_ESTIMATION_METHOD
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        if config.RCM.EVAL_TIMES >= 1:
            bpts0, bpts1 = pts0[mask], pts1[mask]
            R_list, T_list, inliers_list = [], [], []
            for _ in range(5):
                shuffling = np.random.permutation(np.arange(len(bpts0)))
                if _ >= config.RCM.EVAL_TIMES:
                    continue
                bpts0 = bpts0[shuffling]
                bpts1 = bpts1[shuffling]

                if RANSAC == 'RANSAC':
                    ret = estimate_pose(bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf)
                    if ret is None:
                        R_list.append(np.inf)
                        T_list.append(np.inf)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        R, t, inliers = ret
                        t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
                        R_list.append(R_err)
                        T_list.append(t_err)
                        inliers_list.append(inliers)

                elif RANSAC == 'LO-RANSAC':
                    est = estimate_lo_pose(bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf)
                    if not est["success"]:
                        R_list.append(90)
                        T_list.append(90)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        M = est["M_0to1"]
                        inl = est["inliers"].numpy()
                        t_error, r_error = relative_pose_error(T_0to1[bs], M.R, M.t, ignore_gt_t_thr=0.0)
                        R_list.append(r_error)
                        T_list.append(t_error)
                        inliers_list.append(inl)
                else:
                    raise ValueError(f"Unknown RANSAC method: {RANSAC}")

            data['R_errs'].append(R_list)
            data['t_errs'].append(T_list)
            data['inliers'].append(inliers_list[0])


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs, num = [], []
    for thr in thresholds:
        prec_, num_ = [], []
        for errs in errors:
            correct_mask = errs < thr
            num_.append(len(correct_mask))
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        num.append(np.mean(num_) if len(num_) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}, {f'num_matches@{t:.0e}': na for t, na in zip(thresholds, num)}
    else:
        return precs


def epidist_prec_rec(errors, thresholds, max_matches, ret_dict=False):
    precs, rec, num = [], [], []
    for thr in thresholds:
        prec_, rec_, num_ = [], [], []
        for errs, mm in zip(errors, max_matches):
            correct_mask = errs < thr
            num_.append(len(correct_mask))
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
            rec_.append(np.sum(correct_mask)/mm if mm > 0 else 0)
        num.append(np.mean(num_) if len(num_) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
        rec.append(np.mean(rec_) if len(rec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)},  \
            {f'rec@{t:.0e}': rec for t, rec in zip(thresholds, rec)}, \
            {f'num_matches@{t:.0e}': na for t, na in zip(thresholds, num)},

    else:
        return precs


def epidist_prec_rec_max_f1(metrics, errors, thresholds, max_matches, ret_dict=False):
    precision = torch.stack(metrics['precision']).mean()
    recall = torch.stack(metrics['recall']).mean()
    f1 = torch.stack(metrics['f1_score']).mean()
    precs, rec, max, num = [], [], [], []
    for thr in thresholds:
        prec_, rec_, max_, num_ = [], [], [], []
        for errs, mm in zip(errors, max_matches):
            correct_mask = errs < thr
            num_.append(len(correct_mask))
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
            rec_.append(np.sum(correct_mask)/mm if mm > 0 else 0)
            max_.append(mm)
        num.append(np.mean(num_) if len(num_) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
        rec.append(np.mean(rec_) if len(rec_) > 0 else 0)
        max.append(np.mean(max_) if len(max_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)},  \
            {f'rec@{t:.0e}': re for t, re in zip(thresholds, rec)}, \
            {f'max_matches@{t:.0e}': ma for t, ma in zip(thresholds, max)}, \
            {f'num_matches@{t:.0e}': na for t, na in zip(thresholds, num)}, \
            {f'precision': precision}, \
            {f'recall': recall}, \
            {f'f1_score': f1}
    else:
        return precs


def epidist_prec_rec_max(errors, thresholds, max_matches, ret_dict=False):
    precs, rec, max, num = [], [], [], []
    for thr in thresholds:
        prec_, rec_, max_, num_ = [], [], [], []
        for errs, mm in zip(errors, max_matches):
            correct_mask = errs < thr
            num_.append(len(correct_mask))
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
            rec_.append(np.sum(correct_mask)/mm if mm > 0 else 0)
            max_.append(mm)
        num.append(np.mean(num_) if len(num_) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
        rec.append(np.mean(rec_) if len(rec_) > 0 else 0)
        max.append(np.mean(max_) if len(max_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)},  \
            {f'rec@{t:.0e}': re for t, re in zip(thresholds, rec)}, \
            {f'max_matches@{t:.0e}': ma for t, ma in zip(thresholds, max)}, \
            {f'num_matches@{t:.0e}': na for t, na in zip(thresholds, num)},
    else:
        return precs


def prec_rec_max_f1(metrics, ret_dict):
    prec = torch.stack(metrics['precision']).mean()
    rec = torch.stack(metrics['recall']).mean()
    f1 = torch.stack(metrics['f1_score']).mean()
    max = np.mean(metrics['max_matches']) if len(metrics['max_matches']) > 0 else 0
    if ret_dict:
        return {f'precision': prec},  \
            {f'recall': rec}, \
            {f'f1_score': f1}, \
            {f'max_matches': max},
    else:
        return prec


def aggregate_metrics_train_val(metrics, epi_err_thr=5e-4, config=None):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    if config.RCM.EVAL_TIMES >= 1:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0).reshape(-1, config.RCM.EVAL_TIMES)[unq_ids].reshape(-1)
    else:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]

    precs, rec, max, num, precision, recall, f1 = epidist_prec_rec_max_f1(metrics, np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, metrics['max_matches'], True)  # (prec@err_thr)

    # scale
    if 'scale_errs' in metrics:
        scale_acc = torch.stack(metrics['scale_errs']).float().mean()
        print("scale_acc: {}".format(scale_acc))
        return {**aucs, **precs, **rec, **max, **num, 'scale_acc': scale_acc, **precision, **recall, **f1}
    else:
        return {**aucs, **precs, **rec, **max, **num, **precision, **recall, **f1}


def aggregate_metrics_test(metrics, epi_err_thr=5e-4, config=None):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    # pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    if config.RCM.EVAL_TIMES >= 1:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0).reshape(-1, config.RCM.EVAL_TIMES)[unq_ids].reshape(-1)
    else:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]

    precs, num = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    return {**aucs, **precs, **num}


def aggregate_metrics_f1(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    precs, rec, f1, max = prec_rec_max_f1(metrics, True)  # (prec@err_thr)

    return {**aucs, **precs, **rec, **f1, **max}


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]

    precs, num = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    # scale
    if 'scale_errs' in metrics:
        scale_acc = torch.stack(metrics['scale_errs']).float().mean()
        print("scale_acc: {}".format(scale_acc))
        return {**aucs, **precs, **num, 'scale_acc': scale_acc}
    else:
        return {**aucs, **precs, **num}

