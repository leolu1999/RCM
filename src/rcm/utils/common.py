# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import glob
import math
import os
import random
import re
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import Thread

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as torch_dist
import matplotlib.cm as cm

matplotlib.use('Agg')

superpoint_url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superpoint_v1.pth"
superglue_indoor_url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_indoor.pth"
superglue_outdoor_url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_outdoor.pth"
superglue_cocohomo_url = "https://github.com/gouthamk1998/files/releases/download/1.0/release_model.pt"
coco_test_images_url = "https://github.com/gouthamk1998/files/releases/download/1.0/coco_test_images.zip"
indoor_test_images_url = "https://github.com/gouthamk1998/files/releases/download/1.0/indoor_test_images.zip"
outdoor_test_imags_url = "https://github.com/gouthamk1998/files/releases/download/1.0/outdoor_test_images.zip"
weights_mapping = {
    'superpoint': Path(__file__).parent.parent / 'models/weights/superpoint_v1.pth',
    'indoor': Path(__file__).parent.parent / 'models/weights/superglue_indoor.pth',
    'outdoor': Path(__file__).parent.parent / 'models/weights/superglue_outdoor.pth',
    'coco_homo': Path(__file__).parent.parent / 'models/weights/superglue_cocohomo.pt'
}

test_images_mapping = {
    'coco_test_images': coco_test_images_url,
    'indoor_test_images': indoor_test_images_url,
    'outdoor_test_images': outdoor_test_imags_url
}


def disk_res(features_, device):
    disk_result = {
        'keypoints': [],
        'scores': [],
        'descriptors': [],
    }
    for features in features_.flat:
        features = features.to(device)
        keypoints = features.kp
        descriptors = features.desc.T
        scores = features.kp_logp
        disk_result['keypoints'].append(keypoints)
        disk_result['descriptors'].append(descriptors)
        disk_result['scores'].append(scores)
    return disk_result


def download_test_images():
    for i, k in test_images_mapping.items():
        zip_path = Path(__file__).parent.parent / ('assets/' + i + '.zip')
        directory_path = Path(__file__).parent.parent / ('assets/' + i)
        if not directory_path.exists():
            print("Downloading and unzipping {}...".format(i))
            os.system("curl -L {} -o {}".format(k, str(zip_path)))
            os.system("unzip {} -d {}".format(str(zip_path), str(directory_path.parent)))
            os.remove(str(zip_path))


def download_base_files():
    directory = Path(__file__).parent.parent / 'models/weights'
    if not directory.exists():
        os.makedirs(str(directory))
    superpoint_path = weights_mapping['superpoint']
    indoor_path = weights_mapping['indoor']
    outdoor_path = weights_mapping['outdoor']
    coco_homo_path = weights_mapping['coco_homo']
    command = "curl -L {} -o {}"
    if not superpoint_path.exists():
        print("Downloading superpoint model...")
        os.system(command.format(superpoint_url, str(superpoint_path)))
    if not indoor_path.exists():
        print("Downloading superglue indoor model...")
        os.system(command.format(superglue_indoor_url, str(indoor_path)))
    if not outdoor_path.exists():
        print("Downloading superglue outdoor model...")
        os.system(command.format(superglue_outdoor_url, str(outdoor_path)))
    if not coco_homo_path.exists():
        print("Downloading coco homography model...")
        os.system(command.format(superglue_cocohomo_url, str(coco_homo_path)))


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def clean_checkpoint(ckpt):
    new_ckpt = {}
    for i, k in ckpt.items():
        if i[0:6] == "module":
            new_ckpt[i[7:]] = k
        else:
            new_ckpt[i] = k
    return new_ckpt


def get_world_size():
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size()


def reduce_tensor(inp, avg=True):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    if not avg: return reduced_inp
    return reduced_inp / world_size


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1. / total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                # Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            # print('IPCAMERA THREAD got frame {}'.format(self._ip_index))

    def cleanup(self):
        self._ip_running = False


# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def read_image_with_homography(path, homo_matrix, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (w, h))
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
        warped_image = cv2.resize(warped_image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
        warped_image = cv2.resize(warped_image, (w_new, h_new)).astype('float32')
    if rotation != 0:
        image = np.rot90(image, k=rotation)
        warped_image = np.rot90(warped_image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    inp = frame2tensor(image, device)
    warped_inp = frame2tensor(warped_image, device)
    scaled_homo = scale_homography(homo_matrix, h, w, h_new, w_new).astype(np.float32)
    return image, warped_image, inp, warped_inp, scales, scaled_homo


def read_image_with_homography_and_sift(path, homo_matrix, resize, rotation, max_keypoints):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_keypoints)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (w, h))
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image, (w_new, h_new))
    warped_image = cv2.resize(warped_image, (w_new, h_new))
    if rotation != 0:
        image = np.rot90(image, k=rotation)
        warped_image = np.rot90(warped_image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    scaled_homo = scale_homography(homo_matrix, h, w, h_new, w_new).astype(np.float32)

    # sift
    kpt0, desc_np0 = sift.detectAndCompute(image, None)
    kpt1, desc_np1 = sift.detectAndCompute(warped_image, None)
    kpt_num0 = min(max_keypoints, len(kpt0))
    kpt_num1 = min(max_keypoints, len(kpt1))
    # skip this image pair if no keypoints detected in image
    if kpt_num0 == 0 or kpt_num1 == 0:
        return None
    # square-root
    eps = 1e-10
    desc_np0 /= (desc_np0.sum(axis=1, keepdims=True) + eps)
    desc_np1 /= (desc_np1.sum(axis=1, keepdims=True) + eps)
    desc_np0, desc_np1 = np.sqrt(desc_np0), np.sqrt(desc_np1)
    kp_np0 = np.array([(kp.pt[0], kp.pt[1]) for kp in kpt0]).astype(np.float32)
    kp_np1 = np.array([(kp.pt[0], kp.pt[1]) for kp in kpt1]).astype(np.float32)
    scores_np0 = np.array([kp.response for kp in kpt0]).astype(np.float32)
    scores_np1 = np.array([kp.response for kp in kpt1]).astype(np.float32)
    sort_index0, sort_index1 = np.argsort(-scores_np0)[:kpt_num0], np.argsort(-scores_np1)[:kpt_num1]
    scores_np0, kp_np0, desc_np0 = scores_np0[sort_index0], kp_np0[sort_index0], desc_np0[sort_index0]
    scores_np1, kp_np1, desc_np1 = scores_np1[sort_index1], kp_np1[sort_index1], desc_np1[sort_index1]
    kpt0, kpt1 = torch.from_numpy(kp_np0), torch.from_numpy(kp_np1)
    desc0, desc1 = torch.from_numpy(desc_np0), torch.from_numpy(desc_np1)
    scores0, scores1 = torch.from_numpy(scores_np0), torch.from_numpy(scores_np1)
    return {'kpt0': kpt0.unsqueeze(0),
            'kpt1': kpt1.unsqueeze(0),
            'desc0': desc0.unsqueeze(0),
            'desc1': desc1.unsqueeze(0),
            'scores0': scores0.unsqueeze(0),
            'scores1': scores1.unsqueeze(0),
            'image0': torch.from_numpy(image).unsqueeze(0),
            'image1': torch.from_numpy(warped_image).unsqueeze(0),
            'homographies': torch.from_numpy(scaled_homo).unsqueeze(0), }


def read_image_inp(path, device, resize):
    image_unresize = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image_unresize is None:
        return None, None
    w, h = image_unresize.shape[0], image_unresize.shape[1]
    if w > h:
        image_unresize = np.rot90(image_unresize, k=1)
    w_new, h_new = process_resize(w, h, resize)
    image = cv2.resize(image_unresize, (w_new, h_new)).astype('float32')

    inp = frame2tensor(image, device)
    return image, inp, image_unresize


# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w - 1 - cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w - 1 - cx],
                         [0., fy, h - 1 - cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h - 1 - cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1. / scales[0], 1. / scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                      + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
    return d


def compute_pixel_error(pred_points, gt_points):
    diff = gt_points - pred_points
    diff = (diff ** 2).sum(-1)
    sqrt = np.sqrt(diff)
    return sqrt.mean()


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs

def error_auc(errors, thresholds):
    """
    Args:
    errors (list): [N,]
    thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)
    return aucs


# --- VISUALIZATION ---
def plot_image_pair_scale(images, dpi=80):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), dpi=80, sharex=True, sharey=True)
    # ax[0].imshow(image0, cmap='gray')
    # ax[1].imshow(image1, cmap='gray')
    ax[0].imshow(images[0])
    ax[1].imshow(images[1])


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        # ax[i].imshow(imgs[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_image_pair_cai(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        # ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].imshow(imgs[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)



def plot_keypoints_1(kpt, kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    if isinstance(color, str) :
        ax[0].scatter(kpt[:, 0], kpt[:, 1], c='r', s=ps)
        ax[1].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=ps)
        ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c='r', s=ps)
    else:
        ax[0].scatter(kpt[:, 0], kpt[:, 1], c=color, s=ps)
        ax[1].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=ps)
        ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    if isinstance(color, str):
        ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)
    else:
        ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color[0], s=ps)
        ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color[1], s=ps)


def plot_keypoints_2(kpts0, kpts1, kpts0_c, kpts1_c, ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=kpts0_c, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=kpts1_c, s=ps)


def plot_keypoints_patch(kpts0, kpts1, color0, color1, ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color0, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color1, s=ps)


def plot_keypoints_dgmc(kpts0, kpts1, color1, color2, ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color1, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color2, s=ps)


def plot_attention_weight(kpts0, aw1, kpts1, aw2, lw=1.5):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()
    which_head = 3
    aw1, aw2 = aw1[0, which_head, :, :], aw2[0, which_head, :, :]
    aw1 = (aw1 - aw1.min(axis=0)) / (np.sort(aw1, axis=0)[-2] - aw1.min(axis=0))
    aw1 = np.minimum(aw1, 1)
    aw2 = (aw2 - aw2.min(axis=0)) / (np.sort(aw2, axis=0)[-2] - aw2.min(axis=0))
    aw2 = np.minimum(aw2, 1)  # 用第二大的值归一化，排除对角线太大的影响
    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))
    index1 = round(len(kpts0) / 3)
    index2 = round(len(kpts0) * 2 / 3)
    index3 = round(len(kpts0) - 1)
    index4 = round(len(kpts1) / 3)
    index5 = round(len(kpts1) * 2 / 3)
    index6 = round(len(kpts1) - 1)
    # fig.lines = [matplotlib.lines.Line2D(
    #     (fkpts0[index, 0], fkpts0[i+1, 0]), (fkpts0[index, 1], fkpts0[i+1, 1]), zorder=1,
    #     transform=fig.transFigure, c='r', linewidth=lw, alpha=aw1[index][i]) for i in range(len(kpts0)-1)]
    lines1 = [matplotlib.lines.Line2D(
        (fkpts0[index1, 0], fkpts0[i, 0]), (fkpts0[index1, 1], fkpts0[i, 1]), zorder=1,
        transform=fig.transFigure, c='r', linewidth=lw, alpha=aw1[index1][i]) for i in range(len(kpts0))]
    lines2 = [matplotlib.lines.Line2D(
        (fkpts0[index2, 0], fkpts0[i, 0]), (fkpts0[index2, 1], fkpts0[i, 1]), zorder=1,
        transform=fig.transFigure, c='g', linewidth=lw, alpha=aw1[index2][i]) for i in range(len(kpts0))]
    lines3 = [matplotlib.lines.Line2D(
        (fkpts0[index3, 0], fkpts0[i, 0]), (fkpts0[index3, 1], fkpts0[i, 1]), zorder=1,
        transform=fig.transFigure, c='b', linewidth=lw, alpha=aw1[index3][i]) for i in range(len(kpts0))]
    lines4 = [matplotlib.lines.Line2D(
        (fkpts1[index4, 0], fkpts1[i, 0]), (fkpts1[index4, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c='y', linewidth=lw, alpha=aw2[index4][i]) for i in range(len(kpts1))]
    lines5 = [matplotlib.lines.Line2D(
        (fkpts1[index5, 0], fkpts1[i, 0]), (fkpts1[index5, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c='c', linewidth=lw, alpha=aw2[index5][i]) for i in range(len(kpts1))]
    lines6 = [matplotlib.lines.Line2D(
        (fkpts1[index6, 0], fkpts1[i, 0]), (fkpts1[index6, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c='m', linewidth=lw, alpha=aw2[index6][i]) for i in range(len(kpts1))]
    lines = lines1 + lines2 + lines3 + lines4 + lines5 + lines6
    fig.lines = lines


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw, alpha=0.1)
        for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_self_attention(aw1, aw2, image0, image1, kpts0, kpts1, text, path, show_keypoints=False, small_text=[]):
    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)

    plot_attention_weight(kpts0, aw1, kpts1, aw2)
    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot(image0, image1, mkpts0, mkpts1,
                       color, save_path, kpts0=None, kpts1=None, text=[], show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[], dpi=75):
    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, save_path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=0.5)
        plot_keypoints(kpts0, kpts1, color='w', ps=0.25)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()


def kpt_scale_plot_cai(image0, image1, kpts0, kpts1, save_path):

    # plot_image_pair_cai([image0, image1])
    plot_image_pair_scale([image0, image1])

    plot_keypoints(kpts0, kpts1, color='r', ps=0.5)

    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close()


def kpt_scale_plot(image0, image1, kpts0, kpts1, save_path):

    # plot_image_pair_scale([image0, image1])
    plot_image_pair([image0, image1])

    plot_keypoints(kpts0, kpts1, color='r', ps=0.5)

    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_warp_plot(image, image1, kpt, kpts0, kpts1, save_path, color='k'):
    plot_image_pair([image, image1])
    plot_keypoints_1(kpt, kpts0, kpts1, color=color, ps=0.5)
    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def make_kpt_plot(image0, image1, kpts0, kpts1, kpts0_c, kpts1_c, save_path):
    plot_image_pair([image0, image1])
    plot_keypoints_2(kpts0, kpts1, kpts0_c, kpts1_c, ps=0.5)
    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def make_grouping_plot(image0, image1, kpts0, kpts1,
                       color, save_path):

    plot_image_pair([image0, image1])
    plot_keypoints(kpts0, kpts1, color=color, ps=4)
    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_grouping_plot_g(image0, image1, kpts0, kpts1, idx0, idx1,
                       color, save_path):

    plot_image_pair([image0, image1])
    plot_keypoints(kpts0, kpts1, color=color, ps=4)
    plot_keypoints(kpts0[idx0], kpts1[idx1], color='k', ps=20)
    plot_keypoints(kpts0[idx0], kpts1[idx1], color='w', ps=10)
    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_pool_plot(idx0_list, idx1_list, image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path0, path1, path2, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches'):
    idx0_stage1 ,idx0_stage2 = idx0_list[0].squeeze().cpu().numpy(), idx0_list[1].squeeze().cpu().numpy()
    idx1_stage1 ,idx1_stage2 = idx1_list[0].squeeze().cpu().numpy(), idx1_list[1].squeeze().cpu().numpy()

    # plot_image_pair([image0, image1])
    # if show_keypoints:
    #     plot_keypoints(kpts0, kpts1, color='k', ps=8)
    #     plot_keypoints(kpts0, kpts1, color='r', ps=6)
    # fig = plt.gcf()
    # text = [
    #     'gPool',
    #     'Original'
    # ]
    # txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    # fig.text(
    #     0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
    #     fontsize=15, va='top', ha='left', color=txt_color)
    # plt.savefig(str(path0), bbox_inches='tight', pad_inches=0)
    # plt.close()

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0[idx0_stage1[:50],:], kpts1[idx1_stage1[:50],:], color='k', ps=8)
        plot_keypoints(kpts0[idx0_stage1[:50],:], kpts1[idx1_stage1[:50],:], color='r', ps=6)
    fig = plt.gcf()
    text = [
        'gPool',
        # 'Stage 1'
    ]
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)
    plt.savefig(str(path1), bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot_image_pair([image0, image1])
    # if show_keypoints:
    #     plot_keypoints(kpts0[idx0_stage2,:], kpts1[idx1_stage2,:], color='k', ps=8)
    #     plot_keypoints(kpts0[idx0_stage2,:], kpts1[idx1_stage2,:], color='r', ps=6)
    # fig = plt.gcf()
    # text = [
    #     'gPool',
    #     'Stage 2'
    # ]
    # txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    # fig.text(
    #     0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
    #     fontsize=15, va='top', ha='left', color=txt_color)
    # plt.savefig(str(path2), bbox_inches='tight', pad_inches=0)
    # plt.close()


def make_matching_plot_patch(patch_label0, patch_label1, image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                             color, text, path, show_keypoints=False,
                             fast_viz=False, opencv_display=False,
                             opencv_title='matches', small_text=[]):
    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])

    patch_label0, patch_label1 = patch_label0, patch_label1
    patch_label0, patch_label1 = patch_label0 / 3, patch_label1 / 3
    color0 = cm.jet(patch_label0)
    color1 = cm.jet(patch_label1)

    if show_keypoints:
        plot_keypoints_patch(kpts0, kpts1, color0=color0, color1=color1, ps=8)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_dgmc(d0, d1, image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                            color, text, path, show_keypoints=False,
                            fast_viz=False, opencv_display=False,
                            opencv_title='matches', small_text=[]):
    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
        plot_keypoints_dgmc(kpts0, kpts1, d0, d1, ps=10)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def debug_image_plot(debug_path, keypoints0, keypoints1, match_list0, match_list1, image0, image1, epoch, it):
    np_image0, np_image1 = (image0.detach().cpu().numpy() * 255).astype(np.uint8)[0], \
                           (image1.detach().cpu().numpy() * 255).astype(np.uint8)[0]
    np_image0, np_image1 = cv2.cvtColor(np_image0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(np_image1, cv2.COLOR_GRAY2BGR)
    kp0, kp1 = keypoints0.detach().cpu().numpy(), keypoints1.detach().cpu().numpy()
    ma0, ma1 = match_list0.detach().cpu().numpy()[:25], match_list1.detach().cpu().numpy()[:25]
    for i, k in zip(kp0, kp1):
        cv2.circle(np_image0, (int(i[0]), int(i[1])), 2, (255, 0, 0), 1)
        cv2.circle(np_image1, (int(k[0]), int(k[1])), 2, (0, 0, 255), 1)
    write_image = np.concatenate([np_image0, np_image1], axis=1)
    for key1, key2 in zip(ma0, ma1):
        k1, k2 = kp0[key1], kp1[key2]
        cv2.line(write_image, (int(k1[0]), int(k1[1])), (int(k2[0]) + 640, int(k2[1])),
                 color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    write_path = os.path.join(debug_path, "{}_{}.jpg".format(epoch, it))
    cv2.imwrite(write_path, write_image)


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin:] = image1
    out = np.stack([out] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


def weighted_score(results):
    weight = [0.0, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1]
    values = [results['dlt_auc'][0], results['dlt_auc'][1], results['dlt_auc'][2], results['ransac_auc'][0],
              results['ransac_auc'][1], results['ransac_auc'][2], results['precision'], results['recall']]
    weight_score = (np.array(weight) * np.array(values)).sum()
    return weight_score


# def find_pred_disk(inp, extract, superglue_model):
#     pred = {}
#     inp['image0'], inp['image1'] = inp['image0'].permute(0, 1, 4, 2, 3), inp['image1'].permute(0, 1, 4, 2, 3)
#     inp['image0'], inp['image1'] = inp['image0'].squeeze(1), inp['image1'].squeeze(1)
#     image0, image1 = inp['image0'], inp['image1']
#     if 'keypoints0' not in inp:
#         features_ = extract(image0, kind='nms')
#         pred0 = disk_res(features_, device=inp['image0'].device)
#         pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
#     if 'keypoints1' not in inp:
#         features_ = extract(image1, kind='nms')
#         pred1 = disk_res(features_, device=inp['image1'].device)
#         pred = {**pred, **{k + '1': v for k, v in pred1.items()}}
#     data = {**inp, **pred}
#     for k in data:
#         if isinstance(data[k], (list, tuple)):
#             data[k] = torch.stack(data[k])
#     data['keypoints0'], data['keypoints1'] = data['keypoints0'].transpose(0, 1), data['keypoints1'].transpose(0, 1)
#     data['descriptors0'], data['descriptors1'] = data['descriptors0'].permute(2, 0, 1), data['descriptors1'].permute(2,
#                                                                                                                      0,
#                                                                                                                      1)
#     data['scores0'], data['scores1'] = data['scores0'].transpose(0, 1), data['scores1'].transpose(0, 1)
#     pred = {**pred, **superglue_model(data)}
#     return pred


def find_pred(inp, superpoint_model, superglue_model):
    pred = {}

    pred0 = superpoint_model({'image': inp['image0']}, curr_max_kp=512, curr_key_thresh=0.005)
    pred = {**pred, **{k + '0': v for k, v in pred0.items()}}

    pred1 = superpoint_model({'image': inp['image1']}, curr_max_kp=512, curr_key_thresh=0.005)
    pred = {**pred, **{k + '1': v for k, v in pred1.items()}}
    data = {**inp, **pred}
    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])

    data['scores0'], data['scores1'] = data['scores0'].unsqueeze(1), data['scores1'].unsqueeze(1)
    data['image_shape'] = data['image0'].shape[-2:]
    for k, v in data.items():
        if k != 'image_shape':
            data[k] = v.half()
    pred = {**pred, **superglue_model(data)}
    return pred


def find_pred_sift(inp, sift_model, superglue_model):
    pred = {}

    pred0 = sift_model({'image': inp['image0']})
    pred = {**pred, **{k + '0': v for k, v in pred0.items()}}

    pred1 = sift_model({'image': inp['image1']})
    pred = {**pred, **{k + '1': v for k, v in pred1.items()}}
    data = {**inp, **pred}
    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])
    data['keypoints0'], data['keypoints1'] = data['keypoints0'].transpose(0, 1), data['keypoints1'].transpose(0, 1)
    data['descriptors0'], data['descriptors1'] = data['descriptors0'].permute(2, 0, 1), data['descriptors1'].permute(2, 0, 1)
    data['scores0'], data['scores1'] = data['scores0'].transpose(0, 1), data['scores1'].transpose(0, 1)
    pred = {**pred, **superglue_model(data)}
    return pred


def test_model(test_loader, superpoint_model, superglue_model, val_count, device, viz_path, viz_idx, epoch, i, min_matches=12):
    all_recall, all_precision, all_error_dlt, all_error_ransac, all_group_acc = [], [], [], [], []
    for i_no, (orig_warped, homography) in enumerate(test_loader):
        if i_no >= val_count:
            break
        orig_image, warped_image = orig_warped[0:1, :, :, :].to(device), orig_warped[1:2, :, :, :].to(device)
        homography = homography[0].to(device)
        pred = find_pred({'image0': orig_image, 'image1': warped_image}, superpoint_model, superglue_model)
        kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        # if i_no in viz_idx:
        #     sive_path = viz_path / 'group' / 'idx{}_epoch{}_iter{}.png'.format(i_no, epoch, i)
        #     attn0, attn1 = pred['attn0'], pred['attn1']
        #     color0 = cm.jet(((np.where(attn0 == 1)[-1]) + 1) / 4)
        #     color1 = cm.jet(((np.where(attn1 == 1)[-1]) + 1) / 4)
        #     color = [color0, color1]
        #     image0, image1 = (orig_image*255.).squeeze().detach().cpu().numpy(), (warped_image*255.).squeeze().detach().cpu().numpy()
        #     make_grouping_plot(image0, image1, kpts0, kpts1, color, sive_path)
        #     sive_path = viz_path / 'match' / 'idx{}_epoch{}_iter{}.png'.format(i_no, epoch, i)
        #     # Visualize the matches.
        #     color = cm.jet(mconf)
        #     text = [
        #         'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        #         'Matches: {}'.format(len(mkpts0)),
        #     ]
        #     make_matching_plot(
        #         image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        #         sive_path, text, show_keypoints=True)

        if len(mconf) < min_matches:
            all_precision.append(0)
            all_recall.append(0)
            all_group_acc.append(0)
            all_error_dlt.append(500)
            all_error_ransac.append(500)
            continue
        ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, homography, dist_thresh=3, n_iters=3)
        ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
        gt_match_vec = np.ones((len(matches),), dtype=np.int32) * -1
        gt_match_vec[ma_0] = ma_1
        corner_points = np.array([[0, 0], [0, orig_image.shape[2]], [orig_image.shape[3], orig_image.shape[2]],
                                  [orig_image.shape[3], 0]]).astype(np.float32)
        sort_index = np.argsort(mconf)[::-1][0:4]
        est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
        est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)
        corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
        corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(
            1)
        corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)),
                                                    homography.cpu().numpy()).squeeze(1)
        error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
        error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
        match_flag = (matches[ma_0] == ma_1)
        precision = match_flag.sum() / valid.sum()
        fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
        # group0, group1 = np.where(pred['attn0'] == 1)[1], np.where(pred['attn1'] == 1)[1]
        # group0_m, group1_m = group0[ma_0], group1[ma_1]
        # group_flag = group0_m == group1_m
        # group_acc = group_flag.sum() / group_flag.size
        group_acc = 0
        if (match_flag.sum() + fn_flag.sum()) == 0:
            all_precision.append(0)
            all_recall.append(0)
            all_group_acc.append(0)
            all_error_dlt.append(500)
            all_error_ransac.append(500)
            continue
        recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
        all_precision.append(precision)
        all_recall.append(recall)
        all_error_dlt.append(error_dlt)
        all_error_ransac.append(error_ransac)
        all_group_acc.append(group_acc)
    thresholds = [5, 10, 25]
    aucs_dlt = pose_auc(all_error_dlt, thresholds)
    aucs_ransac = pose_auc(all_error_ransac, thresholds)
    aucs_dlt = [float(100. * yy) for yy in aucs_dlt]
    aucs_ransac = [float(100. * yy) for yy in aucs_ransac]
    prec = float(100. * np.mean(all_precision))
    rec = float(100. * np.mean(all_recall))
    all_group_acc = 100. * np.mean(all_group_acc)
    results_dict = {'dlt_auc': aucs_dlt, 'ransac_auc': aucs_ransac, 'precision': prec, 'recall': rec,
                    'thresholds': thresholds, 'group_acc': all_group_acc}
    weight_score = weighted_score(results_dict)
    results_dict['weight_score'] = float(weight_score)
    if prec + rec == 0:
        f1_score = 0
    else:
        f1_score = (2 * prec * rec) / (prec + rec)
    results_dict['f1_score'] = float(f1_score)
    print("For DLT results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t F1-score\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec, f1_score))
    print("For homography results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t F1-score\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec, f1_score))
    print("For group results, acc = {}".format(all_group_acc))
    return results_dict


def test_model_sift(test_loader, sift_model, superglue_model, val_count, device, min_matches=12):
    superglue_model.eval()
    all_recall, all_precision, all_error_dlt, all_error_ransac = [], [], [], []
    for i_no, data in enumerate(test_loader):
        if i_no >= val_count:
            break
        if data['skip']:
            continue
        homography = data['homographies'][0].to(device)
        orig_image, warped_image = data['image0'], data['image1']
        keypoints0, keypoints1 = data['kpt0'].to(device, non_blocking=True), data['kpt1'].to(device, non_blocking=True)
        descriptors0, descriptors1 = data['desc0'].to(device, non_blocking=True), data['desc1'].to(device,
                                                                                                   non_blocking=True)
        scores0, scores1 = data['scores0'].to(device, non_blocking=True), data['scores1'].to(device, non_blocking=True)
        images0, images1 = data['image0'], data['image1']
        pred = {
            'keypoints0': keypoints0, 'keypoints1': keypoints1,
            'descriptors0': descriptors0.transpose(1, 2), 'descriptors1': descriptors1.transpose(1, 2),
            'image0': images0, 'image1': images1,
            'scores0': scores0, 'scores1': scores1,
        }
        pred = {**pred, **superglue_model(pred)}
        kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        if len(mconf) < min_matches:
            all_precision.append(0)
            all_recall.append(0)
            all_error_dlt.append(500)
            all_error_ransac.append(500)
            continue
        ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, homography, dist_thresh=3, n_iters=3)
        ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
        gt_match_vec = np.ones((len(matches),), dtype=np.int32) * -1
        gt_match_vec[ma_0] = ma_1
        corner_points = np.array([[0, 0], [0, orig_image.shape[1]], [orig_image.shape[2], orig_image.shape[1]],
                                  [orig_image.shape[2], 0]]).astype(np.float32)
        sort_index = np.argsort(mconf)[::-1][0:4]
        est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
        est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)
        corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
        corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(
            1)
        corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)),
                                                    homography.cpu().numpy()).squeeze(1)
        error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
        error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
        match_flag = (matches[ma_0] == ma_1)
        precision = match_flag.sum() / valid.sum()
        fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
        if (match_flag.sum() + fn_flag.sum()) == 0:
            all_precision.append(0)
            all_recall.append(0)
            all_error_dlt.append(500)
            all_error_ransac.append(500)
            continue
        recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
        all_precision.append(precision)
        all_recall.append(recall)
        all_error_dlt.append(error_dlt)
        all_error_ransac.append(error_ransac)

    thresholds = [5, 10, 25]
    aucs_dlt = pose_auc(all_error_dlt, thresholds)
    aucs_ransac = pose_auc(all_error_ransac, thresholds)
    aucs_dlt = [float(100. * yy) for yy in aucs_dlt]
    aucs_ransac = [float(100. * yy) for yy in aucs_ransac]
    prec = float(100. * np.mean(all_precision))
    rec = float(100. * np.mean(all_recall))
    results_dict = {'dlt_auc': aucs_dlt, 'ransac_auc': aucs_ransac, 'precision': prec, 'recall': rec,
                    'thresholds': thresholds}
    weight_score = weighted_score(results_dict)
    results_dict['weight_score'] = float(weight_score)
    if prec + rec == 0:
        f1_score = 0
    else:
        f1_score = (2 * prec * rec) / (prec + rec)
    results_dict['f1_score'] = float(f1_score)
    print("For DLT results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t F1-score\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec, f1_score))
    print("For homography results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t F1-score\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec, f1_score))
    superglue_model.train()
    return results_dict


# def test_model_disk(test_loader, extract, superglue_model, val_count, device, min_matches=12):
#     superglue_model.eval()
#     all_recall, all_precision, all_error_dlt, all_error_ransac = [], [], [], []
#     for i_no, (orig_warped, homography) in enumerate(test_loader):
#         if i_no >= val_count:
#             break
#         orig_image, warped_image = orig_warped[0:1, :, :, :].to(device), orig_warped[1:2, :, :, :].to(device)
#         homography = homography[0].to(device)
#         pred = find_pred_disk({'image0': orig_image, 'image1': warped_image}, extract, superglue_model)
#         kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
#         pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
#         kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
#         matches, conf = pred['matches0'], pred['matching_scores0']
#         valid = matches > -1
#         mkpts0 = kpts0[valid]
#         mkpts1 = kpts1[matches[valid]]
#         mconf = conf[valid]
#         if len(mconf) < min_matches:
#             all_precision.append(0)
#             all_recall.append(0)
#             all_error_dlt.append(500)
#             all_error_ransac.append(500)
#             continue
#         ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, homography, dist_thresh=3, n_iters=3)
#         ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
#         gt_match_vec = np.ones((len(matches),), dtype=np.int32) * -1
#         gt_match_vec[ma_0] = ma_1
#         corner_points = np.array([[0, 0], [0, orig_image.shape[2]], [orig_image.shape[3], orig_image.shape[2]],
#                                   [orig_image.shape[3], 0]]).astype(np.float32)
#         sort_index = np.argsort(mconf)[::-1][0:4]
#         est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
#         est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)
#         corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
#         corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(
#             1)
#         corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)),
#                                                     homography.cpu().numpy()).squeeze(1)
#         error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
#         error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
#         match_flag = (matches[ma_0] == ma_1)
#         precision = match_flag.sum() / valid.sum()
#         fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
#         if (match_flag.sum() + fn_flag.sum()) == 0:
#             all_precision.append(0)
#             all_recall.append(0)
#             all_error_dlt.append(500)
#             all_error_ransac.append(500)
#             continue
#         recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
#         all_precision.append(precision)
#         all_recall.append(recall)
#         all_error_dlt.append(error_dlt)
#         all_error_ransac.append(error_ransac)
#
#     thresholds = [5, 10, 25]
#     aucs_dlt = pose_auc(all_error_dlt, thresholds)
#     aucs_ransac = pose_auc(all_error_ransac, thresholds)
#     aucs_dlt = [float(100. * yy) for yy in aucs_dlt]
#     aucs_ransac = [float(100. * yy) for yy in aucs_ransac]
#     prec = float(100. * np.mean(all_precision))
#     rec = float(100. * np.mean(all_recall))
#     results_dict = {'dlt_auc': aucs_dlt, 'ransac_auc': aucs_ransac, 'precision': prec, 'recall': rec,
#                     'thresholds': thresholds}
#     weight_score = weighted_score(results_dict)
#     results_dict['weight_score'] = float(weight_score)
#     if prec + rec == 0:
#         f1_score = 0
#     else:
#         f1_score = (2 * prec * rec) / (prec + rec)
#     results_dict['f1_score'] = float(f1_score)
#     print("For DLT results...")
#     print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t F1-score\t')
#     print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
#         aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec, f1_score))
#     print("For homography results...")
#     print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t F1-score\t')
#     print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
#         aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec, f1_score))
#     superglue_model.train()
#     return results_dict


def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """
    #Taken from yolov5 repo
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 4000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
