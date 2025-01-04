from configs.data.base import cfg


TRAIN_BASE_PATH = "data/megadepth/index"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TRAIN_DATA_ROOT = "data/megadepth/train"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/trainvaltest_list/train_list_10.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

TEST_BASE_PATH = "data/megadepth/index"
cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list_15.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val
cfg.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
cfg.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
# 368 scenes in total for MegaDepth
# (with difficulty balanced (further split each scene to 3 sub-scenes))
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.MGDPT_IMG_RESIZE = 544  # for training on 11GB mem GPUs
cfg.DATASET.CORR_TH = 5
cfg.TRAINER.EPI_ERR_THR = 1e-4