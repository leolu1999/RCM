from src.config.default import _CN as cfg

cfg.RCM.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
cfg.RCM.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
cfg.RCM.COARSE.D_MODEL = 256
cfg.RCM.FINE.D_MODEL = 64

# SuperPoint
cfg.RCM.MAX_KEYPOINTS = 2048

# pose estimation
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, LO-RANSAC]
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5 

cfg.TRAINER.EPI_ERR_THR = 1e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

cfg.RCM.MATCH_COARSE.BORDER_RM = 2
cfg.RCM.EVAL_TIMES = 1