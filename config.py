# =========================================================
# Usage notes
# =========================================================
# 1) DeepMag only:
#    self.exp_name = 'deepmag_only'
#
# 2) Kubric only:
#    self.exp_name = 'kubric_only'
#
# 3) Pretrain on DeepMag, finetune on Kubric:
#    self.exp_name = 'deepmag_to_kubric'
#    self.pretrained_weights = 'path_to_deepmag_best.pth'   # used only in finetune stage
#
# 4) Quick debug:
#    self.max_samples = 200
#
# Notes:
# - max_samples=None means using all samples in the selected split
# - Each split folder must contain:
#     frameA/, frameB/, frameC/, amplified/, meta/, train_mf.txt
# - save_dir is generated automatically from exp_name and hp_tag
# =========================================================

import os
import time
import json
import math
import numpy as np

from skimage.metrics import mean_squared_error as compare_mse
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from cmath import sqrt


class Config(object):
    def __init__(self):

        # -----------------------------
        # W&B
        # -----------------------------
        self.wandb_project = 'Magnification'
        self.wandb_entity = 'yandashao'
        self.wandb_mode = 'online'   # 'online', 'offline', 'disabled'

        # =========================================================
        # 1) Experiment mode
        # =========================================================
        # options:
        #   'deepmag_only'
        #   'kubric_only'
        #   'deepmag_to_kubric'
        self.exp_name = 'deepmag_only'

        # Optional run tag for manual hyperparameter search / ablation naming
        # e.g. 'bs8_lr1e-4', 'ft_lr5e-5', 'debug200'
        self.hp_tag = 'exp1_deepmag_only'

        # Optional: limit sample count for quick debugging
        # None means use all samples found in the selected split's train_mf.txt
        self.max_samples = None

        # =========================================================
        # 2) General training settings
        # =========================================================
        self.epochs = 10
        self.batch_size = 32
        self.workers = 2

        self.test_batch_size = 1
        self.batch_size_test = self.test_batch_size

        self.test_workers = 2
        self.workers_test = self.test_workers

        self.numtestdata = 600

        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.0

        self.preproc = ['resize', 'poisson']
        self.load_all = False
        self.videos_train = []

        # =========================================================
        # 3) Validation / checkpoint behaviour
        # =========================================================
        # Validation is run every eval_every_epoch epochs
        self.eval_every_iter = 1000   # 大概10分钟一次，需要根据速度调整
        self.eval_every_epoch = None  # 不再按epoch
        self.val_subset = 200

        # Save one checkpoint every save_every_epoch epochs
        self.save_every_epoch = 1

        # Which metric decides best checkpoint:
        # options: 'psnr', 'loss'
        self.save_best_by = 'psnr'

        # Print train log every N iterations
        self.num_print_per_epoch = 10

        # =========================================================
        # 4) Data roots
        # =========================================================
        self.data_dir = '/data/curtin_cumlg/curtin_yanda/work/datasets/'

        # Root folders now point to dataset root, NOT directly to train/
        self.dataset_roots = {
            'deepmag': os.path.join(self.data_dir, 'deepmag'),
            'kubric': os.path.join(self.data_dir, 'Kubric_vmm'),
        }

        # =========================================================
        # 5) Experiment-specific dataset / finetune settings
        # =========================================================
        self.dataset_name = None
        self.finetune = False
        self.pretrained_weights = ''

        if self.exp_name == 'deepmag_only':
            self.dataset_name = 'deepmag'
            self.finetune = False
            self.pretrained_weights = ''

        elif self.exp_name == 'kubric_only':
            self.dataset_name = 'kubric'
            self.finetune = False
            self.pretrained_weights = ''

        elif self.exp_name == 'deepmag_to_kubric':
            self.dataset_name = 'kubric'
            self.finetune = True
            # IMPORTANT:
            # set this path manually before finetuning
            # Example:
            # self.pretrained_weights = '/data/.../weights_FD4MM_deepmag_default/best_psnr.pth'
            self.pretrained_weights = ''

        else:
            raise ValueError(
                f"Unknown exp_name: {self.exp_name}. "
                f"Supported: ['deepmag_only', 'kubric_only', 'deepmag_to_kubric']"
            )

        if self.dataset_name not in self.dataset_roots:
            raise ValueError(
                f"Unknown dataset_name: {self.dataset_name}. "
                f"Supported: {list(self.dataset_roots.keys())}"
            )

        self.dataset_root = self.dataset_roots[self.dataset_name]
        self.dir_train = os.path.join(self.dataset_root, 'train')
        self.dir_val = os.path.join(self.dataset_root, 'val')
        self.dir_test = os.path.join(self.dataset_root, 'test')

        # =========================================================
        # 6) Auto-discover training metadata from train_mf.txt
        # =========================================================
        self.mf_path = os.path.join(self.dir_train, 'train_mf.txt')
        if not os.path.exists(self.mf_path):
            raise FileNotFoundError(f"train_mf.txt not found: {self.mf_path}")

        self.coco_amp_lst = np.loadtxt(self.mf_path)

        # np.loadtxt returns scalar if file has only one number
        if np.ndim(self.coco_amp_lst) == 0:
            self.coco_amp_lst = np.array([float(self.coco_amp_lst)])

        if self.max_samples is not None:
            self.coco_amp_lst = self.coco_amp_lst[:self.max_samples]

        self.numdata = len(self.coco_amp_lst)
        self.frames_train = f'coco{self.numdata}'
        self.cursor_end = self.numdata

        # =========================================================
        # 7) Save / logging names
        # =========================================================
        # Keep naming readable and stable
        if self.exp_name == 'deepmag_only':
            self.date = 'FD4MM_deepmag'
        elif self.exp_name == 'kubric_only':
            self.date = 'FD4MM_kubric'
        elif self.exp_name == 'deepmag_to_kubric':
            self.date = 'FD4MM_deepmag2kubric_ft'
        else:
            self.date = 'FD4MM_unknown'

        self.save_dir = f'weights_{self.date}_{self.hp_tag}'
        self.wandb_run_name = self.save_dir
        self.time_st = time.time()
        self.losses = []

        # =========================================================
        # 8) Dataset sanity checks
        # =========================================================
        self._check_split_dataset(self.dir_train, split_name='train', expected_num=self.numdata)
        self._check_split_dataset(self.dir_val, split_name='val', expected_num=None)
        self._check_split_dataset(self.dir_test, split_name='test', expected_num=None)

        # =========================================================
        # 9) Keep your original evaluation/test paths
        # =========================================================
        # amp test
        self.dir_amp0 = os.path.join(self.data_dir, 'systest/amp0/000000')
        self.dir_amp1 = os.path.join(self.data_dir, 'systest/amp0/000001')
        self.dir_amp2 = os.path.join(self.data_dir, 'systest/amp0/000002')
        self.dir_amp3 = os.path.join(self.data_dir, 'systest/amp0/000003')
        self.dir_amp4 = os.path.join(self.data_dir, 'systest/amp0/000004')
        self.dir_amp5 = os.path.join(self.data_dir, 'systest/amp0/000005')
        self.dir_amp6 = os.path.join(self.data_dir, 'systest/amp0/000006')
        self.dir_amp7 = os.path.join(self.data_dir, 'systest/amp0/000007')
        self.dir_amp8 = os.path.join(self.data_dir, 'systest/amp0/000008')
        self.dir_amp9 = os.path.join(self.data_dir, 'systest/amp0/000009')

        self.dir_5amp0 = os.path.join(self.data_dir, 'systest/amp5/000000')
        self.dir_5amp1 = os.path.join(self.data_dir, 'systest/amp5/000001')
        self.dir_5amp2 = os.path.join(self.data_dir, 'systest/amp5/000002')
        self.dir_5amp3 = os.path.join(self.data_dir, 'systest/amp5/000003')
        self.dir_5amp4 = os.path.join(self.data_dir, 'systest/amp5/000004')
        self.dir_5amp5 = os.path.join(self.data_dir, 'systest/amp5/000005')
        self.dir_5amp6 = os.path.join(self.data_dir, 'systest/amp5/000006')
        self.dir_5amp7 = os.path.join(self.data_dir, 'systest/amp5/000007')
        self.dir_5amp8 = os.path.join(self.data_dir, 'systest/amp5/000008')
        self.dir_5amp9 = os.path.join(self.data_dir, 'systest/amp5/000009')

        self.dir_10amp0 = os.path.join(self.data_dir, 'systest/amp10/000000')
        self.dir_10amp1 = os.path.join(self.data_dir, 'systest/amp10/000001')
        self.dir_10amp2 = os.path.join(self.data_dir, 'systest/amp10/000002')
        self.dir_10amp3 = os.path.join(self.data_dir, 'systest/amp10/000003')
        self.dir_10amp4 = os.path.join(self.data_dir, 'systest/amp10/000004')
        self.dir_10amp5 = os.path.join(self.data_dir, 'systest/amp10/000005')
        self.dir_10amp6 = os.path.join(self.data_dir, 'systest/amp10/000006')
        self.dir_10amp7 = os.path.join(self.data_dir, 'systest/amp10/000007')
        self.dir_10amp8 = os.path.join(self.data_dir, 'systest/amp10/000008')
        self.dir_10amp9 = os.path.join(self.data_dir, 'systest/amp10/000009')

        self.dir_20amp0 = os.path.join(self.data_dir, 'systest/amp20/000000')
        self.dir_20amp1 = os.path.join(self.data_dir, 'systest/amp20/000001')
        self.dir_20amp2 = os.path.join(self.data_dir, 'systest/amp20/000002')
        self.dir_20amp3 = os.path.join(self.data_dir, 'systest/amp20/000003')
        self.dir_20amp4 = os.path.join(self.data_dir, 'systest/amp20/000004')
        self.dir_20amp5 = os.path.join(self.data_dir, 'systest/amp20/000005')
        self.dir_20amp6 = os.path.join(self.data_dir, 'systest/amp20/000006')
        self.dir_20amp7 = os.path.join(self.data_dir, 'systest/amp20/000007')
        self.dir_20amp8 = os.path.join(self.data_dir, 'systest/amp20/000008')
        self.dir_20amp9 = os.path.join(self.data_dir, 'systest/amp20/000009')

        self.dir_50amp0 = os.path.join(self.data_dir, 'systest/amp50/000000')
        self.dir_50amp1 = os.path.join(self.data_dir, 'systest/amp50/000001')
        self.dir_50amp2 = os.path.join(self.data_dir, 'systest/amp50/000002')
        self.dir_50amp3 = os.path.join(self.data_dir, 'systest/amp50/000003')
        self.dir_50amp4 = os.path.join(self.data_dir, 'systest/amp50/000004')
        self.dir_50amp5 = os.path.join(self.data_dir, 'systest/amp50/000005')
        self.dir_50amp6 = os.path.join(self.data_dir, 'systest/amp50/000006')
        self.dir_50amp7 = os.path.join(self.data_dir, 'systest/amp50/000007')
        self.dir_50amp8 = os.path.join(self.data_dir, 'systest/amp50/000008')
        self.dir_50amp9 = os.path.join(self.data_dir, 'systest/amp50/000009')

        self.dir_100amp0 = os.path.join(self.data_dir, 'systest/amp100/000000')
        self.dir_100amp1 = os.path.join(self.data_dir, 'systest/amp100/000001')
        self.dir_100amp2 = os.path.join(self.data_dir, 'systest/amp100/000002')
        self.dir_100amp3 = os.path.join(self.data_dir, 'systest/amp100/000003')
        self.dir_100amp4 = os.path.join(self.data_dir, 'systest/amp100/000004')
        self.dir_100amp5 = os.path.join(self.data_dir, 'systest/amp100/000005')
        self.dir_100amp6 = os.path.join(self.data_dir, 'systest/amp100/000006')
        self.dir_100amp7 = os.path.join(self.data_dir, 'systest/amp100/000007')
        self.dir_100amp8 = os.path.join(self.data_dir, 'systest/amp100/000008')
        self.dir_100amp9 = os.path.join(self.data_dir, 'systest/amp100/000009')

        self.dir_001noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000000')
        self.dir_001noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000001')
        self.dir_001noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000002')
        self.dir_001noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000003')
        self.dir_001noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000004')
        self.dir_001noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000005')
        self.dir_001noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000006')
        self.dir_001noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000007')
        self.dir_001noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000008')
        self.dir_001noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000009')

        self.dir_005noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000000')
        self.dir_005noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000001')
        self.dir_005noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000002')
        self.dir_005noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000003')
        self.dir_005noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000004')
        self.dir_005noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000005')
        self.dir_005noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000006')
        self.dir_005noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000007')
        self.dir_005noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000008')
        self.dir_005noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000009')

        self.dir_01noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000000')
        self.dir_01noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000001')
        self.dir_01noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000002')
        self.dir_01noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000003')
        self.dir_01noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000004')
        self.dir_01noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000005')
        self.dir_01noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000006')
        self.dir_01noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000007')
        self.dir_01noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000008')
        self.dir_01noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000009')

        self.dir_02noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000000')
        self.dir_02noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000001')
        self.dir_02noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000002')
        self.dir_02noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000003')
        self.dir_02noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000004')
        self.dir_02noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000005')
        self.dir_02noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000006')
        self.dir_02noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000007')
        self.dir_02noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000008')
        self.dir_02noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000009')

        self.dir_05noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000000')
        self.dir_05noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000001')
        self.dir_05noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000002')
        self.dir_05noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000003')
        self.dir_05noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000004')
        self.dir_05noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000005')
        self.dir_05noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000006')
        self.dir_05noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000007')
        self.dir_05noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000008')
        self.dir_05noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000009')

        self.dir_baby = os.path.join(self.data_dir, 'train/train_vid_frames/val_baby')
        self.dir_crane_crop = os.path.join(self.data_dir, 'train/train_vid_frames/val_crane_crop')

    def _count_files(self, folder, suffixes):
        return len([
            x for x in os.listdir(folder)
            if x.lower().endswith(suffixes)
        ])

    def _check_split_dataset(self, split_dir, split_name='train', expected_num=None):
        required_dirs = ['frameA', 'frameB', 'frameC', 'amplified', 'meta']
        counts = {}

        for d in required_dirs:
            p = os.path.join(split_dir, d)
            if not os.path.isdir(p):
                raise FileNotFoundError(f'Required folder not found: {p}')

            if d == 'meta':
                counts[d] = self._count_files(p, ('.json',))
            else:
                counts[d] = self._count_files(p, ('.png', '.jpg', '.jpeg'))

        mf_path = os.path.join(split_dir, 'train_mf.txt')
        if not os.path.exists(mf_path):
            raise FileNotFoundError(f'train_mf.txt not found: {mf_path}')

        split_mf = np.loadtxt(mf_path)
        if np.ndim(split_mf) == 0:
            split_mf = np.array([float(split_mf)])
        n_mf = len(split_mf)

        print(f'[Config] split={split_name} | dir={split_dir}')
        print(
            f'[Config] frameA={counts["frameA"]}, '
            f'frameB={counts["frameB"]}, '
            f'frameC={counts["frameC"]}, '
            f'amplified={counts["amplified"]}, '
            f'meta={counts["meta"]}, '
            f'mf={n_mf}'
        )

        all_equal = (
            counts['frameA'] == counts['frameB'] ==
            counts['frameC'] == counts['amplified'] ==
            counts['meta'] == n_mf
        )

        if not all_equal:
            raise ValueError(
                f'Dataset size mismatch in split={split_name}: '
                f'frameA={counts["frameA"]}, '
                f'frameB={counts["frameB"]}, '
                f'frameC={counts["frameC"]}, '
                f'amplified={counts["amplified"]}, '
                f'meta={counts["meta"]}, '
                f'train_mf={n_mf}'
            )

        if expected_num is not None and n_mf != expected_num:
            raise ValueError(
                f'Expected {expected_num} samples in split={split_name}, '
                f'but found {n_mf}'
            )


# def mse(imageA, imageB):
#     mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     mse_error /= float(imageA.shape[0] * imageA.shape[1] * 255 )
#     mse_error /= (np.mean((imageA.astype("float"))))**2
#     return mse_error

# def mae(imageA, imageB):
#     mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float"))))
#     mae /= float(imageA.shape[0] * imageA.shape[1] * 255)
#     if (mae < 0):
#         return mae * -1
#     else:
#         return mae

from skimage.metrics import mean_squared_error as compare_mse
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import math
from cmath import sqrt

# def calc_mae(img1, img2):
#     mae_score = compare_mae(img1, img2)
#     return mae_score

def calc_mse(img1, img2):
    mse_score = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse_score

def calc_rmse(img1, img2):
    mse_score = np.mean((img1 / 255. - img2 / 255.) ** 2)
    rmse_score = sqrt(mse_score)
    return rmse_score

def calc_psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_ssim(img1, img2):
    """
    img1, img2: HWC uint8 RGB image in [0,255]
    Compatible with both old and new skimage APIs.
    """
    try:
        return ssim(
            img1,
            img2,
            data_range=255,
            channel_axis=-1
        )
    except TypeError:
        return ssim(
            img1,
            img2,
            data_range=255,
            multichannel=True
        )


class Configjson(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Configjson(config)