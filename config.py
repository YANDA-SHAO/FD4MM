# =========================================================
# Usage notes
# =========================================================
# 1) Train on original DeepMag dataset from scratch:
#    self.dataset_name = 'deepmag'
#    self.finetune = False
#    self.max_samples = None
#
# 2) Finetune on Kubric dataset:
#    self.dataset_name = 'kubric'
#    self.finetune = True
#    self.max_samples = None
#
# 3) Quick debug with a small subset:
#    self.max_samples = 200
#
# Notes:
# - max_samples=None means using all samples listed in train_mf.txt
# - sample count is inferred automatically from train_mf.txt
# - save_dir is generated automatically based on dataset_name and finetune
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
        self.wandb_entity = 'yandashao'          # 你的 team / username，没有就 None
        self.wandb_mode = 'online'        # 'online', 'offline', 'disabled'
        
        
        # =========================================================
        # 1) High-level switches: only change these in daily use
        # =========================================================
        # options: 'deepmag', 'kubric'
        self.dataset_name = 'kubric'

        # False: train from scratch on selected dataset
        # True : load pretrained weights and finetune on selected dataset
        self.finetune = True

        # Optional: limit sample count for quick debugging
        # None means use all samples found in train_mf.txt
        self.max_samples = None

        # =========================================================
        # 2) General training settings
        # =========================================================
        self.epochs = 100
        self.batch_size = 40
        self.workers = 2

        self.test_batch_size = 1
        self.batch_size_test = self.test_batch_size

        self.test_workers = 4
        self.workers_test = self.test_workers
        self.numtestdata = 600

        self.lr = 1e-4
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.0
        self.preproc = ['resize', 'gaussian']
        self.load_all = False
        self.videos_train = []

        # =========================================================
        # 3) Data roots
        # =========================================================
        self.data_dir = '/data/curtin_cumlg/curtin_yanda/work/datasets/'

        self.dataset_roots = {
            'deepmag': os.path.join(self.data_dir, 'train'),
            'kubric': os.path.join(self.data_dir, 'Kubric_vmm_train'),
        }

        if self.dataset_name not in self.dataset_roots:
            raise ValueError(
                f"Unknown dataset_name: {self.dataset_name}. "
                f"Supported: {list(self.dataset_roots.keys())}"
            )

        self.dir_train = self.dataset_roots[self.dataset_name]

        # =========================================================
        # 4) Pretrained weights
        # =========================================================
        self.pretrained_weights = '/data/curtin_cumlg/curtin_yanda/work/FD4MM/weights_dateFD4MM_kubric_ft/magnet_epoch65_loss3.91e-01.pth'
        if self.finetune:
            self.pretrained_weights = '/data/curtin_cumlg/curtin_yanda/work/FD4MM/weights_dateFD4MM_kubric_ft/magnet_epoch65_loss3.91e-01.pth'
            #self.pretrained_weights = '/data/curtin_cumlg/curtin_yanda/work/FD4MM/weights_dateFD4MM/magnet_epoch4_loss4.99e-01.pth'

        # =========================================================
        # 5) Auto-discover training metadata from train_mf.txt
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
        # 6) Save / logging names
        # =========================================================
        if self.finetune:
            self.date = f'FD4MM_{self.dataset_name}_ft'
        else:
            self.date = f'FD4MM_{self.dataset_name}'

        self.save_dir = f'weights_date{self.date}'
        self.wandb_run_name = self.save_dir
        self.time_st = time.time()
        self.losses = []

        self.num_val_per_epoch = 1000

        # =========================================================
        # 7) Dataset sanity check
        # =========================================================
        self._check_train_dataset()

        # =========================================================
        # 8) Keep your original evaluation/test paths
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

    def _check_train_dataset(self):
        required_dirs = ['frameA', 'frameB', 'amplified']
        counts = {}

        for d in required_dirs:
            p = os.path.join(self.dir_train, d)
            if not os.path.isdir(p):
                raise FileNotFoundError(f'Required folder not found: {p}')
            counts[d] = len([x for x in os.listdir(p) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])

        n_mf = len(self.coco_amp_lst)

        print(f'[Config] dataset_name: {self.dataset_name}')
        print(f'[Config] dir_train: {self.dir_train}')
        print(f'[Config] frameA: {counts["frameA"]}, frameB: {counts["frameB"]}, amplified: {counts["amplified"]}, mf: {n_mf}')

        if not (counts['frameA'] == counts['frameB'] == counts['amplified'] == n_mf):
            raise ValueError(
                'Dataset size mismatch: '
                f'frameA={counts["frameA"]}, '
                f'frameB={counts["frameB"]}, '
                f'amplified={counts["amplified"]}, '
                f'train_mf={n_mf}'
            )



# def mse(imageA, imageB):
#     # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
#     mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     mse_error /= float(imageA.shape[0] * imageA.shape[1] * 255 )
#     mse_error /= (np.mean((imageA.astype("float"))))**2
#     # return the MSE. The lower the error, the more "similar" the two images are.
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

def calc_psnr(img1, img2): #这里输入的是（0,255）的灰度或彩色图像，如果是彩色图像，则numpy.mean相当于对三个通道计算的结果再求均值
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10: # 如果两图片差距过小代表完美重合
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) # 将对数中pixel_max的平方放了下来


# def calc_psnr(img1, img2):

#     # img1 = Image.open(img1_path)
#     # img2 = Image.open(img2_path)
#     # img2 = img2.resize(img1.size)
#     # img1, img2 = np.array(img1), np.array(img2)
#     # 此处的第一张图片为真实图像，第二张图片为测试图片
#     # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
#     psnr_score = psnr(img1, img2, data_range=255)
#     return psnr_score

def calc_ssim(img1, img2):

    # img1 = Image.open(img1_path).convert('L')
    # img2 = Image.open(img2_path).convert('L')
    # img2 = img2.resize(img1.size)
    # img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255 , multichannel=True)
    return ssim_score


import json

""" configuration json """
class Configjson(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Configjson(config)
