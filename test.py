import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

from config import Config, calc_mse, calc_rmse, calc_psnr, calc_ssim
from utils.data_loader import ImageFromFolderTest
from magnet_FD4MM import MagNet
from callbacks import gen_state_dict


def tensor_to_img255(x):
    """
    Input tensor: [B,C,H,W] or [C,H,W], assumed in [-1, 1]
    Output numpy: uint8 image in [0,255]
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()

    if x.ndim == 4:
        x = x[0]

    x = x.permute(1, 2, 0).numpy()  # CHW -> HWC
    x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    return x


def save_image_rgb(save_path, img_rgb):
    """
    img_rgb: HWC uint8 RGB
    """
    import cv2
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


@torch.no_grad()
def evaluate(test_loader, model, device, save_vis=False, vis_dir=None, max_vis=20):
    model.eval()

    psnr_list = []
    ssim_list = []
    mse_list = []
    rmse_list = []

    vis_count = 0

    for i, (y, xa, xb, mag_factor) in enumerate(test_loader):
        y = y.to(device, non_blocking=True)
        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        mag_factor = mag_factor.to(device, non_blocking=True)

        mag_factor = mag_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        y_hat = model(xa, xb, mag_factor, 'train')

        batch_size = y.size(0)
        for b in range(batch_size):
            pred_img = tensor_to_img255(y_hat[b])
            gt_img = tensor_to_img255(y[b])
            xa_img = tensor_to_img255(xa[b])
            xb_img = tensor_to_img255(xb[b])

            mse_score = calc_mse(pred_img, gt_img)
            rmse_score = calc_rmse(pred_img, gt_img)
            psnr_score = calc_psnr(pred_img, gt_img)
            ssim_score = calc_ssim(pred_img, gt_img)

            # rmse may come back as complex because old config uses cmath.sqrt
            if hasattr(rmse_score, 'real'):
                rmse_score = float(rmse_score.real)
            else:
                rmse_score = float(rmse_score)

            mse_list.append(float(mse_score))
            rmse_list.append(float(rmse_score))
            psnr_list.append(float(psnr_score))
            ssim_list.append(float(ssim_score))

            if save_vis and vis_count < max_vis:
                os.makedirs(vis_dir, exist_ok=True)
                save_image_rgb(os.path.join(vis_dir, f'{vis_count:04d}_xa.png'), xa_img)
                save_image_rgb(os.path.join(vis_dir, f'{vis_count:04d}_xb.png'), xb_img)
                save_image_rgb(os.path.join(vis_dir, f'{vis_count:04d}_pred.png'), pred_img)
                save_image_rgb(os.path.join(vis_dir, f'{vis_count:04d}_gt.png'), gt_img)
                vis_count += 1

        if i % 50 == 0:
            print(
                f'[Test] iter={i}/{len(test_loader)} | '
                f'current PSNR={np.mean(psnr_list):.4f} | '
                f'current SSIM={np.mean(ssim_list):.6f}'
            )

    results = {
        'mse': float(np.mean(mse_list)) if len(mse_list) > 0 else 0.0,
        'rmse': float(np.mean(rmse_list)) if len(rmse_list) > 0 else 0.0,
        'psnr': float(np.mean(psnr_list)) if len(psnr_list) > 0 else 0.0,
        'ssim': float(np.mean(ssim_list)) if len(ssim_list) > 0 else 0.0,
        'num_samples': len(psnr_list)
    }
    return results


def main():
    config = Config()

    # ---------------------------------------------------------
    # Optional manual overrides for testing
    # ---------------------------------------------------------
    # You can change this to test any dataset split explicitly:
    # config.dataset_name = 'deepmag'
    # config.dataset_name = 'kubric'
    #
    # and then:
    # config.dataset_root = config.dataset_roots[config.dataset_name]
    # config.dir_test = os.path.join(config.dataset_root, 'test')

    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[Device] using {device}')

    # ---------------------------------------------------------
    # Checkpoint path
    # ---------------------------------------------------------
    checkpoint_path = os.path.join(config.save_dir, 'best_psnr.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'[Test] checkpoint not found: {checkpoint_path}\n'
            f'Please make sure training has produced best_psnr.pth '
            f'or manually edit checkpoint_path in test.py.'
        )

    print(f'[Test] loading checkpoint: {checkpoint_path}')

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = MagNet()
    model.load_state_dict(gen_state_dict(checkpoint_path))

    if torch.cuda.device_count() > 1:
        print(f'[Device] DataParallel with {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Test loader
    # ---------------------------------------------------------
    print(f'[Test] dataset_name: {config.dataset_name}')
    print(f'[Test] dir_test: {config.dir_test}')

    dataset_test = ImageFromFolderTest(
        config.dir_test,
        num_data=None,
        preprocessing=['resize']   # no train noise in testing
    )

    test_loader = data.DataLoader(
        dataset_test,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.test_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---------------------------------------------------------
    # Output dir for qualitative results
    # ---------------------------------------------------------
    vis_dir = os.path.join(config.save_dir, f'test_vis_{config.dataset_name}')

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------
    results = evaluate(
        test_loader=test_loader,
        model=model,
        device=device,
        save_vis=True,
        vis_dir=vis_dir,
        max_vis=20
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------
    print('==============================================================')
    print(f'[Test Summary] dataset = {config.dataset_name}')
    print(f'[Test Summary] split   = {config.dir_test}')
    print(f'[Test Summary] ckpt    = {checkpoint_path}')
    print(f'[Test Summary] samples = {results["num_samples"]}')
    print(f'[Test Summary] MSE     = {results["mse"]:.8f}')
    print(f'[Test Summary] RMSE    = {results["rmse"]:.8f}')
    print(f'[Test Summary] PSNR    = {results["psnr"]:.4f}')
    print(f'[Test Summary] SSIM    = {results["ssim"]:.6f}')
    print(f'[Test Summary] vis_dir = {vis_dir}')
    print('==============================================================')

    # ---------------------------------------------------------
    # Save results to txt
    # ---------------------------------------------------------
    os.makedirs(config.save_dir, exist_ok=True)
    result_txt = os.path.join(config.save_dir, f'test_result_{config.dataset_name}.txt')
    with open(result_txt, 'w') as f:
        f.write(f'dataset={config.dataset_name}\n')
        f.write(f'dir_test={config.dir_test}\n')
        f.write(f'checkpoint={checkpoint_path}\n')
        f.write(f'num_samples={results["num_samples"]}\n')
        f.write(f'mse={results["mse"]:.8f}\n')
        f.write(f'rmse={results["rmse"]:.8f}\n')
        f.write(f'psnr={results["psnr"]:.4f}\n')
        f.write(f'ssim={results["ssim"]:.6f}\n')

    print(f'[Test] results saved to: {result_txt}')


if __name__ == '__main__':
    main()