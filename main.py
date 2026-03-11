import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import wandb
from tensorboardX import SummaryWriter

from config import Config, calc_psnr, calc_ssim
from utils.data_loader import ImageFromFolder, ImageFromFolderVal
from utils.avgMeter import AverageMeter
from utils.utils import ContrastLoss_Ori, EdgeLoss, CharbonnierLoss
from magnet_FD4MM import MagNet
from callbacks import save_model, gen_state_dict


writer = SummaryWriter(log_dir="./log_magnet_FoCR")


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


def save_checkpoint(state_dict, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(state_dict, save_path)
    print(f'[Checkpoint] Saved: {save_path}')


def main():
    config = Config()

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        name=config.wandb_run_name,
        config={
            "exp_name": config.exp_name,
            "dataset": config.dataset_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "workers": config.workers,
            "numdata": config.numdata,
            "hp_tag": config.hp_tag,
            "train_dir": config.dir_train,
            "val_dir": config.dir_val,
            "test_dir": config.dir_test,
            "save_best_by": config.save_best_by,
        }
    )

    # =========================================================
    # Seed
    # =========================================================
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    # =========================================================
    # Device and model
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] using {device}")

    magnet = MagNet()
    start_epoch = 0

    if config.pretrained_weights:
        print(f"=> loading checkpoint '{config.pretrained_weights}'")
        magnet.load_state_dict(gen_state_dict(config.pretrained_weights))
    else:
        print("=> training from scratch")

    if torch.cuda.device_count() > 1:
        print(f"[Device] DataParallel with {torch.cuda.device_count()} GPUs")
        magnet = nn.DataParallel(magnet)

    magnet = magnet.to(device)

    # =========================================================
    # Metrics / losses
    # =========================================================
    criterion_char = CharbonnierLoss().to(device)
    criterion_cr = ContrastLoss_Ori().to(device)
    criterion_edge = EdgeLoss().to(device)

    optimizer = optim.Adam(
        magnet.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6
    )

    os.makedirs(config.save_dir, exist_ok=True)
    print('[Path] Save_dir:', config.save_dir)
    print('[Path] Train dir:', config.dir_train)
    print('[Path] Val dir  :', config.dir_val)
    print('[Path] Test dir :', config.dir_test)

    # =========================================================
    # Data loaders
    # =========================================================
    dataset_train = ImageFromFolder(
        config.dir_train,
        num_data=config.numdata,
        preprocessing=config.preproc
    )
    train_loader = data.DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False,
    )

    dataset_val = ImageFromFolderVal(
        config.dir_val,
        num_data=config.val_subset,
        preprocessing=['resize']  # val/test should not add train noise
    )
    val_loader = data.DataLoader(
        dataset_val,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.test_workers,
        pin_memory=True,
        drop_last=False,
    )

    # =========================================================
    # Summary
    # =========================================================
    print('===================================================================')
    print('PyTorch Version: ', torch.__version__)
    print('===================================================================')
    print('Network parameters {}'.format(sum(p.numel() for p in magnet.parameters())))
    print('Trainable network parameters {}'.format(sum(p.numel() for p in magnet.parameters() if p.requires_grad)))

    train_losses, train_losses_recon, train_losses_edge, train_losses_cr = [], [], [], []
    global_step = 0

    best_val_loss = float('inf')
    best_val_psnr = -float('inf')
    best_val_ssim = -float('inf')
    best_epoch = -1

    # =========================================================
    # Training loop
    # =========================================================
    for epoch in range(start_epoch, config.epochs):
        train_loss, train_loss_recon, train_loss_edge, train_loss_cr, global_step, \
        best_val_loss, best_val_psnr, best_val_ssim, best_epoch = train(
            train_loader, magnet, criterion_char, criterion_edge, criterion_cr,
            optimizer, epoch, device, config, val_loader, global_step,
            best_val_loss, best_val_psnr, best_val_ssim, best_epoch
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f'[Train] Epoch {epoch}: lr = {current_lr:.8f}')
        wandb.log({
            "lr": current_lr,
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_loss_recon": train_loss_recon,
            "train/epoch_loss_edge": train_loss_edge,
            "train/epoch_loss_CR": train_loss_cr,
        })

        train_losses.append(train_loss)
        train_losses_recon.append(train_loss_recon)
        train_losses_edge.append(train_loss_edge)
        train_losses_cr.append(train_loss_cr)

        writer.add_scalar('train/loss_epoch', train_loss, epoch)
        writer.add_scalar('train/loss_recon_epoch', train_loss_recon, epoch)
        writer.add_scalar('train/loss_edge_epoch', train_loss_edge, epoch)
        writer.add_scalar('train/loss_CR_epoch', train_loss_cr, epoch)
        writer.add_scalar('train/lr', current_lr, epoch)

        # -----------------------------
        # Validation
        # -----------------------------
        pass

        # -----------------------------
        # Save latest / periodic
        # -----------------------------
        state_dict_to_save = magnet.module.state_dict() if isinstance(magnet, nn.DataParallel) else magnet.state_dict()

        save_checkpoint(state_dict_to_save, config.save_dir, 'latest.pth')

        if (epoch + 1) % config.save_every_epoch == 0:
            # keep original callback behaviour too
            save_model(state_dict_to_save, train_losses, config.save_dir, epoch)

        scheduler.step()

    print('===================================================================')
    print(f'[Done] best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}, best_val_psnr={best_val_psnr:.4f}, best_val_ssim={best_val_ssim:.6f}')
    print('===================================================================')

    wandb.finish()


def train(loader, model, criterion_char, criterion_edge, criterion_cr, optimizer, epoch, device, config, val_loader, global_step, best_val_loss, best_val_psnr, best_val_ssim, best_epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_edge = AverageMeter()
    losses_cr = AverageMeter()

    model.train()
    end = time.time()

    print_freq = max(1, len(loader) // config.num_print_per_epoch)

    for i, (y, xa, xb, mag_factor) in enumerate(loader):
        if i > 50:   # 只跑50个iteration
            print("DEBUG STOP: iteration test finished")
        break
        
        y = y.to(device, non_blocking=True)
        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        mag_factor = mag_factor.to(device, non_blocking=True)

        data_time.update(time.time() - end)

        mag_factor = mag_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        y_hat = model(xa, xb, mag_factor, 'train')

        loss_recon = criterion_char(y_hat, y)
        loss_edge = criterion_edge(y_hat, y)
        loss_cr = 0.1 * criterion_cr(y_hat, y, xb)

        loss = loss_recon + loss_edge + loss_cr

        losses.update(loss.item(), y.size(0))
        losses_recon.update(loss_recon.item(), y.size(0))
        losses_edge.update(loss_edge.item(), y.size(0))
        losses_cr.update(loss_cr.item(), y.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # 每2000 iteration 做一次 validation
        if global_step % 20 == 0:

            print(f"\n[VAL Trigger] step={global_step}")

            val_loss, val_loss_recon, val_loss_edge, val_loss_cr, val_psnr, val_ssim = validate(
                val_loader, model, criterion_char, criterion_edge, criterion_cr,
                epoch, device, config
            )

            print(
                f'[VAL] step={global_step} '
                f'loss={val_loss:.6f} '
                f'psnr={val_psnr:.4f} '
                f'ssim={val_ssim:.6f}'
            )

            wandb.log({
                "val/loss": val_loss,
                "val/psnr": val_psnr,
                "val/ssim": val_ssim,
                "step": global_step
            })
            
            # save best checkpoint based on step-based validation
            is_best = False
            if config.save_best_by == 'loss':
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_psnr = val_psnr
                    best_val_ssim = val_ssim
                    best_epoch = epoch
                    is_best = True
            else:
                if val_psnr > best_val_psnr:
                    best_val_loss = val_loss
                    best_val_psnr = val_psnr
                    best_val_ssim = val_ssim
                    best_epoch = epoch
                    is_best = True

            if is_best:
                state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                save_checkpoint(state_dict_to_save, config.save_dir, 'best_psnr.pth')
                print(
                    f'[Best] Updated at step={global_step}, epoch={epoch}, '
                    f'val_psnr={best_val_psnr:.4f}, '
                    f'val_ssim={best_val_ssim:.6f}, '
                    f'val_loss={best_val_loss:.6f}'
                )
            model.train()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            wandb.log({
                "train/loss_iter": loss.item(),
                "train/loss_recon_iter": loss_recon.item(),
                "train/loss_edge_iter": loss_edge.item(),
                "train/loss_CR_iter": loss_cr.item(),
                "epoch": epoch,
                "iter": i
            })

        if i % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'LossRe {loss_recon.val:.4f} ({loss_recon.avg:.4f})\t'
                'LossEdge {loss_edge.val:.4f} ({loss_edge.avg:.4f})\t'
                'LossCR {loss_cr.val:.4f} ({loss_cr.avg:.4f})'.format(
                    epoch, i, len(loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, loss_recon=losses_recon,
                    loss_edge=losses_edge, loss_cr=losses_cr
                )
            )

    return (
        losses.avg,
        losses_recon.avg,
        losses_edge.avg,
        losses_cr.avg,
        global_step,
        best_val_loss,
        best_val_psnr,
        best_val_ssim,
        best_epoch
    )

@torch.no_grad()
def validate(loader, model, criterion_char, criterion_edge, criterion_cr, epoch, device, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_edge = AverageMeter()
    losses_cr = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    model.eval()
    end = time.time()

    for i, (y, xa, xb, mag_factor) in enumerate(loader):
        y = y.to(device, non_blocking=True)
        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        mag_factor = mag_factor.to(device, non_blocking=True)

        mag_factor = mag_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        y_hat = model(xa, xb, mag_factor, 'train')

        loss_recon = criterion_char(y_hat, y)
        loss_edge = criterion_edge(y_hat, y)
        loss_cr = 0.1 * criterion_cr(y_hat, y, xb)
        loss = loss_recon + loss_edge + loss_cr

        losses.update(loss.item(), y.size(0))
        losses_recon.update(loss_recon.item(), y.size(0))
        losses_edge.update(loss_edge.item(), y.size(0))
        losses_cr.update(loss_cr.item(), y.size(0))

        # compute PSNR / SSIM sample by sample
        bsz = y.size(0)
        for b in range(bsz):
            pred_img = tensor_to_img255(y_hat[b])
            gt_img = tensor_to_img255(y[b])

            psnr_score = calc_psnr(pred_img, gt_img)
            ssim_score = calc_ssim(pred_img, gt_img)

            psnr_meter.update(float(psnr_score), 1)
            ssim_meter.update(float(ssim_score), 1)

        batch_time.update(time.time() - end)
        end = time.time()

    return (
        losses.avg,
        losses_recon.avg,
        losses_edge.avg,
        losses_cr.avg,
        psnr_meter.avg,
        ssim_meter.avg
    )


if __name__ == '__main__':
    main()