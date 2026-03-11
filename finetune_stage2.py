# finetune_stage2_debug.py
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from magnet_FD4MM import MagNet
from callbacks import gen_state_dict

from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


# -------------------------
# Debug helpers
# -------------------------
def cuda_mem(tag=""):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / (1024 ** 2)
    r = torch.cuda.memory_reserved() / (1024 ** 2)
    p = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"[CUDA] {tag} allocated={a:.1f}MB reserved={r:.1f}MB peak={p:.1f}MB")


def tstats(name, x: torch.Tensor, max_elems=3):
    if x is None:
        print(f"[{name}] None")
        return
    with torch.no_grad():
        mn = float(x.min().item()) if x.numel() else float("nan")
        mx = float(x.max().item()) if x.numel() else float("nan")
        me = float(x.mean().item()) if x.numel() else float("nan")
        sd = float(x.std().item()) if x.numel() else float("nan")
    print(f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} dev={x.device} "
          f"min={mn:.4f} max={mx:.4f} mean={me:.4f} std={sd:.4f}")


def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def parse_tuple_floats(s, n=2, default=None):
    try:
        xs = [float(x.strip()) for x in s.split(",")]
        if len(xs) != n:
            raise ValueError
        return tuple(xs)
    except Exception:
        return default


def parse_tuple_ints(s, n=4, default=None):
    try:
        xs = [int(x.strip()) for x in s.split(",")]
        if len(xs) != n:
            raise ValueError
        return tuple(xs)
    except Exception:
        return default


# -------------------------
# Signal + spectral losses
# -------------------------
def roi_mean_projection(flow_2hw, axis=(1.0, 0.0), roi=None):
    # flow_2hw: (2,H,W)
    assert flow_2hw.dim() == 3 and flow_2hw.shape[0] == 2, f"flow shape must be (2,H,W), got {tuple(flow_2hw.shape)}"
    if roi is not None:
        x0, y0, x1, y1 = roi
        flow_2hw = flow_2hw[:, y0:y1, x0:x1]
    ax = flow_2hw.new_tensor(axis).view(2, 1, 1)
    return (flow_2hw * ax).sum(dim=0).mean()  # scalar


def stft_complex(x_1d, n_fft=256, hop=64):
    win = torch.hann_window(n_fft, device=x_1d.device, dtype=x_1d.dtype)
    return torch.stft(
        x_1d, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=win, center=False, return_complex=True
    )  # (F, TT)


def band_mask(freqs, band):
    f0, f1 = band
    return (freqs >= f0) & (freqs <= f1)


def spectral_losses(d, dp, fs, m, band, n_fft=256, hop=64, eps=1e-6):
    # d, dp: (T,) tensors
    X = stft_complex(d, n_fft=n_fft, hop=hop)
    Xp = stft_complex(dp, n_fft=n_fft, hop=hop)

    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(d.device)
    inb = band_mask(freqs, band)
    outb = ~inb

    mag = X.abs() + eps
    magp = Xp.abs() + eps

    lam = (torch.log(magp) - math.log(m) - torch.log(mag))
    L_amp = (lam[inb, :].pow(2)).mean()

    phase = torch.angle(X)
    phasep = torch.angle(Xp)
    L_phase = (1.0 - torch.cos(phasep[inb, :] - phase[inb, :])).mean()

    L_out = (magp[outb, :].pow(2)).mean()

    return L_amp, L_phase, L_out


# -------------------------
# RAFT input utilities (grad-friendly)
# -------------------------
def _to_4d(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3:
        return img.unsqueeze(0)
    if img.dim() == 4:
        return img
    raise RuntimeError(f"Unexpected dim={img.dim()}, shape={tuple(img.shape)}")


def raft_flow(im1, im2, need_grad: bool):
    im1 = _to_4d(im1)
    im2 = _to_4d(im2)

    # FD4MM output is usually in [-1,1]
    im1 = ((im1 + 1.0) * 0.5).clamp(0, 1)
    im2 = ((im2 + 1.0) * 0.5).clamp(0, 1)

    im1t, im2t = preprocess(im1, im2)

    if need_grad:
        flow = Fnet(im1t, im2t)[-1][0]  # (2,H,W)
    else:
        with torch.no_grad():
            flow = Fnet(im1t, im2t)[-1][0]
    return flow


def _pad_to_multiple(img4d: torch.Tensor, mult=8):
    # pad H,W to multiple of mult (RAFT likes multiples of 8)
    n, c, h, w = img4d.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0:
        return img4d, (0, 0, 0, 0)
    # pad order: (left, right, top, bottom)
    pad = (0, pw, 0, ph)
    img4d = F.pad(img4d, pad, mode="replicate")
    return img4d, pad


def _unpad_flow(flow2hw: torch.Tensor, pad):
    # pad: (left,right,top,bottom)
    l, r, t, b = pad
    if (r == 0) and (b == 0):
        return flow2hw
    h = flow2hw.shape[1] - b
    w = flow2hw.shape[2] - r
    return flow2hw[:, :h, :w]


def normalize_01_for_raft(x):
    """
    FD4MM tensors are usually in [-1,1].
    RAFT weights expect float in [0,1] then normalized by mean/std.
    """
    x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def normalize_meanstd(x, mean, std):
    # x: (N,C,H,W)
    mean = x.new_tensor(mean).view(1, 3, 1, 1)
    std = x.new_tensor(std).view(1, 3, 1, 1)
    return (x - mean) / std


# -------------------------
# Main
# -------------------------
def main():
    torch.backends.cudnn.benchmark = True

    config = Config()

    testset = os.environ.get("TESTSET", "baby")
    amp = float(os.environ.get("AMP", "20"))
    fs = float(os.environ.get("FPS", "30"))

    band = parse_tuple_floats(os.environ.get("BAND", "0.5,6.0"), 2, default=(0.5, 6.0))
    iters = int(os.environ.get("ITERS", "50"))
    lr = float(os.environ.get("LR", "1e-5"))
    T_clip = int(os.environ.get("TCLIP", "16"))

    n_fft = int(os.environ.get("NFFT", "256"))
    hop = int(os.environ.get("HOP", "64"))

    roi = parse_tuple_ints(os.environ.get("ROI", ""), 4, default=None) if os.environ.get("ROI", "") else None
    axis = parse_tuple_floats(os.environ.get("AXIS", "1,0"), 2, default=(1.0, 0.0))

    weights_path = os.environ.get("MAG_WEIGHTS", "weights_dateFD4MM/magnet_epoch6_loss4.75e-01.pth")
    out_w = os.environ.get("OUT_WEIGHTS", f"weights_dateFD4MM/magnet_finetune_{testset}.pth")
    use_amp = int(os.environ.get("USE_AMP", "1")) == 1

    print("=== Stage2 Debug Finetune ===")
    print("TESTSET:", testset, "AMP:", amp, "FPS:", fs, "BAND:", band)
    print("ITERS:", iters, "LR:", lr, "TCLIP:", T_clip, "NFFT:", n_fft, "HOP:", hop)
    print("ROI:", roi, "AXIS:", axis)
    print("MAG_WEIGHTS:", weights_path)
    print("OUT_WEIGHTS:", out_w)
    print("USE_AMP:", use_amp)

    # ---- load magnifier
    state_dict = gen_state_dict(weights_path)
    G = MagNet().cuda()
    G.load_state_dict(state_dict, strict=False)
    G.train()

    # 先全解冻（你后面再精确只训 manipulator / recouple）
    set_requires_grad(G, True)
    params = [p for p in G.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in params))

    optG = torch.optim.Adam(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- load RAFT small
    w = Raft_Small_Weights.DEFAULT
    Fnet = raft_small(weights=w).cuda().eval()
    set_requires_grad(Fnet, False)
    w = Raft_Small_Weights.DEFAULT
    preprocess = w.transforms()

    # ---- data loader (your existing)
    from data import get_gen_ABC
    import cv2

    dl = get_gen_ABC(config, mode='test_on_' + testset)
    print("Data len:", dl.data_len, "batch_size_test:", dl.batch_size)
    assert dl.data_len >= T_clip, f"Not enough frames for TCLIP={T_clip}, data_len={dl.data_len}"

    img0 = cv2.imread(dl.paths[0])
    assert img0 is not None, f"Failed to read {dl.paths[0]}"
    vid_size = img0.shape[:2][::-1]
    print("vid_size:", vid_size)

    def raft_flow(im1, im2, need_grad: bool):
        im1 = _to_4d(im1)
        im2 = _to_4d(im2)

        # FD4MM: [-1,1] -> [0,1]
        im1 = ((im1 + 1.0) * 0.5).clamp(0, 1)
        im2 = ((im2 + 1.0) * 0.5).clamp(0, 1)

        # Use torchvision weight-specific transforms
        im1t, im2t = preprocess(im1, im2)

        # pad to multiple of 8 (optional but good)
        im1t, pad = _pad_to_multiple(im1t, mult=8)
        im2t, _ = _pad_to_multiple(im2t, mult=8)

        if need_grad:
            flow = Fnet(im1t, im2t)[-1][0]  # (2,H,W)
        else:
            with torch.no_grad():
                flow = Fnet(im1t, im2t)[-1][0]

        flow = _unpad_flow(flow, pad)
        return flow

    # ---- train loop
    loss_log = []
    torch.cuda.reset_peak_memory_stats()
    cuda_mem("start")

    for it in range(iters):
        try:
            # optionally: reset anchor each iter for deterministic clip
            dl.anchor = 0

            # build orig clip as list of tensors
            orig = []
            for k in range(T_clip):
                A, C = dl.gen_test0()  # (1,3,H,W)
                if k == 0:
                    orig.append(A)
                else:
                    orig.append(C)

            # sanity
            assert len(orig) == T_clip
            assert orig[0].dim() == 4 and orig[0].shape[1] == 3

            if it == 0:
                tstats("orig0", orig[0])

            amp_factor = torch.tensor([amp], device="cuda", dtype=torch.float32).view(1, 1, 1, 1)

            # forward magnifier
            mag = [orig[0]]
            with torch.cuda.amp.autocast(enabled=use_amp):
                for t in range(1, T_clip):
                    y_hats = G(orig[0], orig[t], amp_factor, mode='evaluate')
                    y = y_hats[0]
                    mag.append(y)

            if it == 0:
                tstats("mag1", mag[1])
                cuda_mem("after G forward")

            # compute displacement series
            d_list, dp_list = [], []
            for t in range(T_clip - 1):
                # original: no grad needed
                flow_o = raft_flow(orig[t], orig[t + 1], need_grad=False)
                # magnified: need grad to magnifier input
                flow_m = raft_flow(mag[t], mag[t + 1], need_grad=True)

                d_list.append(roi_mean_projection(flow_o, axis=axis, roi=roi))
                dp_list.append(roi_mean_projection(flow_m, axis=axis, roi=roi))

            d = torch.stack(d_list)  # (T-1,)
            dp = torch.stack(dp_list)  # (T-1,)

            if it == 0:
                tstats("d", d)
                tstats("dp", dp)

            # spectral losses (d is const-like, dp carries grad)
            with torch.cuda.amp.autocast(enabled=use_amp):
                L_amp, L_phase, L_out = spectral_losses(
                    d.detach(), dp, fs=fs, m=amp, band=band, n_fft=n_fft, hop=hop
                )

                # tiny image stabilizer
                L_img = 0.0
                for t in range(1, T_clip):
                    L_img = L_img + (mag[t] - orig[t]).abs().mean()
                L_img = L_img / (T_clip - 1)

                loss = 1.0 * L_amp + 0.2 * L_phase + 0.05 * L_out + 0.001 * L_img

            optG.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optG)
            scaler.update()

            if (it + 1) % 1 == 0:
                print(f"it {it + 1}/{iters} | Lamp {L_amp.item():.4e} "
                      f"Lph {L_phase.item():.4e} Lout {L_out.item():.4e} "
                      f"Limg {float(L_img):.4e} | total {loss.item():.4e}")
                cuda_mem(f"iter {it + 1}")

            loss_log.append([
                float(L_amp.item()),
                float(L_phase.item()),
                float(L_out.item()),
                float(L_img),
                float(loss.item())
            ])

        except RuntimeError as e:
            print(f"[ERROR] RuntimeError at iter {it + 1}: {e}")
            cuda_mem("on error")
            raise

    # ---- save
    out_dir = os.path.dirname(out_w)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Saving to:", out_w)
    torch.save(G.state_dict(), out_w)
    np.savetxt(out_w + ".log.txt", np.array(loss_log))
    print("Saved finetuned weights:", out_w, "exists?", os.path.exists(out_w))


if __name__ == "__main__":
    main()