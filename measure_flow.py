import os
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from config import Config
from data import get_gen_ABC, unit_postprocessing

def flow_roi_displacement(flow_net, preprocess, im1_uint8, im2_uint8, roi=None, axis=(1.0,0.0)):
    # im*_uint8: H,W,3 RGB uint8
    im1 = torch.from_numpy(im1_uint8).permute(2,0,1).float() / 255.0
    im2 = torch.from_numpy(im2_uint8).permute(2,0,1).float() / 255.0
    im1, im2 = preprocess(im1, im2)
    im1 = im1.unsqueeze(0).cuda()
    im2 = im2.unsqueeze(0).cuda()
    with torch.no_grad():
        flow = flow_net(im1, im2)[-1][0]   # (2,H,W)
    if roi is not None:
        x0,y0,x1,y1 = roi
        flow = flow[:, y0:y1, x0:x1]
    ax = torch.tensor(axis, device=flow.device, dtype=flow.dtype).view(2,1,1)
    d = (flow * ax).sum(dim=0).mean()
    return float(d.item())

def load_video_frames(config, testset):
    dl = get_gen_ABC(config, mode='test_on_'+testset)
    import cv2
    img = cv2.imread(dl.paths[0])
    vid_size = img.shape[:2][::-1]
    frames = [unit_postprocessing(dl.gen_test0()[0], vid_size=vid_size)]
    for _ in range(dl.data_len-1):
        _, C = dl.gen_test0()
        frames.append(unit_postprocessing(C, vid_size=vid_size))
    return frames

def main(testset, mag_dir, out_dir, roi=None, axis=(1.0,0.0)):
    os.makedirs(out_dir, exist_ok=True)
    config = Config()

    # 1) 原视频 frames（来自 frameA/frameC 序列）
    orig = load_video_frames(config, testset)

    # 2) 放大视频 frames（读取 test_video.py 生成的 png 序列）
    # 默认你的视频目录里有 img_{testset}_amp20 这个文件夹
    # 例如 Results_wgt_dateFD4MM/crane_crop_epoch6_amp20/img_crane_crop_amp20
    img_dir = None
    for name in os.listdir(mag_dir):
        if name.startswith("img_") and f"_{testset}_" in name:
            img_dir = os.path.join(mag_dir, name)
            break
    if img_dir is None:
        # fallback：找任意包含 png 的子目录
        for root, dirs, files in os.walk(mag_dir):
            if any(f.endswith(".png") for f in files):
                img_dir = root
                break
    if img_dir is None:
        raise FileNotFoundError(f"Cannot find magnified png frames under {mag_dir}")

    from PIL import Image
    mag_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")],
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    mag = [np.array(Image.open(os.path.join(img_dir, f)).convert("RGB")) for f in mag_files]

    # 3) RAFT
    weights = Raft_Large_Weights.DEFAULT
    flow_net = raft_large(weights=weights).cuda().eval()
    preprocess = weights.transforms()

    # 4) 位移序列
    d = []
    dp = []
    T = min(len(orig), len(mag))
    for t in range(T-1):
        d.append(flow_roi_displacement(flow_net, preprocess, orig[t], orig[t+1], roi=roi, axis=axis))
        dp.append(flow_roi_displacement(flow_net, preprocess, mag[t], mag[t+1], roi=roi, axis=axis))

    np.save(os.path.join(out_dir, "disp_orig.npy"), np.array(d, dtype=np.float32))
    np.save(os.path.join(out_dir, "disp_mag.npy"),  np.array(dp, dtype=np.float32))
    print("Saved:", os.path.join(out_dir, "disp_orig.npy"))
    print("Saved:", os.path.join(out_dir, "disp_mag.npy"))

if __name__ == "__main__":
    # 默认：axis=(1,0) 水平位移；roi=None 全图
    # 你可以把 roi 改成 (x0,y0,x1,y1)
    testset = os.environ.get("TESTSET", "crane_crop")
    mag_dir = os.environ.get("MAGDIR", "")
    out_dir = os.environ.get("OUTDIR", "flow_measure_out")
    if not mag_dir:
        raise ValueError("Please set MAGDIR env var to the magnified result folder (e.g., Results_wgt_dateFD4MM/crane_crop_epoch6_amp20)")
    main(testset, mag_dir, out_dir, roi=None, axis=(1.0,0.0))
