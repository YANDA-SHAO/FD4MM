import os
import cv2
import torch
import numpy as np

from config import Config
from data import get_gen_ABC
from magnet_FD4MM import MagNet
from callbacks import gen_state_dict

# =========================
# Settings
# =========================
AMP = 20
TESTSET = "baby"

WEIGHTS_OLD = "weights_dateFD4MM_kubric_ft/magnet_epoch1_loss8.17e-01.pth"
WEIGHTS_NEW = "weights_dateFD4MM_kubric_ft/magnet_epoch0_loss3.46e-01.pth"

OUT_VIDEO = f"compare_{TESTSET}_orig_old_new_amp{AMP}.mp4"
FPS = 30

# =========================
# helper
# =========================
def tensor_to_img(x):
    x = x.detach().cpu()

    if x.dim() == 4:
        x = x[0]

    x = (x + 1) / 2.0
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255).astype(np.uint8)
    return x


def put_text(img, text):
    img = img.copy()
    cv2.putText(
        img,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def load_model(weights_path):
    state_dict = gen_state_dict(weights_path)
    model = MagNet().cuda()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded: {weights_path}")
    return model


# =========================
# load config / data
# =========================
config = Config()
dl = get_gen_ABC(config, mode=f"test_on_{TESTSET}")
print("Number of test image couples:", dl.data_len)

# first pair to get size
A0, _ = dl.gen_test0()
h = A0.shape[2]
w = A0.shape[3]

# reset loader so we start from the beginning
dl = get_gen_ABC(config, mode=f"test_on_{TESTSET}")

# =========================
# load models
# =========================
G_old = load_model(WEIGHTS_OLD)
G_new = load_model(WEIGHTS_NEW)

amp = torch.tensor([AMP]).cuda().view(1, 1, 1, 1)

writer = cv2.VideoWriter(
    OUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (w * 3, h)
)

# =========================
# build sequence
# Use first frame as reference, same as your test script
# =========================
frame0 = None

for i in range(dl.data_len):
    batch_A, batch_B = dl.gen_test0()

    if frame0 is None:
        frame0 = batch_A

    with torch.no_grad():
        y_old = G_old(frame0, batch_B, amp, mode="evaluate")[0]
        y_new = G_new(frame0, batch_B, amp, mode="evaluate")[0]

    orig = tensor_to_img(batch_B)
    oldm = tensor_to_img(y_old)
    newm = tensor_to_img(y_new)

    orig = put_text(orig, "Original")
    oldm = put_text(oldm, "Old model")
    newm = put_text(newm, "New model")

    frame = np.concatenate([orig, oldm, newm], axis=1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

writer.release()
print("Saved video:", OUT_VIDEO)