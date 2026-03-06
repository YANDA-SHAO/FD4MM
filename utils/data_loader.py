import os
import numpy as np
import torch
import cv2

from torchvision.datasets.folder import ImageFolder, default_loader


def add_gaussian_noise(image, std=0.1):
    """
    image: numpy array in [-1, 1]
    """
    noise = np.random.normal(0.0, std, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, -1.0, 1.0)


def add_poisson_like_noise(image, lam=0.3):
    """
    image: numpy array in [-1, 1]
    Keep the original repo's style, but clip at the end.
    """
    n = np.random.normal(0.0, 1.0, image.shape).astype(np.float32)
    n_str = np.sqrt(image + 1.0) / np.sqrt(127.5)
    noisy_image = image + lam * n * n_str
    return np.clip(noisy_image, -1.0, 1.0)


def preprocess_image(sample, preproc=None, resize_to=(384, 384), is_target=False):
    """
    sample: HWC, RGB, uint8/float numpy array
    preproc: list, e.g. ['resize', 'gaussian'] or ['resize', 'poisson']
    is_target: target image (amplified) should usually not receive input noise
    """
    if preproc is None:
        preproc = []

    # Ensure numpy array
    sample = np.asarray(sample)

    # Resize first
    if 'resize' in preproc:
        sample = cv2.resize(sample, resize_to, interpolation=cv2.INTER_LANCZOS4)

    # Normalize to [-1, 1]
    sample = sample.astype(np.float32) / 127.5 - 1.0

    # Add noise only to inputs, not to target
    if not is_target:
        if 'gaussian' in preproc:
            sample = add_gaussian_noise(sample, std=0.1)
        if 'poisson' in preproc:
            sample = add_poisson_like_noise(sample, lam=0.3)

    # Final safety clip
    sample = np.clip(sample, -1.0, 1.0)

    # HWC -> CHW
    sample = torch.from_numpy(sample).float().permute(2, 0, 1)
    return sample


class BaseImageFromFolder(ImageFolder):
    def __init__(
        self,
        root,
        mf_file='train_mf.txt',
        num_data=None,
        preprocessing=None,
        resize_to=(384, 384),
        loader=default_loader,
    ):
        self.root = root
        self.loader = loader
        self.preproc = preprocessing if preprocessing is not None else []
        self.resize_to = resize_to

        mf_path = os.path.join(root, mf_file)
        if not os.path.exists(mf_path):
            raise FileNotFoundError(f'[DataLoader] mf file not found: {mf_path}')

        mag = np.loadtxt(mf_path)
        if np.ndim(mag) == 0:
            mag = np.array([float(mag)], dtype=np.float32)
        else:
            mag = np.asarray(mag, dtype=np.float32)

        dir_amp = os.path.join(root, 'amplified')
        dir_A = os.path.join(root, 'frameA')
        dir_B = os.path.join(root, 'frameB')

        for d in [dir_amp, dir_A, dir_B]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f'[DataLoader] required folder not found: {d}')

        files_amp = sorted([f for f in os.listdir(dir_amp) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        files_A = sorted([f for f in os.listdir(dir_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        files_B = sorted([f for f in os.listdir(dir_B) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        available = min(len(files_amp), len(files_A), len(files_B), len(mag))

        if available == 0:
            raise ValueError(f'[DataLoader] no valid samples found under: {root}')

        if num_data is None:
            num_data = available
        else:
            num_data = min(int(num_data), available)

        # Build samples by numeric filename convention: %06d.png
        self.samples = [
            (
                os.path.join(dir_amp, f'{i+1:06d}.png'),
                os.path.join(dir_A, f'{i+1:06d}.png'),
                os.path.join(dir_B, f'{i+1:06d}.png'),
                mag[i],
            )
            for i in range(num_data)
        ]
        self.imgs = self.samples

        print(f'[DataLoader] root={root}')
        print(f'[DataLoader] mf_file={mf_file}')
        print(f'[DataLoader] available={available}, using={num_data}')
        print(f'[DataLoader] preprocessing={self.preproc}, resize_to={self.resize_to}')

    def __getitem__(self, index):
        pathAmp, pathA, pathB, target = self.samples[index]

        sampleAmp = np.array(self.loader(pathAmp))
        sampleA = np.array(self.loader(pathA))
        sampleB = np.array(self.loader(pathB))

        sampleAmp = preprocess_image(
            sampleAmp,
            preproc=self.preproc,
            resize_to=self.resize_to,
            is_target=True
        )
        sampleA = preprocess_image(
            sampleA,
            preproc=self.preproc,
            resize_to=self.resize_to,
            is_target=False
        )
        sampleB = preprocess_image(
            sampleB,
            preproc=self.preproc,
            resize_to=self.resize_to,
            is_target=False
        )

        target = torch.tensor(target, dtype=torch.float32)
        return sampleAmp, sampleA, sampleB, target

    def __len__(self):
        return len(self.samples)


class ImageFromFolder(BaseImageFromFolder):
    def __init__(
        self,
        root,
        num_data=None,
        preprocessing=None,
        transform=None,
        target_transform=None,
        loader=default_loader
    ):
        super().__init__(
            root=root,
            mf_file='train_mf.txt',
            num_data=num_data,
            preprocessing=preprocessing,
            resize_to=(384, 384),
            loader=loader,
        )


class ImageFromFolderTest(BaseImageFromFolder):
    def __init__(
        self,
        root,
        num_data=None,
        preprocessing=None,
        transform=None,
        target_transform=None,
        loader=default_loader
    ):
        super().__init__(
            root=root,
            mf_file='test_mf.txt',
            num_data=num_data,
            preprocessing=preprocessing,
            resize_to=(384, 384),
            loader=loader,
        )


class ImageFromFolderVal(BaseImageFromFolder):
    def __init__(
        self,
        root,
        num_data=None,
        preprocessing=None,
        transform=None,
        target_transform=None,
        loader=default_loader
    ):
        super().__init__(
            root=root,
            mf_file='test_mf.txt',
            num_data=num_data,
            preprocessing=preprocessing,
            resize_to=(384, 384),
            loader=loader,
        )


if __name__ == '__main__':
    dataset = ImageFromFolder(
        './../data/train',
        num_data=100,
        preprocessing=['resize', 'gaussian']
    )

    imageAmp, imageA, imageB, mag = dataset[0]
    print('imageAmp:', imageAmp.shape)
    print('imageA  :', imageA.shape)
    print('imageB  :', imageB.shape)
    print('mag     :', mag)
