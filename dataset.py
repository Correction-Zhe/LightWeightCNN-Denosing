import os, random, cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class AugmentedTestDataset(Dataset):
    def __init__(self,
                 ori_dir, den_dir,
                 scales=(1.0,0.9),
                 patch_size=180,
                 num_per_image=3,
                 do_flip=False,
                 do_rotate=False):

        self.ori_list = sorted(os.listdir(ori_dir))
        self.den_list = sorted(os.listdir(den_dir))
        assert len(self.ori_list) == len(self.den_list)
        self.ori_dir = ori_dir
        self.den_dir = den_dir

        self.scales = scales
        self.P = patch_size
        self.n = num_per_image
        self.do_flip = do_flip
        self.do_rotate = do_rotate

    def __len__(self):
        return len(self.ori_list) * self.n

    def __getitem__(self, idx):
        img_idx = idx // self.n
        ori_path = os.path.join(self.ori_dir, self.ori_list[img_idx])
        den_path = os.path.join(self.den_dir, self.den_list[img_idx])
        ori = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        den = cv2.imread(den_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) 
        s = random.choice(self.scales)
        H, W = ori.shape
        new_size = (int(W * s), int(H * s))
        ori_s = cv2.resize(ori, new_size, interpolation=cv2.INTER_CUBIC)
        den_s = cv2.resize(den, new_size, interpolation=cv2.INTER_CUBIC)
        h, w = ori_s.shape
        if h < self.P or w < self.P:
            pad_h = max(0, self.P - h)
            pad_w = max(0, self.P - w)
            ori_s = np.pad(ori_s, ((0,pad_h),(0,pad_w)), mode='reflect')
            den_s = np.pad(den_s, ((0,pad_h),(0,pad_w)), mode='reflect')
            h, w = ori_s.shape
        x = random.randint(0, h - self.P)
        y = random.randint(0, w - self.P)
        p_ori = ori_s[x:x+self.P, y:y+self.P]
        p_den = den_s[x:x+self.P, y:y+self.P]
        if self.do_flip and random.random() < 0.5:
            p_ori = np.fliplr(p_ori)
            p_den = np.fliplr(p_den)
        if self.do_rotate:
            k = random.randint(0,3)
            p_ori = np.rot90(p_ori, k)
            p_den = np.rot90(p_den, k)
        p_ori = np.ascontiguousarray(p_ori[None])
        p_den = np.ascontiguousarray(p_den[None])

        p_ori = torch.from_numpy(p_ori).float()
        p_den = torch.from_numpy(p_den).float()

        return p_ori, p_den