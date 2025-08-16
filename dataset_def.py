"""custom_weather_dataset.py – یک کلاس دیتاست مستقل برای HL‑VAE

این کد را می‌توانید مستقیماً در همان فایل `dataset_def.py` اضافه کنید (یا در فایل جداگانه‌ای ایمپورت کنید).
هیچ وابستگی به توابع یا کلاس‌های دیگر مخزن ندارد؛ فقط به `pandas`, `numpy`, `torch` نیاز دارد.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import HL_VAE.read_functions as rd


class CustomWeatherDataset(Dataset):
    """
    Dataset definition for the Health MNIST dataset when using HLVAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, types_file, true_miss_file,
                 root_dir, range_file=None, transform=None, logvar_network=False):

        if true_miss_file is not None:
            true_miss_file = os.path.join(root_dir, true_miss_file)
        train_data, types_info, miss_mask, true_miss_mask, n_samples, n_variables = \
            rd.read_data(os.path.join(root_dir, csv_file_data),
                                        os.path.join(root_dir, mask_file),
                                        true_miss_file, os.path.join(root_dir, types_file),
                         os.path.join(root_dir, range_file) if range_file is not None else None,
                         logvar_network=logvar_network)

        self.types_info = types_info

        cov_dim_ext = 0
        for t in types_info['types_dict']:
            t['dim'] = int(t['dim'])
            t['nclass'] = int(t['nclass'])
            if t['type'] == 'beta':
                cov_dim_ext += t['dim']
            else:
                cov_dim_ext += t['dim']*t['nclass']


        self.data_source = pd.DataFrame(train_data)
        self.mask_source = pd.DataFrame(miss_mask)
        self.param_mask_source = pd.DataFrame(types_info['param_miss_mask'])
        self.types_dict = types_info['types_dict']
        self.true_miss_mask = pd.DataFrame(true_miss_mask)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        if n_variables == 1296:
            self.label_source = self.label_source[self.label_source.columns.values[np.array([6, 4, 0, 5, 3, 7])]]
        # تبدیل به دادهٔ عددی و جایگزینی مقادیر غیرقابل‌تبدیل با صفر
        self.label_source = self.label_source.apply(pd.to_numeric, errors="coerce").fillna(0)
        self.root_dir = root_dir
        self.transform = transform
        self.cov_dim_ext = cov_dim_ext
        self.n_samples = n_samples
        self.n_variables = n_variables

    def __len__(self):
        return len(self.data_source)


    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        covariate = np.array([digit])

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')
        true_mask = self.true_miss_mask.iloc[idx, :]
        true_mask = np.array([true_mask], dtype='uint8')
        param_mask = self.param_mask_source.iloc[idx, :]
        param_mask = np.array([param_mask], dtype='uint8')

        label = self.label_source.iloc[idx, :]
        label = pd.to_numeric(label, errors='coerce')
        label = np.nan_to_num(label.to_numpy(dtype=np.float32))
        label = torch.from_numpy(label)

        if self.transform:
            covariate = self.transform(covariate)
        else:
            covariate = torch.from_numpy(covariate)

        sample = {
            'digit': covariate,
            'label': label,
            'idx': idx,
            'mask': mask,
            'param_mask': param_mask,
            'true_mask': true_mask,
        }
        return sample
