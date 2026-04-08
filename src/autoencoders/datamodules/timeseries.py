from torch.utils.data import Dataset
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, downsample_factor=1):
        self.data = data
        self.seq_length = seq_length
        self.downsample_factor = downsample_factor
        
        self.t_len = self.data.shape[1] - self.seq_length

    def __len__(self):
        return self.t_len * self.data.shape[0]

    def __getitem__(self, _idx):
        b = _idx // self.t_len
        idx = _idx % self.t_len

        widths = self.data.shape[3:]  # spatial dims
        chan = self.data.shape[2]

        dwidths = [w // self.downsample_factor for w in widths]
        rand_start = [
            torch.randint(0, w - dw + 1, (1,)).item()
            for w, dw in zip(widths, dwidths)
        ]
        _slices = [
            slice(_start, _start + dw)
            for _start, dw in zip(rand_start, dwidths)
        ]

        seq_slices = [
            slice(b, b + 1),
            slice(idx, idx + self.seq_length),
            slice(0, chan),
            *_slices
        ]

        x = self.data[seq_slices][0, ...]  # single sequence

        return x

    def shape(self):
        return (len(self), self.seq_length, *self.data.shape[2:])