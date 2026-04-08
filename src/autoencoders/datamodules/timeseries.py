from torch.utils.data import Dataset
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, stride=1, downsample_factor=1):
        self.data = data
        self.seq_length = seq_length
        self.stride = stride
        self.downsample_factor = downsample_factor

        # Effective temporal span covered by one strided sequence
        self.eff_len = (seq_length - 1) * stride + 1

        # Number of valid starting indices along time
        self.t_len = self.data.shape[1] - self.eff_len + 1

    def __len__(self):
        return self.t_len * self.data.shape[0]

    def __getitem__(self, _idx):
        b = _idx // self.t_len
        idx = _idx % self.t_len

        widths = self.data.shape[3:]   # spatial dims
        chan = self.data.shape[2]

        # Random spatial crop
        dwidths = [w // self.downsample_factor for w in widths]
        rand_start = [
            torch.randint(0, w - dw + 1, (1,)).item()
            for w, dw in zip(widths, dwidths)
        ]
        spatial_slices = [
            slice(start, start + dw)
            for start, dw in zip(rand_start, dwidths)
        ]

        # Pure slicing with stride (no advanced indexing)
        x = self.data[
            b,
            idx : idx + self.eff_len : self.stride,
            :,
            *spatial_slices
        ]

        return x

    def shape(self):
        return (len(self), self.seq_length, *self.data.shape[2:])

class ConditionalDataset(Dataset):
    def __init__(self, datasets, labels):
        """
        datasets: list of Dataset objects (e.g., TimeSeriesDataset instances)
        labels: list of labels (same length)
        """
        assert len(datasets) == len(labels)

        self.datasets = datasets
        self.labels = labels

        self.lengths = [len(d) for d in datasets]
        self.cum_lengths = torch.cumsum(
            torch.tensor(self.lengths), dim=0
        )

    def __len__(self):
        return int(self.cum_lengths[-1])

    def _get_dataset_index(self, idx):
        return int(torch.searchsorted(self.cum_lengths, idx, right=True))

    def __getitem__(self, idx):
        d_idx = self._get_dataset_index(idx)

        prev_cum = 0 if d_idx == 0 else self.cum_lengths[d_idx - 1]
        local_idx = idx - prev_cum

        x = self.datasets[d_idx][local_idx]
        label = self.labels[d_idx]

        return label, x
    
def conditional_collate(batch):
    labels, xs = zip(*batch)
    xs = torch.stack(xs, dim=0)
    return labels, xs