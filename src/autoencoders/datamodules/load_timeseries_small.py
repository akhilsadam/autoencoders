# import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import inspect

def filter_kwargs(dict_to_filter, thing_with_kwargs):
    # sig = inspect.signature(thing_with_kwargs)
    # filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filter_keys = ['num_workers', 'batch_size', 'shuffle', 'pin_memory']
    filter_keys = [key for key in filter_keys if key in dict_to_filter]
    filtered_dict = {filter_key:dict_to_filter[filter_key] for filter_key in filter_keys}
    return filtered_dict

# default parameters
default_memory_length = 20
default_predict_length = 20

import logging
logger = logging.getLogger('load')
# from utilities.gpu import device

def collate_fn(batch):
  return [torch.stack([x['memory'] for x in batch], dim=0),
      torch.stack([x['predict'] for x in batch], dim=0)
      ]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, memory_length, predict_length, downsample_factor=1):
        self.data = data
        self.memory_length = memory_length
        self.predict_length = predict_length
        self.downsample_factor = downsample_factor
        
        self.t_len = self.data.shape[1] - self.memory_length - self.predict_length
        

    def __len__(self):
        return self.t_len * self.data.shape[0]

    def __getitem__(self, _idx):
        
        b = _idx // self.t_len
        idx = _idx % self.t_len

        widths = self.data.shape[3:] # skip batch, time, and channel
        chan = self.data.shape[2]
        dwidths = [w//self.downsample_factor for w in widths]
        rand_start = [torch.randint(0, w-dw+1,(1,)).item() for w, dw in zip(widths, dwidths)]
        _slices = [slice(_start, _start + dw) for _start, dw in zip(rand_start, dwidths)]
        
        input_slices = [slice(b,b+1), slice(idx, idx + self.memory_length),slice(0,chan), *_slices]
        output_slices = [slice(b,b+1), slice(idx + self.memory_length, idx + self.memory_length + self.predict_length),slice(0,chan), *_slices]
        

        x = self.data[input_slices][0,...]  # memory length
        y = self.data[output_slices][0,...]  # predict length
        
        # print(x.shape, y.shape)
        
        return {'memory':x, 'predict': y}
    
    def shape(self):
        return (len(self), self.memory_length, *self.data.shape[2:])

class Normalize():
    def __init__(self, data, **kwargs):
        
        self.kwargs = kwargs
        
        _n = len(data.shape)
        pos_args = tuple(range(3, _n))
    
        self.mean = torch.mean(data, dim=(0,1,*pos_args), keepdim=True) # B T H W, C
        self.range = torch.amax(data, dim=(0,1,*pos_args), keepdim=True) \
            - torch.amin(data, dim=(0,1,*pos_args), keepdim=True)
            
        assert torch.all(self.range > 0), 'range must be positive'
        

    def __call__(self, x, mode):
        
        memory_length = self.kwargs.get(f'{mode}_memory_length', default_memory_length)
        predict_length = self.kwargs.get(f'{mode}_predict_length', default_predict_length)
        downsample_factor = 1 if mode == 'test' else self.kwargs.get('downsample_factor', 1)
        logger.info(f'Normalizing {mode} data with memory length {memory_length} and predict length {predict_length}')

        # return TimeSeriesDataset((x - self.mean) / (self.range), memory_length, predict_length)
        return TimeSeriesDataset(x, memory_length, predict_length, downsample_factor=downsample_factor)  

# def load_h5(filepath, **kwargs):
#     with h5py.File(filepath, 'r') as file:
#         data = torch.from_numpy(np.array(list(file['fields']))) #.to(device) # N C H W
#     return data

def load_npy(filepath, **kwargs):
    data = torch.from_numpy(np.load(filepath)).to(torch.float32) #.to(device) # N C H W
    return data
    
def load_data(folder, **kwargs):
    # if os.path.exists(f'{folder}train.h5'):
    #     logger.info(f'Loading data from {folder}')
    #     _train = load_h5(f'{folder}train.h5')
    #     _test = load_h5(f'{folder}test.h5')
    #     _infer = load_h5(f'{folder}infer.h5')
    if os.path.exists(f'{folder}train.npy'):
        logger.info(f'Loading data from {folder}')
        _train = load_npy(f'{folder}train.npy')
        _test = load_npy(f'{folder}test.npy')
        _infer = load_npy(f'{folder}infer.npy')
    elif isinstance(folder, tuple):
        logger.info(f'Loading data directly')
        _train = (folder[0])
        _test = (folder[1])
        _infer = (folder[2])
    else:
        raise FileNotFoundError('No data found')
    logger.info(f'Train: {_train.shape}, Test: {_test.shape}, Infer: {_infer.shape}')
    
    norm = Normalize(_train, **kwargs)
    logger.info(f'Mean: {norm.mean}, Range: {norm.range}')
    train = norm(_train, 'train')
    test = norm(_test, 'train')
    infer = norm(_infer, 'test')  # infer is treated as test data for normalization
    datasets = [train, test, infer]
    logger.info('normalization complete')
    
    fkwargs = filter_kwargs(kwargs, DataLoader)
    
    _loaders = [DataLoader(data, **fkwargs, shuffle=shuffle, pin_memory=True, collate_fn=collate_fn) for data, shuffle in zip(datasets, [True, True, False])]
    
    
    
    logger.info('dataloaders created')
    return _loaders, [d.shape() for d in datasets], datasets
    
if __name__ == '__main__':
    data = load_h5('data/VB/infer.npy')
