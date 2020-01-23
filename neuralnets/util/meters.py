import torch
import numpy

class average_meter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, shape=None, fmt=':f', cuda=False):
        self.name = name
        self.fmt = fmt
        self.cuda = cuda
        self.shape = shape
        if shape is not None:
            self.reset(shape)

    def reset(self, shape=None):
        if shape == None:
            shape = self.shape
        else:
            self.shape = shape

        if self.cuda:
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")

        self.val = torch.zeros(shape, device=device)
        self.avg = torch.zeros(shape, device=device)
        self.sum = torch.zeros(shape, device=device)
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
