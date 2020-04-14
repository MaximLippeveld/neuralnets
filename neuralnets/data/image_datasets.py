from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import functional
import numpy
from ifcimglib.imglmdb import multidbwrapper
from math import ceil, floor


def centercrop(width, height, image):
    spill = (
        max(0, (image.shape[1] - width)/2.), 
        max(0, (image.shape[2] - height)/2.)
    )

    return image[
        :,
        int(floor(spill[0])): -int(ceil(spill[0])) if spill[0] > 0 else None,
        int(floor(spill[1])): -int(ceil(spill[1])) if spill[1] > 0 else None
    ]

def centerpad(width, height, image):
    pad = (width-image.shape[1])/2., (height-image.shape[2])/2.

    tmp_im = numpy.zeros((image.shape[0], width, height), dtype=image.dtype)
    tmp_im[
        :,
        int(floor(pad[0])): -int(ceil(pad[0])) if pad[0] > 0 else None,
        int(floor(pad[1])): -int(ceil(pad[1])) if pad[1] > 0 else None
    ] = image

    return tmp_im


class LMDBDataset(Dataset):
    """
    Dataset class for loading one or more lightning memory-mapped (LM) databases.
    """    

    def __init__(self, db_paths, size, raw_image, channels=[], transform=None, pass_mask=True, **transform_args):
        """Initialize LMDB dataset for one ore more databases. Multiple databases are virtually concatenated, transparent for user.
        
        Arguments:
            db_paths {[str]} -- Paths to LMDB datasets.
            size {int} -- Size to which images need to be center-cropped or -padded.
            raw_image {bool} -- If `True` mask is not applied.
            pass_mask {bool} -- If `True` mask is passed to transform function.
        
        Keyword Arguments:
            channels {[int]} -- List of channels of interest. For example, in case you're only interested in channels R and G of an RGB image, supply `[0, 1]`. (default: {[]})
            transform {callable} -- Function with signature `fun(image, mask)` to apply on each instance in the database. (Applied at `get`.) (default: {None})
            transform_args {args} -- Keyword arguments passed to transform function.
        
        Raises:
            ValueError: Thrown when not all databases contain the same amount of channels.
        """
        self.db_paths = db_paths
        self.size = size
        self.transform = transform
        self.raw_image = raw_image
        self.pass_mask = pass_mask
        self.transform_args = transform_args

        self.db = multidbwrapper(db_paths)
        self.targets = self.db.targets
        self.classes = self.db.classes
        self.dtype = self.db.dtype
        self.channels_of_interest = self.db.channels_of_interest
        
        self.__image_shape = [len(self.channels_of_interest), size, size]

    @property
    def image_shape(self):
        return self.__image_shape


    def __getitem__(self, index):
        """Fetches instance from database based on index.
        
        Arguments:
            index {int} -- Index of instance to be fetched
        
        Returns:
            tuple -- image [, label]
        """
        image, mask, label = self.db.get_image(index, only_coi=True)

        if not self.raw_image:
            image = numpy.multiply(
                numpy.float32(image),
                numpy.float32(mask)
            )
        else:
            image = numpy.float32(image)
        
        if self.transform is not None:
            if self.pass_mask:
                image = self.transform(image, mask, **self.transform_args)
            else:
                image = self.transform(image, **self.transform_args)

        width, height = image.shape[1], image.shape[2]
        size = self.size
        if width > size or height > size:
            image = centercrop(size, size, image)

        if width < size or height < size:
            image = centerpad(size, size, image)

        return image, label

    def __len__(self):
        return len(self.db)
