from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import functional
import numpy
from ciflmdb.lmdb import ciflmdb
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

    def __init__(self, db_paths, size, raw_image, channels=[], transform=None):
        """Initialize LMDB dataset for one ore more databases. Multiple databases are virtually concatenated, transparent for user.
        
        Arguments:
            db_paths {[str]} -- Paths to LMDB datasets.
            size {int} -- Size to which images need to be center-cropped or -padded.
            raw_image {bool} -- If `True` mask is not applied.
        
        Keyword Arguments:
            channels {[int]} -- List of channels of interest. For example, in case you're only interested in channels R and G of an RGB image, supply `[0, 1]`. (default: {[]})
            transform {callable} -- Function to apply on each instance in the database. (Applied on get.) (default: {None})
        
        Raises:
            ValueError: Thrown when not all databases contain the same amount of channels.
        """
        self.db_paths = db_paths
        self.size = size
        self.transform = transform
        self.raw_image = raw_image
        self.dbs = []
        self.db_start_index = []
        
        self.length = 0
        tmp_channels = []
        self.__targets = []
        for db_path in self.db_paths:
            db = ciflmdb(db_path)
            self.length += len(db)
            tmp_channels.append(db.names)
            self.__targets.extend(db.targets)

        self.__targets = numpy.array(self.__targets)
        self._label_offset = self.__targets.min()
        self.__targets -= self._label_offset
        self.__classes = numpy.unique(self.__targets)

        if all(len(i) == len(tmp_channels[0]) for i in tmp_channels):
            if len(channels) > 0:
                self.channels_of_interest = channels
            else:
                self.channels_of_interest = [i for i in range(len(tmp_channels[0]))]
        else:
            raise ValueError("Not all DBs contain the same amount of channels.")
        
        self.__image_shape = [len(self.channels_of_interest), size, size]

    @property
    def targets(self):
        return self.__targets
    @property
    def classes(self):
        return self.__classes
    @property
    def image_shape(self):
        return self.__image_shape
    @property
    def dtype(self):
        return numpy.float32

    def _setup(self):
        """Private function used to setup databases. Required for multiprocessing setting.
        """
        i = 0
        for db_path in self.db_paths:
            db = ciflmdb(db_path)
            db.set_channels_of_interest(self.channels_of_interest)

            self.dbs.append(db)
            self.db_start_index.append(i)
            
            i += len(db)

        self.db_start_index = numpy.array(self.db_start_index)

    def __getitem__(self, index):
        """Fetches instance from database based on index.
        
        Arguments:
            index {int} -- Index of instance to be fetched
        
        Returns:
            tuple -- image [, label]
        """
        if len(self.dbs) == 0:
            self._setup()

        db_idx = sum(self.db_start_index - index <= 0)-1
        db = self.dbs[db_idx]
        start_idx = self.db_start_index[db_idx]
        image, mask, label = db.get_image(index-start_idx, only_coi=True)

        if not self.raw_image:
            image = numpy.multiply(
                numpy.float32(image),
                numpy.float32(mask)
            )
        else:
            image = np.float32(image)

        width, height = image.shape[1], image.shape[2]
        size = self.size
        if width > size or height > size:
            image = centercrop(size, size, image)

        if width < size or height < size:
            image = centerpad(size, size, image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label-self._label_offset

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'        
