import torch
import numpy
import queue


class _EternalShuffleQueue(queue.Queue):
    def __init__(self, items):
        super(_EternalShuffleQueue, self).__init__()
        self.items = items
    
    def pop(self):
        if super().empty():
            numpy.random.shuffle(self.items)
            for item in self.items:
                super().put(item)

        return super().get()

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    This sampler provides mini-batches with a balanced amount of instances per class.
    """


    def __init__(self, targets, batch_size):
        """Initialize balanced batch sampler.
        
        Arguments:
            targets {[int]} -- List of target label values, corresponding to dataset on which this sampler operates.
            batch_size {int} -- Batch size.
        """
        self.classes = numpy.unique(targets)
        self.length = int(numpy.ceil(len(targets) / batch_size))
        self.batch_size = batch_size

        self.idx_per_class = {}
        targets = numpy.array(targets)
        for i in self.classes:
            # get all indices for class i
            self.idx_per_class[i] = _EternalShuffleQueue(numpy.nonzero(targets == i)[0])

    def __iter__(self):
        for _ in range(self.length):
            yield numpy.array(
                list(
                    map(
                        # map class index to random dataset index of that class
                        lambda x: self.idx_per_class[x].pop(), 
                        # randomly sample class indices
                        numpy.random.choice(self.classes, self.batch_size)
                    )
                )
            )

    def __len__(self):
        return self.length