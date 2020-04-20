import torch
import numpy
import queue
import math


class _EternalShuffleQueue(queue.Queue):
    def __init__(self, items, c):
        super(_EternalShuffleQueue, self).__init__()
        self.items = items
        self.c = c
    
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
        self.batch_size = batch_size

        # Note on length computation. In each batch we have on average batch_size/num_classes of instances per class.
        # In order to see each instance of the majority class once, we have to generate num_instances_major/(batch_size/num_classes) batches.
        # Since all other classes have less instances, we will have seen all of them with this number of batches.
        self.length = math.ceil(int(max(numpy.unique(targets, return_counts=True)[1])/(batch_size/len(self.classes))))

        self.idx_per_class = {}
        targets = numpy.array(targets)
        for i in self.classes:
            # get all indices for class i
            self.idx_per_class[i] = _EternalShuffleQueue(numpy.nonzero(targets == i)[0], i)

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