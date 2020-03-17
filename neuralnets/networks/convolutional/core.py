import torch.nn
from tqdm import tqdm
import torchvision.utils
import numpy
from neuralnets.util import meters
import sklearn.metrics
import pickle
import logging


class Model:
    
    def __init__(self, num_classes, model):
        self.model = model
        self.num_classes = num_classes

        # setup meters
        self.epoch_loss = meters.average_meter("epoch/avg_batch_loss", shape=(1,))
        self.epoch_balacc = meters.average_meter("epoch/avg_batch_balacc", shape=(1,))
        self.epoch_cm = meters.average_meter("epoch/avg_batch_cm", shape=(num_classes, num_classes))

    def meter_values(self, scalar=False):
        """Get dict with names and values from meters defined for this model.
        
        Keyword Arguments:
            scalar {bool} -- Specify to only return scalar values (default: {False})

        Returns
            dict -- keys are meter names, values are average values obtained from meters
        """
        ret = {
            self.epoch_loss.name: self.epoch_loss.avg.item(),
            self.epoch_balacc.name: self.epoch_balacc.avg.item(),
        }
        if not scalar:
            ret[self.epoch_cm.name] = self.epoch_cm.avg.tolist()

        return ret
    
    @staticmethod
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    def meters(self):
        """Get list of meters defined for this model.
        
        Returns
            list -- list of meters
        """
        return [
            self.epoch_loss, self.epoch_balacc, self.epoch_cm
        ]

    def monitor(self):
        """Returns meter containing value to optimize for model selection
        
        Returns
            average_meter -- meter
        """
        return self.epoch_balacc.avg.item()

    def compute_metrics(self, loss, y_true, y_pred):
        """Compute metrics of interest
        
        Arguments:
            loss {Tensor} -- Loss tensor
            y_true {list} -- Vector of true labels
            y_pred {list} -- Vector of predicted labels
        
        Returns
            dict -- computed metrics
        """
        balacc = sklearn.metrics.balanced_accuracy_score(
            y_true, 
            y_pred
        )
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred, 
            torch.arange(0, self.num_classes), normalize='true')
        
        # log
        self.epoch_loss.update(loss.item())
        self.epoch_balacc.update(balacc)
        self.epoch_cm.update(cm)

        return {"balacc": balacc}

    def run(self, phase, epoch, dl, opt, loss, stop_at_batch=-1):
        logger = logging.getLogger(__name__)

        if phase == "train":
            assert opt is not None, "Pass optimizer when training."

        for meter in self.meters():
            meter.reset() 

        torch.autograd.set_grad_enabled(phase == "train")
        torch.autograd.set_detect_anomaly(True)

        if phase=="train":
            self.model.train()
        else:
            self.model.eval()

        total = stop_at_batch if stop_at_batch > 0 else len(dl)
        batch_tqdm = tqdm(iter(dl), total=total, leave=False)
        for i, (batch, y) in enumerate(batch_tqdm):
            if phase == "train":
                opt.zero_grad()

            if i == stop_at_batch:
                break

            batch, y = batch.cuda(), y.cuda()

            if torch.isnan(y).any():
                raise ValueError("something went wrong in y (label), phase %s" % phase, y)
            if torch.isnan(batch).any():
                raise ValueError("something went wrong in input batch, phase %s" % phase, batch)
            
            y_hat = self.model(batch)

            if torch.isnan(y_hat).any():
                logger.debug(torch.cuda.memory_stats())
                raise ValueError("something went wrong in forward, phase %s" % phase, y_hat)

            # compute metrics
            l = loss(y_hat, y)
            if torch.isnan(l).any():
                raise ValueError("something went wrong in loss, phase %s" % phase, l)

            y_true = y.detach().cpu().numpy()
            y_pred = y_hat.detach().cpu().numpy().argmax(axis=1)
            postfix = self.compute_metrics(l, y_true, y_pred)
            postfix.update({"ce_loss": l.item()})
            batch_tqdm.set_postfix(postfix)

            # step
            if phase == "train":
                l.backward()
                opt.step()

        # return checkpoint metric
        return self.monitor()
