import os, os.path, logging, sys
from datetime import datetime
import torch
import numpy
from tensorboard import default, program
from torch import multiprocessing
from torch.utils.tensorboard import SummaryWriter
import neuralnets.util.plotting
import logging
from joblib import dump


class SummaryWriterProcess(torch.multiprocessing.Process):

    def __init__(self, output_dir):
        super(SummaryWriterProcess, self).__init__(name="Reporting")
        self.output_dir = output_dir
        self.queue = multiprocessing.Queue()
        self.queue_alive = True

    def run(self):
        logger = logging.getLogger(__name__)
        logger.debug("Consumer started")

        try:
            writer = SummaryWriter(os.path.join(self.output_dir, "tb"))
            logger.debug("Consumer created summarywriter")
            while True:
                logger.debug("Consumer waiting for queue item")
                item = self.queue.get()

                case, func, args, kwargs = item
                logger.debug("Processing item with case %s, func %s" % (case, str(func)))
                if case == "plotting":
                    fig = getattr(neuralnets.util.plotting, func)(*args)
                    writer.add_figure(figure=fig, **kwargs)
                    del fig
                elif case == "writer":
                    getattr(writer, func)(*args, **kwargs)
                elif case == "stop":
                    logger.debug("Consumer received stop")
                    break

                for arg in args:
                    del arg
                for k,arg in kwargs.items():
                    del arg

        except Exception as e:
            logger.error("Consumer crashed (%s, %s)" % (case, str(func)), e)
        finally:
            logger.debug("Consumer closed writer")
            writer.close()
            self.queue_alive = False

    def put(self, case, func_name, *args, **kwargs):
        if self.queue_alive:
            self.queue.put((case, func_name, args, kwargs or {}))

class TensorBoardProcess(torch.multiprocessing.Process):

    def __init__(self, output, port=6006):
        super(TensorBoardProcess, self).__init__()
        self.output = output
        self.port = port
        self.daemon = True

    def run(self):
        tb = program.TensorBoard(
            default.get_plugins() + default.get_dynamic_plugins(), 
            program.get_default_assets_zip_provider())
        tb.configure(logdir = self.output, port = self.port)
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.CRITICAL)
        tb.main()


class model_optimizer_checkpointer:
    """
    Stores model and optimizer `state_dict` after every epoch, and best model when criterion improves.
    """


    CHECKPOINT_FMT = "%s_epoch%d_%s.pth"
    BEST_CHECKPOINT_FMT = "%s_epoch%d_%s_best.pth"
    BEST_PREDICTIONS_FMT = "best_predictions.joblib"

    def __init__(self, output_dir, goal="maximize"):
        self.output_dir = output_dir
        self.goal = goal
        self.optimal = -numpy.inf if goal == "maximize" else numpy.inf
        self.model_path = None
        self.best_model_path = None
        self.opt_path = None

    def save(self, model, opt, criterion_value, epoch):

        # save intermediate model
        if self.model_path is not None:
            os.remove(os.path.join(self.output_dir, self.model_path))
        if self.opt_path is not None:
            os.remove(os.path.join(self.output_dir, self.opt_path))

        self.model_path = self.CHECKPOINT_FMT % (
            type(model).__name__, epoch, datetime.today().isoformat(timespec="minutes"))
        self.opt_path = self.CHECKPOINT_FMT % (
            type(opt).__name__, epoch, datetime.today().isoformat(timespec="minutes"))
        torch.save(model.state_dict(), os.path.join(self.output_dir, self.model_path))
        torch.save(opt.state_dict(), os.path.join(self.output_dir, self.opt_path))

        # save best performing model
        if (
            self.goal == "maximize" and criterion_value > self.optimal
            or self.goal != "maximize" and criterion_value < self.optimal
        ):
            self.optimal = criterion_value
            if self.best_model_path is not None:
                os.remove(os.path.join(self.output_dir, self.best_model_path))
            self.best_model_path = self.BEST_CHECKPOINT_FMT % (
                type(model).__name__, epoch, datetime.today().isoformat(timespec="minutes"))
            torch.save(model.state_dict(), os.path.join(self.output_dir, self.best_model_path))

        return self.optimal

    def restore_best_model(self, model):
        if self.best_model_path:
            model.load_state_dict(torch.load(self.best_model_path))
        else:
            raise ValueError("Best model path isn't set")

    def save_predictions(self, predictions):
        dump(predictions, os.path.join(self.output_dir, self.BEST_PREDICTIONS_FMT)) 
