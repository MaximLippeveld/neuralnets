import os, os.path, logging
from datetime import datetime
import torch
import numpy
from tensorboard import default, program


class NonBlockingSummaryWriter:

    def __init__(self, summarywriter):
        self.summarywriter = summarywriter
        self.manager = multiprocessing.Manager()
        self.queue = manager.Queue()

        self.consumer = multiprocessing.Process(
            target=NonBlockingSummaryWriter._consumer_func, args=(summarywriter, self.queue), name="Reporting")
        consumer.start()

    def put(self, func_name, *args):
        self.queue.put(func_name, args)

    def __del__(self):
        self.queue.put(None)
        self.consumer.join()
        self.summarywriter.close()

    @staticmethod
    def _consumer_func(writer, queue):
        while True:
            item = queue.get()
            if item == None:
                break

            func, args = item
            getattr(writer, func)(*args)

            for arg in args:
                del arg
        
class TensorBoardProcess(torch.multiprocessing.Process):

    def __init__(self, output, port=6006):
        super(TensorBoardProcess, self).__init__()
        self.output = output
        self.port = port

        self.daemon = True
        self.start()

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
