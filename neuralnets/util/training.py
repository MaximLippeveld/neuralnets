import os, os.path
from datetime import datetime
import torch
import numpy

class model_optimizer_checkpointer:
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
            if self.best_model_path is not None:
                os.remove(os.path.join(self.output_dir, self.best_model_path))
            self.best_model_path = self.BEST_CHECKPOINT_FMT % (
                type(model).__name__, epoch, datetime.today().isoformat(timespec="minutes"))
            torch.save(model.state_dict(), os.path.join(self.output_dir, self.best_model_path))
