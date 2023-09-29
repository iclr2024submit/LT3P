import numpy as np
import torch

class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, model, path):
        #score = -val_loss
        #if self.best_score is None:
        #    self.best_score = score
        #    self.save_checkpoint(val_loss, model, path)
        #elif score >= self.best_score + self.delta:
        #    self.best_score = score
        self.save_checkpoint(model, path)

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        #self.val_loss_min = val_loss
