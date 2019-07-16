import warnings
import torch


class TrainingSignalAnnealing:

    def __init__(self, num_steps, min_threshold=0.0, max_threshold=1.0, mode='linear', preds_as_probas=True):
        assert mode in ("linear", )
        self.thresholds = torch.linspace(min_threshold, max_threshold, steps=num_steps)
        self._step = 0
        self.preds_as_probas = preds_as_probas

    def __call__(self, y_pred, y, step=None):
        step = self._step if step is None else step
        self._step += 1

        if step >= len(self.thresholds) or step < 0:
            warnings.warn("Step {} is out of bounds".format(step))
            return y_pred, y

        t = self.thresholds[step].item()
        tmp_y_pred = y_pred.detach()
        if self.preds_as_probas:
            tmp_y_pred = torch.softmax(tmp_y_pred, dim=1)
        res = tmp_y_pred.gather(dim=1, index=y.unsqueeze(dim=1))
        mask = (res < t).squeeze(dim=1)
        if mask.sum() > 0:            
            return y_pred[mask], y[mask]

        warnings.warn("Threshold {} is too low, all predictions are discarded.\n".format(t) +
                      "y_pred.min/max: {}, {}".format(tmp_y_pred.min(), tmp_y_pred.max()))
        return y_pred, y
