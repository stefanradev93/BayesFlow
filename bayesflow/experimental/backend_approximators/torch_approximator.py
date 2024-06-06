
import torch


from .base_approximator import BaseApproximator


class TorchApproximator(BaseApproximator):
    def train_step(self, data):
        with torch.enable_grad():
            metrics = self.compute_metrics(data, mode="training")

        loss = metrics.pop("loss")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics
