import keras
import torch


class TorchWorkflow(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, data: any, stage: str = "training") -> dict[str, torch.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def train_step(self, data: any, stage: str = "training") -> dict[str, torch.Tensor]:
        raise NotImplementedError
