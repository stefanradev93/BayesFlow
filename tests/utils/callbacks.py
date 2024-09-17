import keras.callbacks


class FitInterruptedError(Exception):
    pass


class InterruptFitCallback(keras.callbacks.Callback):
    def __init__(self, batches=None, epochs=None, error_type=FitInterruptedError):
        super().__init__()
        if batches is None and epochs is None:
            raise ValueError("Either batches or epochs must be specified.")

        self.batches = batches
        self.epochs = epochs
        self.error_type = error_type

    def on_train_batch_end(self, batch, logs=None):
        if self.batches is not None and self.batches <= batch:
            raise self.error_type()

    def on_epoch_end(self, epoch, logs=None):
        if self.epochs is not None and self.epochs <= epoch:
            raise self.error_type()
