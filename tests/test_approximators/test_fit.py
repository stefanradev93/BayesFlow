import pytest
import io
import numpy as np
from contextlib import redirect_stdout


@pytest.mark.skip(reason="not implemented")
def test_compile(amortizer):
    amortizer.compile(optimizer="AdamW")


@pytest.mark.skip(reason="not implemented")
def test_fit(amortizer, dataset):
    amortizer.compile(optimizer="AdamW")
    amortizer.fit(dataset)

    assert amortizer.losses is not None


def test_loss_progress(approximator, dataset):
    # TODO: Add assertion for validation history (this will make string searching less hacky)
    approximator.compile(optimizer="AdamW")
    num_epochs = 3
    num_batches = len(dataset)

    # Capture ostream and train model
    ostream = io.StringIO()
    with redirect_stdout(ostream):
        history = approximator.fit(dataset, epochs=num_epochs).history
    output = ostream.getvalue()
    ostream.close()

    losses = [line for line in output.splitlines() if "loss" in line]
    assert len(losses) == (num_batches + 1) * num_epochs

    # Assert losses are not NaN and epoch summaries match history
    epoch = 0
    for i, line in enumerate(losses):
        loss = float(line.split()[-1])
        assert not np.isnan(loss)

        if i % (num_batches + 1) == num_batches:
            print(f"History: {round(history["loss"][epoch], 4)} | Captured: {loss}")
            assert round(history["loss"][epoch], 4) == loss
            epoch += 1
