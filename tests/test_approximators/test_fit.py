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


def test_loss_progress(approximator, train_dataset, validation_dataset):
    approximator.compile(optimizer="AdamW")
    num_epochs = 3

    # Capture ostream and train model
    ostream = io.StringIO()
    with redirect_stdout(ostream):
        history = approximator.fit(train_dataset, validation_data=validation_dataset, epochs=num_epochs).history
    output = ostream.getvalue()
    ostream.close()

    loss_output = [line for line in output.splitlines() if "loss" in line]

    # Test losses are not NaN and that epoch summaries match loss histories
    epoch = 0
    for loss_stats in loss_output:
        content = loss_stats.split()
        if "val_loss" in loss_stats:
            assert float(content[-4]) == round(history["loss"][epoch], 4)
            assert float(content[-1]) == round(history["val_loss"][epoch], 4)
            epoch += 1
            continue

        assert not np.isnan(float(content[-1]))
