import pytest


@pytest.mark.skip(reason="not implemented")
def test_compile(amortizer):
    amortizer.compile(optimizer="AdamW")


@pytest.mark.skip(reason="not implemented")
def test_fit(amortizer, dataset):
    amortizer.compile(optimizer="AdamW")
    amortizer.fit(dataset)

    assert amortizer.losses is not None
