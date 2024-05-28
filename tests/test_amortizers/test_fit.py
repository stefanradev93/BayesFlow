def test_compile(amortizer):
    amortizer.compile(optimizer="AdamW")


def test_fit(amortizer, dataset):
    amortizer.compile(optimizer="AdamW")
    amortizer.fit(dataset)

    assert amortizer.losses is not None


