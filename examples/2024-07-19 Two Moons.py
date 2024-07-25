def plot():
    import os

    os.environ["KERAS_BACKEND"] = "torch"

    import bayesflow as bf
    import keras
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("whitegrid")

    simulator = bf.simulators.TwoMoonsSimulator()

    def condition(data):
        return keras.ops.norm(data["x"], ord=2, axis=-1) < 0.01

    samples = simulator.rejection_sample((2048,), condition=condition, numpy=True)

    print({key: keras.ops.shape(value) for key, value in samples.items()})

    samples["x2"] = samples["x"][:, 1]
    samples["x1"] = samples["x"][:, 0]
    samples["theta1"] = samples["theta"][:, 0]
    samples["theta2"] = samples["theta"][:, 1]

    del samples["x"]
    del samples["theta"]

    samples = {key: keras.ops.squeeze(value, -1) for key, value in samples.items()}

    g = sns.relplot(samples, x="x1", y="x2", hue="r", lw=0)
    print(type(g))
    print(dir(g))
    g.ax.xaxis.grid(True, "minor", linewidth=0.25)
    g.ax.yaxis.grid(True, "minor", linewidth=0.25)
    g.despine(left=True, bottom=True)
    plt.show()


def main():
    import os

    os.environ["KERAS_BACKEND"] = "torch"

    import bayesflow as bf
    import keras

    epochs = 100
    steps_per_epoch = 100

    approximator = bf.ContinuousApproximator(
        inference_network=bf.networks.FlowMatching(),
    )

    learning_rate = keras.optimizers.schedules.CosineDecay(1e-3, epochs * steps_per_epoch, 1e-6)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    approximator.compile(optimizer=optimizer)

    approximator.fit(
        simulator=bf.simulators.TwoMoonsSimulator(),
        inference_variables=["theta"],
        inference_conditions=["x"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size="auto",
        memory_budget="4 GB",
        # we don't need workers or multiprocessing because our dataset is small
        workers=1,
        use_multiprocessing=False,
    )


if __name__ == "__main__":
    plot()
