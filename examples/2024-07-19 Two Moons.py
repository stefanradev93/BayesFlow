def flatten_samples(samples):
    import keras

    flattened_samples = {}
    for key, value in samples.items():
        shape = keras.ops.shape(value)
        if shape[1] == 1:
            flattened_samples[key] = keras.ops.squeeze(value, 1)
        else:
            for i in range(shape[1]):
                flattened_samples[f"{key}{i + 1}"] = value[:, i]
    return flattened_samples


def predicate(samples):
    import keras

    return keras.ops.norm(samples["x"], axis=-1) <= 1e-3


def plot(samples):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # add dashed minor grid lines
    sns.set_style(
        "whitegrid", {"axes.grid": True, "axes.grid.which": "both", "grid.linestyle": "--", "grid.linewidth": 0.5}
    )

    samples = flatten_samples(samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(
        samples["theta1"], samples["theta2"], s=10, lw=0, c=samples["r"], cmap=sns.cubehelix_palette(as_cmap=True)
    )
    axes[1].scatter(
        samples["theta1"], samples["theta2"], s=10, lw=0, c=samples["alpha"], cmap=sns.cubehelix_palette(as_cmap=True)
    )

    axes[0].set_title(r"Hue by $r$")
    axes[0].set_xlabel(r"$\theta_1$")
    axes[0].set_ylabel(r"$\theta_2$")

    axes[1].set_title(r"Hue by $\alpha$")
    axes[1].set_xlabel(r"$\theta_1$")
    axes[1].set_ylabel(r"$\theta_2$")

    fig.suptitle("Two Moons Samples")

    plt.show()


def main():
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
        use_multiprocessing=True,
    )

    samples = approximator.sample((2048,), numpy=True)

    plot(samples)


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "torch"

    main()
