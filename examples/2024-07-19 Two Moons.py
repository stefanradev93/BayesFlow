import bayesflow as bf
import keras

epochs = 100
steps_per_epoch = 100
batch_size = 512

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
    batch_size=batch_size,
    # we don't need workers or multiprocessing because our dataset is small
    workers=1,
    use_multiprocessing=False,
)
