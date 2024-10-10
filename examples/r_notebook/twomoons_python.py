import os

import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

import bayesflow as bf
import keras


def noise_prior():
    alpha = np.random.uniform(-np.pi / 2, np.pi / 2)
    r = np.random.normal(0.1, 0.01)
    return dict(alpha=alpha, r=r)


def theta_prior():
    theta = np.random.uniform(-1, 1, 2)
    return dict(theta=theta)


def forward_model(theta, alpha, r):
    x1 = -np.abs(theta[0] + theta[1]) / np.sqrt(2) + r * np.cos(alpha) + 0.25
    x2 = (-theta[0] + theta[1]) / np.sqrt(2) + r * np.sin(alpha)
    return dict(x=np.array([x1, x2]))


simulator = bf.simulators.CompositeLambdaSimulator(
    [noise_prior, theta_prior, forward_model]
)

sample_data = simulator.sample((64,))

print("Type of sample_data:\n\t", type(sample_data))
print("Keys of sample_data:\n\t", sample_data.keys())
print("Types of sample_data values:\n\t", {k: type(v) for k, v in sample_data.items()})
print("Shapes of sample_data values:\n\t", {k: v.shape for k, v in sample_data.items()})

data_adapter = bf.ContinuousApproximator.build_data_adapter(
    inference_variables=["theta"],
    inference_conditions=["x"],
)

num_training_batches = 512
num_validation_batches = 128
batch_size = 64

training_samples = simulator.sample((num_training_batches * batch_size,))
validation_samples = simulator.sample((num_validation_batches * batch_size,))

training_dataset = bf.datasets.OfflineDataset(
    training_samples, batch_size=batch_size, data_adapter=data_adapter
)
validation_dataset = bf.datasets.OfflineDataset(
    validation_samples, batch_size=batch_size, data_adapter=data_adapter
)

inference_network = bf.networks.FlowMatching(
    subnet="mlp",
    subnet_kwargs=dict(
        depth=6,
        width=256,
    ),
)

approximator = bf.ContinuousApproximator(
    inference_network=inference_network,
    data_adapter=data_adapter,
)

learning_rate = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

approximator.compile(optimizer=optimizer)

print(" ------> Start Training < ------ ")

history = approximator.fit(
    epochs=3,
    dataset=training_dataset,
    validation_data=validation_dataset,
)
