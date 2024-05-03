

class OnlineDataset:
    # TODO: do not rely on keras.utils.PyDataset (not backend-agnostic because reasons)
    pass


# class OnlineDataset(keras.utils.PyDataset):
#     """
#     A dataset that is generated on-the-fly.
#     """
#     def __init__(self, joint_distribution: JointDistribution, batch_size: int, steps_per_epoch: int, **kwargs):
#         super().__init__(**kwargs)
#         self.joint_distribution = joint_distribution
#         self.batch_size = batch_size
#         self.steps_per_epoch = steps_per_epoch
#
#     def __getitem__(self, item: int) -> (dict, dict):
#         """ Sample a batch of data from the joint distribution unconditionally """
#         data = self.joint_distribution.sample((self.batch_size,))
#         return data, {}
#
#     def __len__(self) -> int:
#         return self.steps_per_epoch
