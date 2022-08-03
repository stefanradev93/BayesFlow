# Copyright 2022 The BayesFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module implements all 10 benchmark models (tasks) from the paper:
#
# Lueckmann, J. M., Boelts, J., Greenberg, D., Goncalves, P., & Macke, J. (2021). 
# Benchmarking simulation-based inference. 
# In International Conference on Artificial Intelligence and Statistics (pp. 343-351). PMLR.
#
# https://arxiv.org/pdf/2101.04653.pdf
#
# However, it lifts the dependency on `torch` and implements the models as ready-made
# tuples of prior and simulator functions capable of interacting with BayesFlow.
