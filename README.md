# FEDSSG: EXPECTATION-GATED AND HISTORY-AWARE DRIFT ALIGNMENT FOR FEDERATED LEARNING

This repository provides a minimal environment for experimenting with the FedSSG optimizer alongside other federated learning baselines. The example scripts illustrate how to configure data partitioning, participation, and optimization hyperparameters for common benchmarks such as CIFAR-10.

All helper utilities now live inside the [`utils/`](utils) package so they can be imported with standard module paths such as `utils.general` or `utils.methods`. The package exposes the same functionality that was previously found in the flat `utils_*.py` files, but groups related helpers (datasets, models, training loops, etc.) into dedicated modules for easier discovery.

## Configuring Non-IID Experiments

The [`example_code_cifar10.py`](example_code_cifar10.py) script shows how to instantiate `DatasetObject` and select the training method. The key configuration arguments control both the heterogeneity of the data and the scale of the federation:

### Controlling Dirichlet Partitioning and Client Count

`DatasetObject` accepts three arguments that determine how data is split across clients:

* `rule`: choose `"dirichlet"` to enable Dirichlet-based non-IID partitioning. Other options (e.g., `"iid"`) can be used for alternative splits.
* `rule_arg`: set this to the desired Dirichlet concentration parameter (often referred to as \(\alpha\)). Smaller values (e.g., `0.1`) create highly skewed client datasets, while larger values (e.g., `1.0` or higher) approach IID behavior.
* `n_client`: change the number of participating clients in the federation. Increasing this value adds more clients with correspondingly smaller local datasets; decreasing it concentrates data among fewer clients.

### Adjusting Client Participation and Training Horizon

In the training configuration, the following parameters determine how often clients participate and how long training lasts:

* `act_prob`: sets the probability that a client is selected in each communication round. Lower values simulate partial participation, while `1.0` enables full participation every round.
* `com_amount`: the number of communication rounds (global aggregations) to execute.
* `epoch`: the number of local epochs each selected client runs per round. Increasing `epoch` or `com_amount` lengthens total training time.

### Selecting the Training Method and Editing Hyperparameters

The example script imports multiple training entry points, including `train_FedSSG` and baseline methods such as `train_FedAvg`, `train_FedProx`, and others defined in the modules inside `utils/`. To choose a method, call the corresponding function with your `DatasetObject`, model, and optimizer settings. For instance:

```python
from utils.methods_fedssg import train_FedSSG
from utils.methods import train_FedAvg

# Choose one training routine
train_FedSSG(args, Dataset)
# or
train_FedAvg(args, Dataset)
```

When switching between methods, review and update the relevant hyperparameters:

* **FedSSG** (`train_FedSSG` in `utils/methods_fedssg.py`): tune parameters such as `lambda_reg`, `momentum`, and optimizer learning rates defined in the script.
* **FedAvg / FedProx / other baselines** (`utils/methods.py`, `utils/methods_feddc.py`, etc.): adjust baseline-specific parameters (e.g., `mu` for FedProx, control variates for FedDC) along with shared settings (`learning_rate`, `batch_size`).

Refer to the module docstrings inside `utils/` for additional guidance on available classes, helper functions, and expected configuration arguments.

Modify these hyperparameters in the example script or pass them via an argument parser to explore different federated training scenarios.


### Citation

If you find this implementation useful, please cite the MAGIA paper:

```
FEDSSG: EXPECTATION-GATED AND HISTORY-AWARE DRIFT ALIGNMENT FOR FEDERATED LEARNING
```

---
