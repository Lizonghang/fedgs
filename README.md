# FedGS: Data Heterogeneity-Robust Federated Learning via Group Client Selection in Industrial IoT

## Preparation 

* For instructions on generating data, please go to the folder of the corresponding dataset. For FEMNIST, please refer to [femnist](https://github.com/Lizonghang/fedgs/tree/main/data/femnist).

* NVIDIA-Docker is required.

* NVIDIA CUDA version 10.1 and higher is required.

## How to run FedGS
### Build a docker image

Enter the [scripts](https://github.com/Lizonghang/fedgs/tree/main/scripts) folder and build a docker image named <code>fedgs</code>.
 
> sudo docker build -f build-env.dockerfile -t fedgs .

Modify <code>/home/lizh/fedgs</code> to your actual project path in [scripts/run.sh](https://github.com/Lizonghang/fedgs/blob/main/scripts/run.sh). Then run [scripts/run.sh](https://github.com/Lizonghang/fedgs/blob/main/scripts/run.sh), which will create a container named <code>fedgs.0</code> if <code>CONTAINER_RANK</code> is set to 0 and starts the task.

> chmod a+x run.sh && ./run.sh

The output logs and models will be stored in a <code>logs</code> folder created automatically. For example, outputs of the FEMNIST task with container rank 0 will be stored in <code>logs/femnist/0/</code>.

## Hyperparameters
We categorize hyperparameters into default settings and custom settings, and we will introduce them separately.

### Default Hyperparameters
These hyperparameters are included in [utils/args.py](https://github.com/Lizonghang/fedgs/blob/main/utils/args.py). We list them in the table below (except for custom hyperparameters), but in general, we do not need to pay attention to them.

| Variable Name | Default Value | Optional Values | Description |
|---|---|---|---|
| --seed | 0 | integer | Seed for client selection and batch splitting. |
| --metrics-name | "metrics" | string | Name for metrics file. |
| --metrics-dir | "metrics" | string | Folder name for metrics files. |
| --log-dir | "logs" | string | Folder name for log files. |
| --use-val-set | None | None | Set this option to use the validation set, otherwise the test set is used. (NOT TESTED) |

### Custom Hyperparameters
These hyperparameters are included in [scripts/run.sh](https://github.com/Lizonghang/fedgs/blob/main/scripts/run.sh). We list them below.

| Environment Variable | Default Value | Description |
|---|---|---|
| CONTAINER_RANK | 0 | This identify the container (e.g., <code>fedgs.0</code>) and log files (e.g., <code>logs/femnist/0/output.0</code>). |
| BATCH_SIZE | 32 | Number of training samples in each batch. |
| LEARNING_RATE | 0.01 | Learning rate for local optimizers. |
| NUM_GROUPS | 10 | Number of groups. |
| CLIENTS_PER_GROUP | 10 | Number of clients selected in each group. |
| SAMPLER | gbp-cs | Sampler to be used, can be random, brute, bayesian, probability, ga and gbp-cs. |
| NUM_SYNCS | 50 | Number of internal synchronizations in each round. |
| NUM_ROUNDS | 500 | Total rounds of external synchronizations. |
| DATASET | femnist | Dataset to be used, only FEMNIST is supported currently. |
| MODEL | cnn | Neural network model to be used. |
| EVAL_EVERY | 1 | Interval rounds for model evaluation. |
| NUM_GPU_AVAILABLE | 2 | Number of GPUs available. |
| NUM_GPU_BEGIN | 0 | Index of the first available GPU. |
| IMAGE_NAME | fedgs | Experimental image to be used. |

> NOTE: If you wish to specify a GPU device (e.g., GPU0), please set <code>NUM_GPU_AVAILABLE=1</code> and <code>NUM_GPU_BEGIN=0</code>.

> NOTE: This script will mount project files <code>/home/lizh/fedgs</code> from the host into the container <code>/root</code>, so please check carefully whether your file path is correct.

## Visualization

The visualizer [metrics/visualize.py](https://github.com/Lizonghang/fedgs/blob/main/metrics/visualize.py) reads metrics 
logs (e.g., <code>metrics/metrics_stat_0.csv</code> and <code>metrics/metrics_sys_0.csv</code>) and draws curves of accuracy, loss and so on. 

## Note
This project is licensed under the terms of the apache-2.0 license.

## Reference

* This demo is implemented on [LEAF-MX](https://github.com/Lizonghang/leaf-mx), which is a [MXNET](https://github.com/apache/incubator-mxnet) implementation of the well-known federated learning framework [LEAF](https://github.com/TalwalkarLab/leaf).

* Li, Zonghang, Yihong He, Hongfang Yu, *et al.*: "Data heterogeneity-robust federated learning via group client selection in industrial IoT." *IEEE Internet of Things Journal*, vol. 9, no. 18, pp. 17844-17857. IEEE, 2021, doi: 10.1109/JIOT.2022.3161943.

* If you get trouble using this repository, please kindly contact us. Our email: lizhuestc@gmail.com
