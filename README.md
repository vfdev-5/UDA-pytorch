# Unsupervised Data Augmentation experiments in PyTorch

**~~~ /!\ Work in Progress /!\ ~~~**

Experiments with "Unsupervised Data Augmentation" method on Cifar10 dataset.

Based on ["Unsupervised Data Augmentation"](https://arxiv.org/pdf/1904.12848.pdf)

## Unsupervised Data Augmentation in nutshell

![UDA](assets/uda.png)

## Requirements

All experiments are run using [`mlflow`](https://github.com/mlflow/mlflow), please install the latest version of this library
```
pip install --upgrade mlflow
```

## Experiments

### Start MLFlow UI server

Please create output folder (e.g. `$PWD/output`) and setup mlflow server:

```
export OUTPUT_PATH=/path/to/output
```
and 
```
mlflow server --backend-store-uri $OUTPUT_PATH/mlruns --default-artifact-root $OUTPUT_PATH/mlruns -p 5566 -h 0.0.0.0
```

MLflow dashboard is available in the browser at [0.0.0.0:5566](0.0.0.0:5566)

### CIFAR10 dataset

Create once "CIFAR10" experiment
```
export MLFLOW_TRACKING_URI=$OUTPUT_PATH/mlruns
mlflow experiments create -n CIFAR10
```

Start a single run

```
export MLFLOW_TRACKING_URI=$OUTPUT_PATH/mlruns

mlflow run experiments/ --experiment-name=CIFAR10 -P dataset=CIFAR10 -P network=fastresnet -P params="data_path=../input/cifar10;num_epochs=100;learning_rate=0.01;batch_size=512;TSA_proba_min=0.2"
```

Current implementation :
- Model : FastResnet inspired from [cifar10-fast repository](https://github.com/davidcpage/cifar10-fast) (original paper uses Wide-ResNet)
- Consistency loss: KL
- Data augs: AutoAugment + Cutout
- Cosine LR decay
- Training Signal Annealing

### Tensorboard 

All experiments are also logged to the Tensorboard. To visualize the experiments, please install `tensorboard` and run :
```
# tensorboard --logdir=$OUTPUT_PATH/mlruns/<experiment_id>
tensorboard --logdir=$OUTPUT_PATH/mlruns/1
```

## Acknowledgements

In this repository we are using the code from 
- [DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment) 
- [cifar10-fast repository](https://github.com/davidcpage/cifar10-fast)

Thanks to the authors for sharing their code!