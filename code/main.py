
# Unsupervised Data Augmentation method on a dataset
# https://arxiv.org/pdf/1904.12848.pdf


import argparse
import traceback
from pathlib import Path
import tempfile
import math

from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import convert_tensor

from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler

from ignite.contrib.handlers import create_lr_scheduler_with_warmup

import mlflow

from utils import set_seed, get_train_test_loaders, get_model
from utils.tsa import TrainingSignalAnnealing


def run(output_path, config):

    device = "cuda"
    batch_size = config['batch_size']

    train_labelled_loader, train_unlabelled_loader, test_loader = \
        get_train_test_loaders(dataset_name=config['dataset'], 
                               num_labelled_samples=config['num_labelled_samples'],
                               path=config['data_path'],
                               batch_size=batch_size,
                               unlabelled_batch_size=config.get('unlabelled_batch_size', None),
                               num_workers=config['num_workers'])

    model = get_model(config['model'])
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'],
                          nesterov=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    if config['consistency_criterion'] == "MSE":
        consistency_criterion = nn.MSELoss()
    elif config['consistency_criterion'] == "KL":
        consistency_criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        raise RuntimeError("Unknown consistency criterion {}".format(config['consistency_criterion']))
    
    consistency_criterion = consistency_criterion.to(device)

    le = len(train_labelled_loader)
    num_train_steps = le * config['num_epochs']
    mlflow.log_param("num train steps", num_train_steps)

    lr = config['learning_rate']
    eta_min = lr * config['min_lr_ratio']
    num_warmup_steps = config['num_warmup_steps']

    lr_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=num_train_steps - num_warmup_steps)

    if num_warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
                                                       warmup_start_value=0.0,
                                                       warmup_end_value=lr * (1.0 + 1.0 / num_warmup_steps),
                                                       warmup_duration=num_warmup_steps)

    def _prepare_batch(batch, device, non_blocking):
        x, y = batch
        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking))

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_unlabelled_loader_iter = cycle(train_unlabelled_loader)

    lam = config['consistency_lambda']
    
    tsa = TrainingSignalAnnealing(num_steps=num_train_steps,
                                  min_threshold=config['TSA_proba_min'], 
                                  max_threshold=config['TSA_proba_max'])

    with_tsa = config['with_TSA']
    with_UDA = not config['no_UDA']

    def uda_process_function(engine, labelled_batch):
        
        x, y = _prepare_batch(labelled_batch, device=device, non_blocking=True)
    
        if with_UDA:
            unsup_x, unsup_aug_x = next(train_unlabelled_loader_iter)
            unsup_x = convert_tensor(unsup_x, device=device, non_blocking=True)
            unsup_aug_x = convert_tensor(unsup_aug_x, device=device, non_blocking=True)

        model.train()
        # Supervised part        
        y_pred = model(x)
        loss = criterion(y_pred, y)

        supervised_loss = loss
        step = engine.state.iteration - 1
        if with_tsa and with_UDA:
            new_y_pred, new_y = tsa(y_pred, y, step=step)
            new_loss = criterion(new_y_pred, new_y)

            engine.state.tsa_log = {
                "new_y_pred": new_y_pred,
                "loss": loss.item(),
                "tsa_loss": new_loss.item() 
            }
            supervised_loss = new_loss

        # Unsupervised part
        if with_UDA:
            unsup_orig_y_pred = model(unsup_x).detach()
            unsup_orig_y_probas = torch.softmax(unsup_orig_y_pred, dim=-1)

            unsup_aug_y_pred = model(unsup_aug_x)
            unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_pred, dim=-1)

            consistency_loss = consistency_criterion(unsup_aug_y_probas, unsup_orig_y_probas)

        final_loss = supervised_loss
        
        if with_UDA:
            final_loss += lam * consistency_loss

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        return {
            'supervised batch loss': supervised_loss.item(),
            'consistency batch loss': consistency_loss.item() if with_UDA else 0.0,
            'final batch loss': final_loss.item(),
        }    

    trainer = Engine(uda_process_function)

    if with_UDA and with_tsa:
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_tsa(engine):
            step = engine.state.iteration - 1
            if step % 50 == 0:
                mlflow.log_metric("TSA threshold", tsa.thresholds[step].item(), step=step)
                mlflow.log_metric("TSA selection", engine.state.tsa_log['new_y_pred'].shape[0], step=step)
                mlflow.log_metric("Original X Loss", engine.state.tsa_log['loss'], step=step)
                mlflow.log_metric("TSA X Loss", engine.state.tsa_log['tsa_loss'], step=step)
    
    if not hasattr(lr_scheduler, "step"):
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    else:
        trainer.add_event_handler(Events.ITERATION_STARTED, lambda engine: lr_scheduler.step())

    @trainer.on(Events.ITERATION_STARTED)
    def log_learning_rate(engine):
        step = engine.state.iteration - 1
        if step % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning rate", lr, step=step)

    metric_names = [
        'supervised batch loss',
        'consistency batch loss',
        'final batch loss'
    ]

    def output_transform(x, name):
        return x[name]    

    for n in metric_names:
        RunningAverage(output_transform=partial(output_transform, name=n), epoch_bound=False).attach(trainer, n)

    ProgressBar(persist=True, bar_format="").attach(trainer,
                                                    event_name=Events.EPOCH_STARTED,
                                                    closing_event_name=Events.COMPLETED)

    tb_logger = TensorboardLogger(log_dir=output_path)
    tb_logger.attach(trainer,
                     log_handler=tbOutputHandler(tag="train", metric_names=['final batch loss', 'consistency batch loss', 'supervised batch loss']),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=tbOptimizerParamsHandler(optimizer, param_name="lr"),
                     event_name=Events.ITERATION_STARTED)

    metrics = {
        "accuracy": Accuracy(),
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    
    def run_validation(engine, val_interval):
        if (engine.state.epoch - 1) % val_interval == 0:
            train_evaluator.run(train_labelled_loader)
            evaluator.run(test_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED, run_validation, val_interval=2)
    trainer.add_event_handler(Events.COMPLETED, run_validation, val_interval=1)

    tb_logger.attach(train_evaluator,
                     log_handler=tbOutputHandler(tag="train",
                                                 metric_names=list(metrics.keys()),
                                                 another_engine=trainer),
                     event_name=Events.COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=tbOutputHandler(tag="test",
                                                 metric_names=list(metrics.keys()),
                                                 another_engine=trainer),
                     event_name=Events.COMPLETED)

    def mlflow_batch_metrics_logging(engine, tag):
        step = trainer.state.iteration
        for name, value in engine.state.metrics.items():
            mlflow.log_metric("{} {}".format(tag, name), value, step=step)

    def mlflow_val_metrics_logging(engine, tag):
        step = trainer.state.epoch
        for name in metrics.keys():
            value = engine.state.metrics[name]
            mlflow.log_metric("{} {}".format(tag, name), value, step=step)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, mlflow_batch_metrics_logging, "train")
    train_evaluator.add_event_handler(Events.COMPLETED, mlflow_val_metrics_logging, "train")
    evaluator.add_event_handler(Events.COMPLETED, mlflow_val_metrics_logging, "test")

    trainer.run(train_labelled_loader, max_epochs=config['num_epochs'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a CNN on a dataset")
    
    parser.add_argument('dataset', type=str, choices=['CIFAR10', 'CIFAR100'],
                        help="Training/Testing dataset")

    parser.add_argument('network', type=str, help="CNN to train")

    parser.add_argument('--params', type=str,
                        help='Override default configuration with parameters: '
                        'data_path=/path/to/dataset;batch_size=64;num_workers=12 ...')

    args = parser.parse_args()

    dataset_name = args.dataset    
    network_name = args.network    
    
    print("Train {} on {}".format(network_name, dataset_name))    
    print("- PyTorch version: {}".format(torch.__version__))
    print("- Ignite version: {}".format(ignite.__version__))
    
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print("- CUDA version: {}".format(torch.version.cuda))

    batch_size = 64
    num_epochs = 200
    config = {
        "dataset": dataset_name,
        "data_path": ".",

        "model": network_name,

        "momentum": 0.9,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "unlabelled_batch_size": 320,
        "num_workers": 10,

        "num_epochs": num_epochs,
        
        "learning_rate": 0.03,
        "min_lr_ratio": 0.004,
        "num_warmup_steps": 0,

        "num_labelled_samples": 4000,
        "consistency_lambda": 1.0,
        "consistency_criterion": "KL",

        "with_TSA": True,
        "TSA_proba_min": 0.1,
        "TSA_proba_max": 1.0,

        "no_UDA": False,  # disable UDA training 
    }

    # Override config:
    if args.params:
        for param in args.params.split(";"):
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    print("\n")
    print("Configuration:")
    for key, value in config.items():
        print("\t{}: {}".format(key, value))
    print("\n")

    mlflow.log_params(config)

    # dump all python files to reproduce the run
    mlflow.log_artifacts(Path(__file__).parent.as_posix())

    with tempfile.TemporaryDirectory() as tmpdirname:                        
        try:
            run(tmpdirname, config)
        except Exception as e:
            traceback.print_exc()
            mlflow.log_artifacts(tmpdirname)
            mlflow.log_param("run status", "FAILED")
            exit(1)

        mlflow.log_artifacts(tmpdirname)
        mlflow.log_param("run status", "OK")
