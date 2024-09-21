import torch
import numpy as np
import os
import sys
from argparse import ArgumentParser
from omegacli import parse_config, OmegaConf
from libcll.models import build_model
from libcll.strategies import build_strategy
from libcll.datasets import prepare_cl_data_module
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.cli import LightningCLI
import random


def main(args):
    print("Preparing Dataset......")
    pl.seed_everything(args.training.seed, workers=True)
    cl_data_module = prepare_cl_data_module(
        args.dataset._name,
        batch_size=args.training.batch_size,
        valid_split=args.training.valid_split,
        valid_type=args.training.valid_type,
        one_hot=(args.strategy._name == "MCL"),
        num_cl=args.dataset.num_cl,
        transition_matrix=args.dataset.transition_matrix,
        augment=args.dataset.augment,
        noise=args.dataset.noise,
        seed=args.training.seed,
        ssl=args.training.ssl, 
        samples_per_class=args.training.samples_per_class, 
    )
    cl_data_module.prepare_data()
    cl_data_module.setup(stage="fit")

    input_dim, num_classes = cl_data_module.train_set.input_dim, cl_data_module.train_set.num_classes
    Q, class_priors = cl_data_module.get_distribution_info()
    print("Preparing Model......")

    pl.seed_everything(args.training.seed, workers=True)
    model = build_model(
        args.model._name,
        input_dim=input_dim,
        hidden_dim=args.model.hidden_dim,
        num_classes=num_classes,
        depth=args.model.depth, 
        widen_factor=args.model.widen_factor, 
    )

    strategy = build_strategy(
        args.strategy._name,
        model=model,
        valid_type=args.training.valid_type,
        num_classes=num_classes,
        type=args.strategy.type,
        lr=args.optimizer.lr,
        weight_decay=args.optimizer.weight_decay, 
        Q=Q,
        class_priors=class_priors,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.training.output_dir)
    checkpoint_callback_best = ModelCheckpoint(
        monitor=f"Valid_{args.training.valid_type}",
        dirpath=args.training.output_dir,
        filename=f"{{epoch}}-{{Valid_{args.training.valid_type}:.2f}}",
        save_top_k=1,
        mode="max" if args.training.valid_type == "Accuracy" else "min",
        every_n_epochs=args.training.eval_epoch if not args.training.ssl else None, 
        every_n_train_steps=5000 if args.training.ssl else None, 
    )
    checkpoint_callback_last = ModelCheckpoint(
        monitor=f"step",
        dirpath=args.training.output_dir,
        filename="{epoch}-{step}",
        save_top_k=1,
        mode="max",
        every_n_epochs=args.training.eval_epoch if not args.training.ssl else None,
        every_n_train_steps=5000 if args.training.ssl else None, 
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    print("Start Training......")
    trainer = pl.Trainer(
        max_epochs=args.training.epoch if not args.training.ssl else None,
        max_steps=args.training.max_steps, 
        accelerator="gpu",
        logger=tb_logger,
        log_every_n_steps=args.training.log_step,
        deterministic=True,
        check_val_every_n_epoch=args.training.eval_epoch if not args.training.ssl else None,
        val_check_interval=5000 if args.training.ssl else None, 
        callbacks=[checkpoint_callback_best, checkpoint_callback_last, lr_monitor],
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto", 
        use_distributed_sampler=False, 
        # num_nodes=2, 
    )
    if args.training.do_train:
        with open(f"{args.training.output_dir}/config.yaml", "w") as f:
            OmegaConf.save(args, f)
        trainer.fit(
            strategy,
            datamodule=cl_data_module, 
        )
    if args.training.do_predict:
        if args.training.do_train:
            trainer.test(datamodule=cl_data_module, ckpt_path="best")
        else:
            trainer.test(
                strategy, datamodule=cl_data_module, ckpt_path=args.training.model_path
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        dest="training.config_file",
        type=str,
        default="libcll/configs/base.yaml",
    )
    parser.add_argument("--model", dest="model._name", type=str, default="dense")
    parser.add_argument(
        "--model_path", dest="training.model_path", type=str, default=None
    )
    parser.add_argument("--dataset", dest="dataset._name", type=str, default="mnist")
    parser.add_argument(
        "--valid_type", dest="training.valid_type", type=str, default="URE"
    )
    parser.add_argument("--num_cl", dest="dataset.num_cl", type=int, default=1)
    parser.add_argument(
        "--valid_split", dest="training.valid_split", type=float, default=0.1
    )
    parser.add_argument(
        "--eval_epoch", dest="training.eval_epoch", type=int, default=10
    )
    parser.add_argument(
        "--output_dir", dest="training.output_dir", type=str, default=None
    )
    parser.add_argument(
        "--batch_size", dest="training.batch_size", type=int, default=256
    )
    parser.add_argument("--hidden_dim", dest="model.hidden_dim", type=int, default=500)
    parser.add_argument("--depth", dest="model.depth", type=int, default=28)
    parser.add_argument("--widen_factor", dest="model.widen_factor", type=int, default=2)
    parser.add_argument("--epoch", dest="training.epoch", type=int, default=300)
    parser.add_argument("--max_steps", dest="training.max_steps", type=int, default=2 ** 20)
    parser.add_argument("--do_train", dest="training.do_train", action="store_true")
    parser.add_argument("--do_predict", dest="training.do_predict", action="store_true")
    parser.add_argument("--ssl", dest="training.ssl", action="store_true")
    parser.add_argument("--samples_per_class", dest="training.samples_per_class", type=int, default=10)
    parser.add_argument("--strategy", dest="strategy._name", type=str, default="SCL")
    parser.add_argument("--type", dest="strategy.type", type=str, default=None)
    parser.add_argument("--lr", dest="optimizer.lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", dest="optimizer.weight_decay", type=float, default=5e-4)
    parser.add_argument("--augment", dest="dataset.augment", action="store_true")
    parser.add_argument(
        "--transition_matrix",
        dest="dataset.transition_matrix",
        type=str,
        default="uniform",
    )
    parser.add_argument("--seed", dest="training.seed", type=int, default=1126)
    parser.add_argument("--log_step", dest="training.log_step", type=int, default=50)
    parser.add_argument("--noise", dest="dataset.noise", type=float, default=0.1)
    args = parser.parse_args()
    args = parse_config(
        parser, getattr(args, "training.config_file"), args=sys.argv[1:]
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    args.training.batch_size = int(args.training.batch_size / torch.cuda.device_count())
    os.makedirs(args.training.output_dir, exist_ok=True)
    main(args)
