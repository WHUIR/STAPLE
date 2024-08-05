import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2024, help='random seed')

    # Data Splits
    parser.add_argument("--dataset", type=str, default='Office')
    parser.add_argument("--item_count", type=int, default=0)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--user_k', type=int, default=5)
    parser.add_argument('--item_k', type=int, default=5)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--nega_count', type=int, default=10)
    parser.add_argument('--train_nega_count', type=int, default=10)
    parser.add_argument('--target_domain', type=str, default=None)
    parser.add_argument('--token_ratio', type=float, default=1.0)
    parser.add_argument('--debias', action='store_true')

    # Checkpoint
    parser.add_argument('--output', type=str, default='./ckp/')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--save_by_step', type=int, default=3000, help='save model by step or epoch')

    # CPU/GPU
    # parser.add_argument("--multdiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_false')
    parser.add_argument("--multiGPU", action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--port', type=int, default=12347)
    parser.add_argument('--valid_ratio', type=float, default=1.0)

    # Model Config
    parser.add_argument('--valid_first', action='store_true')
    parser.add_argument('--root_path', type=str, default='plm/')
    parser.add_argument('--backbone', type=str, default='opt-125m')
    parser.add_argument('--debias_backbone', type=str, default='opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_seq_length', type=int, default=10)
    parser.add_argument('--max_token_length', type=int, default=1024)
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--sim_method', type=str, default='dot')
    parser.add_argument('--scaled_dot', action='store_true')


    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--skip_valid', action='store_true')
    parser.add_argument('--use_cache', action='store_false')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--debias_alpha', type=float, default=0)
    parser.add_argument('--distill_type', type=int, default=3)
    parser.add_argument('--valid_all', action='store_true')
    parser.add_argument('--teacher_embs', type=str, default='None')
    parser.add_argument('--nega_strategy', type=str, default='random')
    parser.add_argument('--train_stage', type=int, default=1)
    parser.add_argument('--sample_alpha', type=float, default=1.0)

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
  
    # set some parameters
    if args.train_stage == 1:
        args.sample_alpha = 1.0
        args.epoch = 100
    elif args.train_stage == 2:
        args.sample_alpha = 0.0
        args.epoch = 5
        args.skip_valid = True
    elif args.train_stage == 3:
        args.sample_alpha = 1.0
        args.epoch = 100
        if args.dataset == 'Office':
            args.debias_alpha = 1.5
        elif args.dataset in ['Arts', 'Food']:
            args.debias_alpha = 1.0
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    args.gradient_accumulation_steps = int(96 / args.num_gpus / args.batch_size)
    return args

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
