import os
import argparse
import random
import string

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

import train
import eval
from libs.core import load_config

parser = argparse.ArgumentParser(
    description='Train a point-based transformer for action localization')
parser.add_argument('config', type=str, metavar='DIR',
                    help='path to a config file')

parser.add_argument('-epoch', type=int, default=-1,
                    help='checkpoint epoch')
parser.add_argument('-t', '--topk', default=-1, type=int,
                    help='max number of output actions (default: -1)')
parser.add_argument('--saveonly', action='store_true',
                    help='Only save the ouputs without evaluation (e.g., for test set)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10 iterations)')

parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                    help='checkpoint frequency (default: every 5 epochs)')

parser.add_argument('--output', default='', type=str,
                    help='name of exp folder (default: none)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to a checkpoint (default: none)')

args = parser.parse_args()

# sanity check
if os.path.isfile(args.config):
    cfg = load_config(args.config)
else:
    raise ValueError("Config file does not exist.")


# @ray.remote
def train_and_val(config):
    # print(cfg['output_folder'])
    cfg['opt']['learning_rate'] = config['lr']
    cfg['opt']['epochs'] = 1
    cfg['opt']['warmup_epochs'] = 1

    cfg['flags']['report_ray'] = True
    args.output = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))

    # breakpoint()
    train.main(args, cfg)
    args.ckpt = 'a101_' + args.output
    # print(cfg)
    # print(cfg['opts'])
    eval.main(args, cfg)


def main(num_samples=10, max_num_epochs=1, gpus_per_trial=2):
    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-5, 1e-1),
        # "batch_size": tune.choice([2, 4, 8, 16])
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_and_val),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="MAP",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    best_result = results.get_best_result("MAP", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["MAP"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))


if __name__ == '__main__':
    # config = {
    #     # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #     # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #     "lr": tune.loguniform(1e-5, 1e-1),
    #     # "batch_size": tune.choice([2, 4, 8, 16])
    # }
    # futures = [train_and_val.remote(config) for i in range(2)]
    # print(ray.get(futures))
    main(num_samples=200, max_num_epochs=1, gpus_per_trial=0.2)