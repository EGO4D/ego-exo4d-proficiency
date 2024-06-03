import argparse
import copy
import random
import string
import os
import time
import datetime
from pprint import pprint
import wandb

# torch imports
import torch
# import torch._dynamo
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.modeling.losses import init_config
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed
from libs.utils.envutils import source_bash_file


import ray
from ray import train, tune
from ray.air import session
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

# our code


def tune_main(args, cfg):
    # torch._dynamo.config.verbose = True
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    # pprint(cfg)
    init_config(cfg)
    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        print(f"Creating folder {ckpt_folder}")
        os.makedirs(ckpt_folder)
    print(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    print("---------------------------------------------------------------------------------------------")
    print(cfg['model']['input_dim'])
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = model.to('cuda:0')
    # model = torch.compile(model)
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    wandb.login()
    wandb.init(config=cfg, project='tune-egoexo_omni')
    wandb.watch(model, log_freq=100, log_graph=True, idx=0)
    wandb.watch(model_ema, log_freq=100, log_graph=True, idx=1)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            # args.start_epoch = checkpoint['epoch']
            ckpt = checkpoint['state_dict']
            del ckpt['module.cls_head.cls_head.conv.weight']
            del ckpt['module.cls_head.cls_head.conv.bias']
            model.load_state_dict(ckpt, strict=False)
            ckpt = checkpoint['state_dict_ema']
            del ckpt['module.cls_head.cls_head.conv.weight']
            del ckpt['module.cls_head.cls_head.conv.bias']
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'], strict=False)
            # also load the optimizer / scheduler if necessary
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        checkpoint = torch.load("/home/abrham/ego4d_omnivore_egovlp_reproduce/epoch_010.pth.tar",
            map_location = lambda storage, loc: storage.cuda(
                cfg['devices'][0]))
        # args.start_epoch = checkpoint['epoch']
        ckpt = checkpoint['state_dict']
        del ckpt['module.cls_head.cls_head.conv.weight']
        del ckpt['module.cls_head.cls_head.conv.bias']
        model.load_state_dict(ckpt, strict=False)
        ckpt = checkpoint['state_dict_ema']
        del ckpt['module.cls_head.cls_head.conv.weight']
        del ckpt['module.cls_head.cls_head.conv.bias']
        model_ema.module.load_state_dict(checkpoint['state_dict_ema'], strict=False)


    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    val_db_vars = val_dataset.get_attributes()
    det_eval, output_file = None, None
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds=val_db_vars['tiou_thresholds']
    )
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        final_loss = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        # final_loss = 0.0
        mAP = valid_one_epoch(
            val_loader,
            model,
            -1,
            evaluator=det_eval,
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=None,
            print_freq=args.print_freq
        )
        session.report({"MAP": mAP * 100, "final_loss": final_loss})
        # wandb.log({'MAP': mAP * 100})
        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return


################################################################################
# @ray.remote(num_gpus=1)
def train_and_val(config, args):
    global cfg
    cfg = copy.deepcopy(cfg)
    # print(cfg['output_folder'])
    if 'lr' in config:
        cfg['opt']['learning_rate'] = config['lr']
    cfg['opt']['epochs'] = 10
    cfg['opt']['warmup_epochs'] = 5
    if 'mode' in config:
        cfg['dataset']['mode'] = config['mode']
    cfg['flags']['report_ray'] = True
    if 'loss_weight' in config:
        cfg["train_cfg"]["loss_weight"] = config["loss_weight"]
    print(cfg['dataset'])

    if cfg['dataset']['mode'] == 'exo':
        cfg['dataset']['input_dim'] = 4 * 1536
    elif cfg['dataset']['mode'] == 'ego_exo':
        cfg['dataset']['input_dim'] = 5 * 1536
    if "max_seg_num" in config:
        cfg["test_cfg"]["max_seg_num"] = int(round(config["max_seg_num"]))

    cfg["model"]["input_dim"] = cfg["dataset"]["input_dim"]
    cfg["model"]["num_classes"] = cfg["dataset"]["num_classes"]
    cfg["model"]["max_seq_len"] = cfg["dataset"]["max_seq_len"]
    cfg["model"]["train_cfg"] = cfg["train_cfg"]
    cfg["model"]["test_cfg"] = cfg["test_cfg"]
    args.output = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))

    # breakpoint()
    tune_main(args, cfg)


def main(args, num_samples=10, max_num_epochs=1, gpus_per_trial=2):
    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-3, 1e-2),
        # "mode": tune.choice(['ego', 'exo', 'ego_exo']),
        "mode": tune.choice(['exo', 'ego_exo']),
        # "batch_size": tune.choice([2, 4, 8, 16])
        # "loss_weight": tune.loguniform(1e-15, 1),
        "max_seg_num": tune.loguniform(150, 400),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=max_num_epochs//2,
        reduction_factor=4)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_and_val, args=args),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="MAP",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    best_result = results.get_best_result("MAP", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final MAP: {}".format(
        best_result.metrics["MAP"]))
    # print("Best trial final validation accuracy: {}".format(
    #     best_result.metrics["accuracy"]))


if __name__ == '__main__':
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
    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": 1e-5,
        "mode": "ego",
        # "batch_size": tune.choice([2, 4, 8, 16])
        # "loss_weight": tune.loguniform(1e-15, 1),
        "max_seg_num": 50,
    }
    # futures = [train_and_val.remote(config, args) for i in range(200)]
    # print(ray.get(futures))

    main(args=args, num_samples=2000, max_num_epochs=15, gpus_per_trial=0.5)
