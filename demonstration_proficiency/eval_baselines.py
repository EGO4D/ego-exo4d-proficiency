import json
import pickle
# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# import ray
# from ray import train, tune
# from ray.air import session

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.modeling.losses import init_config
from libs.utils import ANETdetection, fix_random_seed
from libs.utils.envutils import source_bash_file
from libs.utils.postprocessing import postprocess_results

def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    baseline="positive",
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)
    with open('/mnt/nas/abrsh/egoexo.json') as f:
        annotations = json.load(f)['database']

    # set up meters
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        'times' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # if iter_idx > 100:
        #     break
        # forward the model (wo. grad)
        num_vids = len(video_list)
        for vid in range(num_vids):
            vid_ann = video_list[vid]

            if baseline == 'annotation':
                ann = annotations[vid_ann['video_id']]
                times = torch.as_tensor([a['segment'] for a in ann['annotations']])
            else:
                times = torch.arange(0, vid_ann['duration'], 2.14)
                # times = torch.arange(0, vid_ann['duration'], 5.992)
            scores = torch.ones_like(times)
            if baseline == 'positive':
                labels = torch.ones_like(times)
            elif baseline == 'negative':
                labels = torch.zeros_like(times)
            elif baseline == 'random':
                labels = torch.randint_like(times, 0, 2)
            elif baseline == 'annotation':
                labels = torch.as_tensor([a['label_id'] for a in ann['annotations']])
            results['video-id'].extend(
                [vid_ann['video_id']] *
                times.shape[0]
            )
            results['times'].append(times)

            results['score'].append(scores)
            results['label'].append(labels)


        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            start = time.time()

            # print timing
            # print('Test: [{0:05d}/{1:05d}]\t'
            #       'Time {batch_time.val:.2f} '.format(
            #       iter_idx, len(val_loader)))
    # gather all stats and evaluate
    # results['t-start'] = torch.cat(results['t-start']).numpy()
    results['times'] = torch.cat(results['times']).numpy()
    # results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        # _, mAP, _ = evaluator.evaluate(results, verbose=True)
        print("Evaluation started")
        _, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP
################################################################################
def main(args, cfg):
    """0. load config"""

    pprint(cfg)
    init_config(cfg)

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

    """3. create model and evaluator"""
    # model

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )
    else:
        pass

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        None,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=10,
        baseline=args.baseline,
    )
    end = time.time()
    # if cfg['flags']['report_ray']:
    #     session.report({"MAP": mAP})

    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    source_bash_file('/home/abrham/mistake/bin/activate')
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-b', '--baseline', default="positive", type=str,
                        help='max number of output actions (default: -1)')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    args = parser.parse_args()
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    main(args, cfg)
