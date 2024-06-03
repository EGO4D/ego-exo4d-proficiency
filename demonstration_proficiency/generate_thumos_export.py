"""
python -m gen_take_vid_mapping_ego_exo /home/abrham/egoexo-v2-exports/
"""

import os
import glob
import json
import argparse
import numpy as np

from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_take_uid_duration_mapping(export_dir, output_json=False):
    takes_json_path = os.path.join(export_dir, 'takes.json')
    with open(takes_json_path) as f:
        take_json = json.load(f)

    mapping = dict()
    view_counts = list()
    for vid in take_json:
        exos = list()
        ego = list()
        temp_ego = list()
        temp_exo = list()
        for val_cam in vid['capture']['cameras']:
            if val_cam['cam_id'] not in ['aria', 'Aria', 'aria01', 'aria02', 'aria03', 'aria04', 'aria05', 'aria06', 'aria07', 'aria09', 'aria1', 'cam01', 'cam02', 'cam03', 'cam04', 'cam05', 'gp01', 'gp02', 'gp03', 'gp04', 'gp05', 'gp06']:
                continue

            if val_cam['is_ego'] == True:
                ego.append(val_cam['cam_id'])
            else:
                exos.append(val_cam['cam_id'])
        # assert len(exos) == 1
        # assert len(ego) > 0
        root_dir = vid['root_dir']

        if '/' in root_dir:
            root_dir = vid['root_dir'].split('/')[-1]

        if len(ego) == 0 or len(exos) < 4:
            continue
        view_counts.append(len(ego[:1]) + len(exos[:4]))


        mapping[root_dir] = {
            'take_uid': vid['take_uid'],
            'duration': vid['duration_sec'],
            'ego': ego[0],
            'exos': exos[:4],
        }

    print(np.unique(view_counts))
    print(len(mapping))

    if output_json:
        out_path = os.path.join(export_dir, "take_to_vid_name.json")
        print(f"Writing to {out_path}")
        with open(out_path, 'w') as f:
            json.dump(mapping, f)

    return mapping


def get_split_annotations(all_videos, split, take_vid_mapping):
    annotation = defaultdict(lambda: dict())
    #
    # with open('/mnt/nas/abrsh/take_to_vid_name.json') as f:
    #     take_vid_mapping = json.load(f)
    #
    #
    total_time = 0.0
    total_ann_count = 0
    ann_count_per_video = list()
    #
    # from_json = 0
    #
    exo_count = defaultdict(lambda: 0)
    intervals = list()
    skipped_videos = list()

    for vid in tqdm(all_videos):
        # all_samples = new_df[new_df['video_name'] == vid]
        # if vid not in take_vid_mapping:
        #     continue
        # exo_views = take_vid_mapping[vid]['exos']
        # exo_count[len(exo_views)] += 1
        # if len(exo_views) < 4:
        #     continue
        #
        # annotation[vid] = dict()
        # annotation[vid]['annotations'] = list()
        # annotation[vid]['subset'] = np.random.choice(['Train', 'validation', 'Test'], p=[0.7, 0.15, 0.15])
        # if vid in take_vid_mapping:
        #     annotation[vid]['duration'] = take_vid_mapping[vid]['duration']
        #     annotation[vid]['take_uid'] = take_vid_mapping[vid]['take_uid']
        #     from_json += 1
        # else:
        #     continue
        #     annotation[vid]['duration'] = durations[vid]
        # annotation[vid]['fps'] = fps[vid]
        # annotation[vid]['ego'] = take_vid_mapping[vid]['ego'],
        # annotation[vid]['exos'] = take_vid_mapping[vid]['exos'][:4],
        #
        # add_time = True
        # sorted_times = all_samples['video_time'].sort_values()
        # separation = sorted_times[1:].values - sorted_times[:-1].values
        # intervals.extend(separation.tolist())
        annotations_per_vid = 0
        take_uid = vid['take_uid']
        video_paths = vid['video_paths']
        video_name = video_paths['ego'].split('/')[1]

        if video_name not in take_vid_mapping:
            skipped_videos.append(video_name)
            continue

        scenario = vid['scenario_name']
        annotation[video_name]['annotations'] = list()
        annotation[video_name]['subset'] = split
        annotation[video_name]['fps'] = 30.0
        # On arxivb and later

        annotation[video_name]['duration'] = take_vid_mapping[video_name]['duration']
        total_time += annotation[video_name]['duration']
        annotation[video_name]['ego'] = [take_vid_mapping[video_name]['ego']]
        annotation[video_name]['exos'] = [take_vid_mapping[video_name]['exos']]
        annotation[video_name]['take_uid'] = take_uid
        for label in ['good_executions', 'tips_for_improvement']:
            for i, sample in enumerate(vid[label]):
                # annotation_count.append(all_samples.shape[0])
                video_name = sample['video_name']
                label_id = 0 if label == 'good_executions' else 1
                total_ann_count += 1

                try:
                    annotation[video_name]['annotations'].append({
                        'scenario': scenario,
                        # 'user_id': sample['user_id'],
                        'label_id': label_id,
                        'label': label,
                        'segment': sample['video_time'],
                        'id': vid,
                        'take_id': take_uid,
                    })
                    total_ann_count += 1
                    annotations_per_vid += 1
                except KeyError:
                    print(i, sample)
                    add_time = False
                    break
            ann_count_per_video.append(annotations_per_vid)
        # if add_time:
        #     total_time += durations[vid]

    print("Average annotations interval", total_time / total_ann_count, split)
    print(len(annotation))
    print(skipped_videos)
    return annotation


if __name__ == '__main__':
    __THUMOS14_SPLITS = {
        'train': 'Train',
        'val': 'Validation',
        'test': 'Test'
    }

    parser = argparse.ArgumentParser()

    parser.add_argument('export_dir', type=str)
    parser.add_argument('data_version', type=str, default='arxivb')
    args = parser.parse_args()

    mapping = get_take_uid_duration_mapping(args.export_dir, output_json=False)

    split_files = glob.glob(args.export_dir + "/proficiency_demonstration_*.json")

    print(split_files)

    annotation = defaultdict(lambda: dict())

    for export in split_files:
        split = export.split('.')[0].split('_')[2]
        thumos_split = __THUMOS14_SPLITS[split]

        with open(export) as f:
            all_videos = json.load(f)['annotations']

        annotations_split = get_split_annotations(all_videos, thumos_split, mapping)
        annotation = {**annotations_split, **annotation}

    ann = dict()
    ann['database'] = annotation

    out_path = os.path.join(args.export_dir, f"egoexo_{args.data_version}.json")

    with open(out_path, 'w') as f:
        json.dump(ann, f)

