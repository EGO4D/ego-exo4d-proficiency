from collections import defaultdict
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# df = pd.read_csv('/home/abrham/GPT_preds.csv')
# print(df.shape)
# df = df.dropna(subset=['video_name'])
# print(df.shape)
# df.reset_index(inplace=True)
# print(df.columns)
#
# selected_columns = ['scenario', 'user_id', 'video_name', 'video_time', 'good execution', 'tip for improvement']
#
# new_df = df[selected_columns]
#
# new_df.loc[:, 'good execution'] = new_df.loc[:, 'good execution'].fillna(0)
# new_df.loc[:, 'tip for improvement'] = new_df.loc[:, 'tip for improvement'].fillna(0)
# new_df.loc[:, 'tip for improvement'] = new_df.loc[:, 'tip for improvement'].apply(lambda g: g if g != -1 else 0)
# new_df.loc[:, 'good execution'] = new_df.loc[:, 'good execution'].apply(lambda g: g if g != -1 else 0)
#
# # print(new_df.shape)
# label_sums = new_df[selected_columns[-2:]].sum(axis=1)
#
# idxs = np.where(label_sums > 1)[0].tolist()
# idxs_zero = np.where(label_sums == 0)[0].tolist()
#
# drop_idxs = idxs + idxs_zero
#
# print(new_df.shape, "Before drop")
# new_df = new_df.drop(drop_idxs, axis=0)
# print(new_df.shape, "After drop")
# label_sums = new_df[selected_columns[-2:]].sum(axis=1)
# assert np.all(label_sums <= 1)
# # print(new_df.head())
#
# positive = new_df[new_df['good execution'] == 1.0]
# print("Positive count", positive.shape[0])
# negative = new_df[new_df['tip for improvement'] == 1.0]
# print("Negative count", negative.shape[0])
#
# multi_labelled = new_df[(new_df['tip for improvement'] == 1.0) & (new_df['good execution'] == 1.0)]
# print("Multilabelled count", multi_labelled.shape[0])
#
# print(new_df['good execution'].unique())
# print(new_df['tip for improvement'].unique())
#
# new_df['label_id'] = new_df['good execution'].apply(lambda g: int(g != 1))
#
# print(new_df['label_id'].sum())
#
#
# def gen_label_names(f):
#     if f == 0:
#         return 'good execution'
#     return 'tip for improvement'
#
#
# selected_columns = ['scenario', 'user_id', 'video_name', 'video_time', 'label_id']
#
# new_df = new_df[selected_columns]
# new_df['label'] = new_df['label_id'].apply(gen_label_names)
#
# unique_videos = new_df['video_name'].unique()
# print("Video count:", len(unique_videos))
#
with open('/home/abrham/egoexo_arxiv_duration.json') as f:
    durations = json.load(f)

with open('/mnt/nas/abrsh/take_to_vid_name.json') as f:
    take_vid_mapping = json.load(f)


def get_split_annotations(all_videos, split):
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
        scenario = vid['scenario_name']
        annotation[video_name]['annotations'] = list()
        annotation[video_name]['subset'] = split
        annotation[video_name]['fps'] = 30.0
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
    return annotation

jsons = {
    'Train': '/home/abrham/demonstration_arxiv23_train.json',
    'Validation': '/home/abrham/demonstration_arxiv23_val.json',
    'Test': '/home/abrham/demonstration_arxiv23_test.json',
}

annotation = defaultdict(lambda: dict())

for split, export in jsons.items():
    with open(export) as f:
        all_videos = json.load(f)

    annotations_split = get_split_annotations(all_videos, split)
    annotation = {**annotations_split, **annotation}


# print(exo_count)
# median = np.median(intervals)
# plt.hist(intervals, bins=100, density=True)
# plt.savefig("intervals.png")
# plt.figure()
# plt.hist(ann_count_per_video, bins=100, density=True)
# plt.savefig("ann_count.png")
# plt.show()
# print("Max/Min/Mean/Median annotation count", max(ann_count_per_video), min(ann_count_per_video),
#       sum(ann_count_per_video)/len(ann_count_per_video), np.median(ann_count_per_video))
# print("Median annotation interval (seconds)", median)
# print("Min annotation interval (seconds)", min(intervals))
# print("Average annotations interval", total_time / total_ann_count)
# print(f"{from_json} durations used from Ego-Exo take JSON")
# # print(annotation)
ann = dict()
ann['database'] = annotation
# print(annotation[vid])

# with open('/mnt/nas/abrsh/egoexo_arxiv_gen.json', 'w') as f:
#     json.dump(ann, f)

# plt.hist(annotation_count, bins=50, density=True)
#
# plt.show()