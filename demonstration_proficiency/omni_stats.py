import glob
import json
from collections import defaultdict

from tqdm import tqdm

take_ids = set()

with open('/mnt/nas/abrsh/egoexo.json') as f:
    take_vid_mapping = json.load(f)

take_vid_mapping = take_vid_mapping['database']

uids = [t['take_uid'] for t in take_vid_mapping.values()]
print(len(uids))
print(len(set(uids)))

omni_dir = '/mnt/nas/abrsh/omni'
cam_types = defaultdict(lambda: list())
stream_types = defaultdict(lambda: list())

cam_set = defaultdict(lambda: 0)
stream_set = defaultdict(lambda: 0)

for uid in tqdm(uids):
    results = glob.glob(f"{uid}*", root_dir=omni_dir)
    for res in results:
        take_id, cam_id, stream_id = res.split("_")
        cam_types[take_id].append(cam_id)
        stream_types[take_id].append(stream_id)
        cam_set[cam_id] += 1
        stream_set[stream_id] += 1


lengths_cam = [len(v) for v in cam_types.values()]
lengths_stream = [len(v) for v in stream_types.values()]
print(set(lengths_cam))
print(set(lengths_stream))

aria_names = ['aria01', 'Aria', 'aria06', 'aria02', 'aria03', 'aria05', 'aria04', 'aria']
