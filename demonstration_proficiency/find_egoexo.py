import os
import glob

config = glob.glob('/home/abrham/actionformer_release/ckpt/**/config.txt', recursive=True)
config.sort(key=os.path.getmtime, reverse=True)
# print(config)
egoexo_list = list()
for con in config:
    with open(con) as f:
        data = f.readlines()
    # print(data[11])

    if 'ego_exo' in data[11]:
        # print(data[11])
        base = os.path.dirname(con)
        pth = os.path.join(base, 'epoch_005.pth.tar')
        # print(pth)
        if os.path.exists(pth):
            egoexo_list.append(pth)

# print(egoexo_list)

egoexo_list.sort(key=os.path.getsize, reverse=True)

print(egoexo_list)


