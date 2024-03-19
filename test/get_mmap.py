import os
import numpy as np
from tqdm import tqdm


data_dir = '../dataset/downstream/Scientific'

mmap_idx_path = '../dataset/customized/Scientific/mmap/mmap_idx'
mmap_emb_path = '../dataset/customized/Scientific/mmap/mmap_emb'

feat_file = os.path.join(data_dir, 'Scientific.feat1CLS')

feature = np.fromfile(feat_file, dtype=np.float32).reshape(-1, 768)
zero_feat = np.zeros(shape=(1, 768))
feature = np.concatenate((zero_feat, feature), axis=0)

idx = np.arange(feature.shape[0])

# turn this feature into mmap
id_mmap = np.memmap(mmap_idx_path + f'_{np.max(idx) + 1}', mode="w+", dtype=np.int32, shape=(np.max(idx) + 1,))
emb_mmap = np.memmap(mmap_emb_path + f'_{feature.shape[0]}_{feature.shape[1]}', mode="w+", dtype=np.float32, shape=feature.shape)

for i, item_id in tqdm(enumerate(idx), "create index mmap "):
    id_mmap[item_id] = i
id_mmap.flush()

for i, item_emb in tqdm(enumerate(feature), "create embedding mmap "):
    emb_mmap[i][:] = item_emb[:]
emb_mmap.flush()
