import os
import numpy as np


data_dir = '../dataset/downstream/Scientific'

mmap_idx_path = '../dataset/customized/Scientific/mmap/mmap_idx'
mmap_emb_path = '../dataset/customized/Scientific/mmap/mmap_emb'

feat_file = os.path.join(data_dir, 'Scientific.feat1CLS')

feature = np.fromfile(feat_file, dtype=np.float32).reshape(-1, 768)

idx = 123
print(f"item{idx}: {feature[idx]}")
