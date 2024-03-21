import json
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import os


def build_mmap(
        config,
):
    selected_keys = config['selected_keys'] or ["tag1", "tag2", "title", "description"]

    all_item_ids = []
    all_item_text = []
    all_item_embs = []

    # init tokenizer and model
    device = config['device']
    tokenizer = AutoTokenizer.from_pretrained(config['language_model'])
    model = AutoModel.from_pretrained(config['language_model']).to(device)

    # read json file
    with open(config['metadata_path']) as mdf:
        for line in tqdm(mdf.readlines(), "processing metadata jsonl "):
            # deal with NaN
            item_entry = json.loads(line.replace("NaN", '""').replace("null", '""'))

            # register item id
            all_item_ids.append(item_entry['item_id'])

            # build combined text entry
            item_text = "".join([item_entry[key] + "; " for key in selected_keys])
            all_item_text.append(item_text)

            # get embedding and convert to numpy
            tokenized_text = tokenizer(item_text, return_tensors='pt', padding=True, max_length=512,
                                       truncation=True).to(device)
            text_embedding = model(**tokenized_text).pooler_output.cpu().detach().numpy()
            all_item_embs.append(dict(zip(["item_id", "embedding"], [item_entry['item_id'], text_embedding])))

    # generate mmap files
    ids = all_item_ids
    assert not ids.__contains__(0)  # id=0 should be exclusively used for padding
    ids.insert(0, 0)
    item_ids_array = np.array(ids)

    emb = [item['embedding'][0] for item in all_item_embs]
    emb.insert(0, [0 for i in range(768)])
    item_embs_array = np.array(emb)

    # create index map
    idx_shape_0 = np.max(item_ids_array) + 1
    mmap_idx_path = os.path.join(config['mmap_out'], f"mmap_idx_{idx_shape_0}")
    id_mmap = np.memmap(mmap_idx_path, mode="w+", dtype=np.int32, shape=(np.max(item_ids_array) + 1,))
    for i, item_id in tqdm(enumerate(item_ids_array), "create index mmap "):
        id_mmap[item_id] = i
    id_mmap.flush()

    # create metadata map
    emb_shape_0 = item_embs_array.shape[0]
    emb_shape_1 = item_embs_array.shape[1]
    mmap_emb_path = os.path.join(config['mmap_out'], f"mmap_emb_{emb_shape_0}_{emb_shape_1}")
    emb_mmap = np.memmap(mmap_emb_path, mode="w+", dtype=np.float32, shape=item_embs_array.shape)
    for i, item_emb in tqdm(enumerate(item_embs_array), "create embedding mmap "):
        emb_mmap[i][:] = item_emb[:]
    emb_mmap.flush()

    return mmap_idx_path, (idx_shape_0,), mmap_emb_path, (emb_shape_0, emb_shape_1)
