import json
import os


data_dir = '../dataset/downstream/Scientific'
src_inter_file = os.path.join(data_dir, 'Scientific.test.inter')
tgt_inter_file = '../dataset/customized/Scientific/interaction.jsonl'
MAX_ITEM_LIST_LEN = 50

interaction = []
for i, line in enumerate(open(src_inter_file)):
    if i == 0:
        continue
    contents = line.split('\t')
    history_seq = contents[1].split(' ')
    history_seq = [int(idx) + 1 for idx in history_seq] # convert string to integer
    target_item = int(contents[-1]) + 1
    history_seq.append(target_item)
    while len(history_seq) < MAX_ITEM_LIST_LEN:
        history_seq.insert(0, 0)    # zero padding at the front
    interaction.append(
        json.dumps({
            'user_id': i,
            'user_sequence': history_seq
        })
    )

with open(tgt_inter_file, "w") as tf:
    for entry in interaction:
        tf.write(entry + '\n')
