import numpy as np
import torch

from interaction import Interaction


class PLMEmb:
    def __init__(self, config):
        self.mmap_idx_path = config['mmap_idx_path']
        self.mmap_emb_path = config['mmap_emb_path']
        self.mmap_idx_shape = config['mmap_idx_shape']
        self.mmap_emb_shape = config['mmap_emb_shape']
        self.device = config['device']

    # TODO: can we eliminate 'dataset' parameter here?
    def __call__(self, dataset, interaction):
        item_seq = interaction['item_id_list']
        pos_item = interaction['item_id']
        idx_mmap = np.memmap(self.mmap_idx_path, dtype=np.int32, shape=self.mmap_idx_shape)
        data_mmap = np.memmap(self.mmap_emb_path, dtype=np.float32, shape=self.mmap_emb_shape)
        item_emb_seq = torch.Tensor(data_mmap[idx_mmap[item_seq]])
        pos_item_emb = torch.Tensor(data_mmap[idx_mmap[pos_item]])
        interaction.update(Interaction(
            {
                'item_emb_list': item_emb_seq,
                'pos_item_emb': pos_item_emb
            }
        ))
        return interaction

    def idx_convert(self, index):
        """
        convert item index to mmap index
        """
        idx_mmap = np.memmap(self.mmap_idx_path, dtype=np.int32, shape=self.mmap_idx_shape)
        return idx_mmap[index]

    def get_scores(self, seq_emb: torch.Tensor, transform: torch.nn.Module, stride: int = 1024 * 1024):
        num_items = self.mmap_emb_shape[0]
        scores = []
        for i in range(num_items // stride + 1):
            data_mmap = np.memmap(self.mmap_emb_path, dtype=np.float32, shape=self.mmap_emb_shape)
            if i * stride == num_items:
                break
            end_idx = (i + 1) * stride if (i + 1) * stride < num_items else num_items
            partial_target_emb = transform(torch.tensor(data_mmap[:end_idx], dtype=torch.float32).to(self.device))
            # normalize & multiply
            seq_emb = torch.nn.functional.normalize(seq_emb, dim=-1)
            partial_target_emb = torch.nn.functional.normalize(partial_target_emb, dim=-1)
            partial_scores = seq_emb @ partial_target_emb.T
            # save result to scores matrix
            scores.append(partial_scores)
        scores = torch.cat(scores, dim=-1)
        return scores

