import torch
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader

from embedding import PLMEmb
import numpy as np


class CustomizedTrainDataloader(DataLoader):
    def __init__(self, config, dataset, shuffle=False):
        self.config = config
        self.original_dataset = dataset
        self.sample_size = len(dataset)
        self.shuffle = shuffle
        self.plm_embedding = PLMEmb(config)
        self._init_batch_size_and_step()
        self.generator = Generator()
        # distributed scenario
        index_sampler = None
        if not config["single_spec"]:
            index_sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            self.step = max(1, self.step // config["world_size"])
            shuffle = False
        super().__init__(
            dataset=list(range(self.sample_size)),
            batch_size=self.step,
            collate_fn=self.collate_fn,
            num_workers=config['worker'],
            shuffle=shuffle,
            sampler=index_sampler,
            generator=self.generator
        )

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        # todo: distributed scenario
        self.step = batch_size

    def collate_fn(self, index):
        index = np.array(index)
        data = self.original_dataset[index]
        transformed_data = self.plm_embedding(self.original_dataset, data)
        return transformed_data


class CustomizedFullSortEvalDataloader(FullSortEvalDataLoader):
    def __init__(self, dataset, config, sampler, shuffle):
        super().__init__(dataset=dataset, config=config, sampler=sampler, shuffle=shuffle)
        self.original_dataset = dataset
        self.plm_embedding = PLMEmb(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self.original_dataset[index]
        transformed_data = self.plm_embedding(self.original_dataset, data)
        positive_u = torch.arange(len(transformed_data))
        positive_i = transformed_data[self.iid_field]
        return transformed_data, None, positive_u, positive_i
