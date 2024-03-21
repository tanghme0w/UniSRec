import torch
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader

from embedding import PLMEmb
import numpy as np
from recbole.sampler import RepeatableSampler
from logging import getLogger
from recbole.utils import set_color


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
        self.step = batch_size

    def collate_fn(self, index):
        index = np.array(index)
        data = self.original_dataset[index]
        transformed_data = self.plm_embedding(data)
        return transformed_data


class CustomizedFullSortEvalDataloader(FullSortEvalDataLoader):
    def __init__(self, dataset, config, shuffle):
        self.sampler = RepeatableSampler(
            phases=['train', 'valid', 'test'],
            dataset=dataset,
            distribution='uniform',
            alpha=0.1
        )
        super().__init__(dataset=dataset, config=config, sampler=self.sampler, shuffle=shuffle)
        self.original_dataset = dataset
        self.plm_embedding = PLMEmb(config)


    def collate_fn(self, index):
        index = np.array(index)
        data = self.original_dataset[index]
        transformed_data = self.plm_embedding(data)
        positive_u = torch.arange(len(transformed_data))
        positive_i = transformed_data[self.iid_field]
        return transformed_data, None, positive_u, positive_i


def build_dataloader(config, datasets):
    train_dataset, valid_dataset, test_dataset = datasets

    valid_sampler = RepeatableSampler(
        phases=['train', 'valid', 'test'],
        dataset=valid_dataset,
        distribution='uniform',
        alpha=0.1
    ).set_phase('valid')
    test_sampler = valid_sampler.set_phase('test')  # set_phase returns a deep copy of the sampler with appointed phase

    train_dataloader = CustomizedTrainDataloader(dataset=train_dataset, config=config)
    valid_dataloader = CustomizedFullSortEvalDataloader(
        config=config, dataset=valid_dataset, sampler=valid_sampler, shuffle=False
    )
    test_dataloader = CustomizedFullSortEvalDataloader(
        config=config, dataset=test_dataset, sampler=test_sampler, shuffle=False
    )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_dataloader, valid_dataloader, test_dataloader
