from recbole.utils import set_color
from logging import getLogger
from dataset_mini import PretrainDataset
from recbole.sampler import RepeatableSampler
from dataloader import CustomizedTrainDataloader, CustomizedFullSortEvalDataloader


# based on recbole.data.utils.data_preparation
def build_dataloader(config, datasets: list[PretrainDataset]):

    train_dataset, valid_dataset, test_dataset = datasets

    train_sampler = None
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
