import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.utils import init_seed, init_logger, set_color
from trainer import Trainer

from model import MISSRec
from dataset import LazyLoadDataset
from dataloader import build_dataloader
from preprocess import build_mmap


def pretrain(pretrained_file, **kwargs):
    # configurations initialization
    props = ['props/model.yaml', 'props/pretrain.yaml']
    config = Config(model=MISSRec, dataset='', config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # get language model embedding
    if config['train_from_rawtext']:
        assert config['language_model'] is not None
        assert config['metadata_path'] is not None
        assert config['mmap_out'] is not None
        (config['mmap_idx_path'],
         config['mmap_idx_shape'],
         config['mmap_emb_path'],
         config['mmap_emb_shape']) = build_mmap(config)

    # check mmap existence
    assert config['mmap_idx_path'] is not None
    assert config['mmap_idx_shape'] is not None
    assert config['mmap_emb_path'] is not None
    assert config['mmap_emb_shape'] is not None

    # create dataset
    train_dataset = LazyLoadDataset(config, 'valid')
    valid_dataset = LazyLoadDataset(config, 'valid')
    test_dataset = LazyLoadDataset(config, 'test')

    # create dataloader
    train_data, valid_data, test_data = build_dataloader(config, [train_dataset, valid_dataset, test_dataset])

    # model loading and initialization
    model = MISSRec(config, train_data.original_dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    args, unparsed = parser.parse_known_args()
    print(args)

    pretrain(pretrained_file=args.p)
