import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.utils import init_seed, init_logger, set_color
from trainer_mini import Trainer

from model_mini import CustomizedUniSRec
from dataset_mini import PretrainDataset
from data_utils import build_dataloader


def finetune(pretrained_file, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/UniSRec_mini.yaml', 'props/finetune_mini.yaml']
    print(props)

    # configurations initialization
    config = Config(model=CustomizedUniSRec, dataset='', config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # custom configurations
    config['mmap_idx_path'] = 'dataset/customized/mmap/mmap_idx_100'
    config['mmap_idx_shape'] = (100,)
    config['mmap_emb_shape'] = (100, 768)
    config['mmap_emb_path'] = 'dataset/customized/mmap/mmap_data_100_768'
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = PretrainDataset(config)

    # dataset splitting
    train_data, valid_data, test_data = build_dataloader(config, [dataset, dataset, dataset])

    # model loading and initialization
    model = CustomizedUniSRec(config, train_data.original_dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
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
    parser.add_argument('-f', type=bool, default=True)
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(pretrained_file=args.p, fix_enc=args.f)
