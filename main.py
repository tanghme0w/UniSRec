import argparse
import os.path
from logging import getLogger
import torch
from recbole.config import Config
from recbole.utils import init_seed, init_logger, set_color
from trainer import Trainer

from model import MISSRec
from dataset import LazyLoadDataset
from dataloader import CustomizedTrainDataloader, CustomizedFullSortEvalDataloader
from preprocess import build_mmap
import numpy as np


def main(**kwargs):
    # configurations initialization
    props = ['props/model.yaml', 'props/config.yaml']
    config = Config(model=MISSRec, dataset='', config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # get language model embedding
    config['mmap_out'] = args.mmap_out or config['mmap_out']  # commandline argument overrides config file
    if config['preprocess']:
        assert config['language_model'] is not None
        assert config['metadata_path'] is not None
        assert config['mmap_out'] is not None
        logger.info(set_color('Preprocessing raw text.', 'red'))
        (config['mmap_idx_path'],
         config['mmap_idx_shape'],
         config['mmap_emb_path'],
         config['mmap_emb_shape']) = build_mmap(config)

    # check mmap existence
    assert config['mmap_idx_path'] is not None
    assert config['mmap_idx_shape'] is not None
    assert config['mmap_emb_path'] is not None
    assert config['mmap_emb_shape'] is not None

    # create model and initialize
    model = MISSRec(config).to(config['device'])
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        logger.info(f'Loading from {checkpoint}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(model)

    # create trainer
    trainer = Trainer(config, model)

    # create dataset and dataloader
    config['data_path'] = args.data or config['data_path']  # commandline argument overrides config file
    train_loader = CustomizedTrainDataloader(
        dataset=LazyLoadDataset(config, 'train' if args.mode == 'train' else ''),
        config=config
    ) if args.mode in ['train', 'infer-user'] else None

    valid_loader = CustomizedFullSortEvalDataloader(
        dataset=LazyLoadDataset(config, 'valid'),
        config=config,
        shuffle=False
    ) if args.mode == 'train' else None

    test_loader = CustomizedFullSortEvalDataloader(
        dataset=LazyLoadDataset(config, 'test' if args.mode == 'train' else ''),
        config=config,
        shuffle=False
    ) if args.mode in ['train', 'eval'] else None

    # model training
    if args.mode == 'train':
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=True, show_progress=config['show_progress']
        )

        # after training, show the best valid result and load best model for evaluation
        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        model.load_state_dict(torch.load(trainer.saved_model_file), strict=False)

    elif args.checkpoint == '':
        logger.warning(set_color('The model is neither trained nor loaded from checkpoint.', 'red'))

    if args.mode == 'infer-item':
        model.eval()
        idx_mmap = np.memmap(config['mmap_idx_path'], dtype=np.int32, shape=tuple(config['mmap_idx_shape']))
        emb_mmap = np.memmap(config['mmap_emb_path'], dtype=np.float32, shape=tuple(config['mmap_emb_shape']))
        with torch.no_grad():
            model_input = torch.tensor(emb_mmap[idx_mmap], dtype=torch.float32).to(config['device'])
            model_output = model.moe_adaptor(model_input)
            model_output = torch.nn.functional.normalize(model_output).cpu().numpy()
        item_id = np.arange(model_output.shape[0]).reshape((-1, 1))
        item_emb = np.concatenate([item_id, model_output], axis=1)
        np.save(os.path.join(config['mmap_out'], "all_item_embedding.npy"), item_emb)
        return

    if args.mode == 'infer-user':
        model.eval()
        user_emb = []
        for user in train_loader:
            with torch.no_grad():
                item_seq = user['item_id_list'].to(config['device'])
                item_seq_len = user['item_length'].to(config['device'])
                item_emb_list = model.moe_adaptor(user['item_emb_list'].to(config['device']))
                model_output = model.forward(item_seq, item_emb_list, item_seq_len)
                model_output = torch.nn.functional.normalize(model_output).cpu().numpy()
            user_emb.append(model_output)
        user_emb = np.concatenate(user_emb)
        np.save(os.path.join(config['mmap_out'], "all_user_embedding.npy"), user_emb)
        return

    result = trainer.evaluate(test_loader, model=model, show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='', help='pre-trained model path')
    parser.add_argument('--data', type=str, default='', help='root data path')
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['train', 'eval', 'infer-user', 'infer-item'],
                        help='run mode (train, eval, infer-user, infer-item)')
    parser.add_argument('--mmap-out', type=str, default='',
                        help='memory map & item embedding output path')
    args, unparsed = parser.parse_known_args()
    main()
