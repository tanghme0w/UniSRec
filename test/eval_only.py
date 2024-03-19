import argparse
from logging import getLogger
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from trainer_mini import Trainer

from model_mini import CustomizedUniSRec
from dataset_mini import PretrainDataset
from data_utils import build_dataloader


def inference(pretrained_file, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/UniSRec_mini.yaml', 'props/finetune_mini.yaml']
    print(props)

    # configurations initialization
    config = Config(model=CustomizedUniSRec, dataset='', config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # custom configurations
    config['mmap_idx_path'] = 'dataset/customized/Scientific/mmap/mmap_idx_4386'
    config['mmap_idx_shape'] = (4386,)
    config['mmap_emb_shape'] = (4386, 768)
    config['mmap_emb_path'] = 'dataset/customized/Scientific/mmap/mmap_emb_4386_768'
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

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model evaluation
    test_result = trainer.evaluate(test_data, model_file='saved/UniSRec-FHCKM-300.pth', load_best_model=True, show_progress=config['show_progress'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=bool, default=True)
    args, unparsed = parser.parse_known_args()
    print(args)

    inference(pretrained_file=args.p, fix_enc=args.f)
