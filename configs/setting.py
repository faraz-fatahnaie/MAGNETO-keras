from pathlib import Path
import json


def setting(config_file: json = None, project: str = 'DAE'):
    config = dict()
    BASE_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR.joinpath(f'session_{project}').mkdir(exist_ok=True)
    BASE_DIR.joinpath('trained_ae').mkdir(exist_ok=True)

    if config_file is None:
        config_name = 'CONFIG'  # config file name in config dir
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)

    config['SEED'] = config_file['seed']

    config['DATASET_NAME'] = config_file['dataset']['name']
    config['CLASSIFICATION_MODE'] = config_file['dataset']['classification_mode']
    config['DATASET_PATH'] = BASE_DIR.joinpath('dataset', config['DATASET_NAME'])
    config['NUM_WORKER'] = config_file['dataset']['n_worker']
    config['DEEPINSIGHT'] = config_file['dataset']['deepinsight']
    dataset_name = config['DATASET_NAME']
    config['DATASET_PATH'] = BASE_DIR.joinpath('dataset', dataset_name)

    config['AE_USE'] = config_file['autoencoder']['use_autoencoder']
    config['AE_TRAINABLE'] = config_file['autoencoder']['trainable']
    config['AE_FINETUNE'] = config_file['autoencoder']['fine_tuning']

    return config, config_file
