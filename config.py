import configparser

config = configparser.ConfigParser()

config['dataset_root'] = {}
config['dataset_root']['clean_data'] = './dataset/trainset_clean/'
config['dataset_root']['noisy_data'] = './dataset/trainset_noisy'
 
config['path'] = {}
config['path']['experiment_name'] = '신경망'
config['path']['best_path'] = './LOG/BEST/'
config['path']['save_dir'] = './LOG'
config['path']['pretrained_model'] = 'False'
config['path']['pretrained_model_path'] = 'False'


config['train'] = {}
config['train']['config_root'] = '5'
config['train']['model'] = 'StoDec'
config['train']['epoch'] = '80'
config['train']['batch_size'] = '6'
config['train']['clip_grad_norm_value'] = '10'
config['train']['save_checkpoint_interval'] = '1'
config["train"]["resume"] = 'False'
## config["train"]["resume"] = 'True'

config['validation'] = {}
config['validation']['val_clean_data'] = './dataset/valset_clean/'
config['validation']['val_noisy_data'] = './dataset/valset_noisy'
config['validation']['save_max_metric_score'] = 'True'
config['validation']['validation_interval'] = '1'
### config["validation"]["validation_only"] = 'True'
config["validation"]["validation_only"] = 'False'

config["visualization"] = {}
config["visualization"]["n_samples"] = '10'
config["visualization"]["num_workers"] = '10'

config['optimizer'] = {}
config['optimizer']['lr'] = '0.0001'
config['optimizer']['weight_decay'] = '0.05' 

config['acoustics'] = {}
config['acoustics']['n_fft'] = '320' 
config['acoustics']['win_length'] = '320' 
config['acoustics']['sr'] = '16000'
config['acoustics']['hop_length'] = '160'

   
with open('/data/hyunjoo/asd/project/config.ini', 'w', encoding='utf-8') as configfile:
    config.write(configfile)

