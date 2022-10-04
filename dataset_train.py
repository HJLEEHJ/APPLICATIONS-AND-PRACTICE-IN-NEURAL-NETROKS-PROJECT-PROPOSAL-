import os

from utils import load_wav
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
            self,
            config
    ):
        noisy_files_list = []
        self.noisy_dir = config['dataset_root']['noisy_data'] 
        self.clean_dir = config['dataset_root']['clean_data'] 
        print("self.noisy_dir",self.noisy_dir)
        
        for (root, _ , files) in os.walk(self.noisy_dir):
            for file in files:
                noisy_files_list.append(os.path.join(root, file))

        self.sr = int(config['acoustics']['sr'])

        self.length = len(noisy_files_list)
        self.noisy_files_list = noisy_files_list

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        print("self.noisy_dir",self.noisy_dir)

        noisy_file_path = self.noisy_files_list[item]
        _ , file_name = os.path.split(noisy_file_path)
        clean_file_path = os.path.join(self.clean_dir,file_name) 

        noisy = load_wav(noisy_file_path, sr=self.sr)
        clean = load_wav(clean_file_path, sr=self.sr)

        return noisy, clean 
 