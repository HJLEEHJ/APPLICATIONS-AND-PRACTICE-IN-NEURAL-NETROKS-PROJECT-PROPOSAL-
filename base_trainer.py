from functools import partial
from pathlib import Path
from tqdm import tqdm

from pesq import pesq
from pystoi.stoi import stoi

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rich import print
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

from utils import ExecutionTime, prepare_empty_dir, istft, stft, power_compression, inverse_power_compression

import gc 

from rich import print


gc.collect() 

torch.cuda.empty_cache()

plt.switch_backend("agg")
console = Console()

def transform_pesq_range(pesq_score):
    """Transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]"""
    return (pesq_score + 0.5) / 5

class Trainer(object):
    def __init__(
        self, config, data, device, model, optimizer
    ):
        self.train_dataloader = data["tr_loader"]
        self.valid_dataloader = data["vl_loader"]
        self.device = device

        self.model = model
        self.optimizer = optimizer
        self.loss_function = nn.MSELoss() 

        self.resume = eval(config["train"]["resume"]) 
        # self.resume = False

        # Acoustics
        self.acoustic_config = config["acoustics"]
        n_fft = int(self.acoustic_config["n_fft"])
        hop_length = int(self.acoustic_config["hop_length"])
        win_length = int(self.acoustic_config["win_length"])

        # STFT
        self.torch_stft = partial(
            stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.torch_istft = partial(
            istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.librosa_stft = partial(
            librosa.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.librosa_istft = partial(
            librosa.istft, hop_length=hop_length, win_length=win_length
        )

        self.train_config = config["train"]
        self.epochs = int(self.train_config["epoch"])
        self.save_checkpoint_interval = float(self.train_config["save_checkpoint_interval"])
        self.clip_grad_norm_value = float(self.train_config["clip_grad_norm_value"])
        assert (
            self.save_checkpoint_interval >= 1
        ), "Check the 'save_checkpoint_interval' parameter in the config. It should be large than one."

        # Trainer.validation in the config
        self.validation_config = config["validation"]
        self.validation_interval = float(self.validation_config["validation_interval"])
        self.save_max_metric_score = eval(self.validation_config["save_max_metric_score"])
        assert (
            self.validation_interval >= 1
        ), "Check the 'validation_interval' parameter in the config. It should be large than one."

        # Trainer.visualization in the config
        self.visualization_config = config["visualization"]

        # In the 'train.py' file,  ifthe 'resume' item is 'True', we will update the following args:
        self.start_epoch = 1 
        self.best_score = -np.inf if self.save_max_metric_score else np.inf
        self.save_dir = (
            Path(config["path"]["save_dir"])
            / config["path"]["experiment_name"]
        )
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"
        self.source_code_dir = Path(__file__).absolute().parent.parent.parent ###

        if self.resume:
            self._resume_checkpoint()
        else:
            pass

        # Debug   
        self.validation_only = eval(config["validation"]["validation_only"])

        if eval(config["path"]["pretrained_model"]):
            self._preload_model(Path(config['path']['pretrained_model_path']))

        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=self.resume)

        self.writer = SummaryWriter(
            self.logs_dir.as_posix(), max_queue=5, flush_secs=30
        )

        self._print_networks([self.model])

    def _preload_model(self, model_path):
        model_path = model_path
        assert (
            model_path.exists()
        ), f"The file {model_path.as_posix()} is not exist. please check path."

        model_checkpoint = torch.load(model_path.as_posix(), map_location="cpu")
        self.model.load_state_dict(model_checkpoint, strict=False)
        self.model.to(self.device)

        print(f"Model preloaded successfully from {model_path.as_posix()}.")

    def _resume_checkpoint(self):
        latest_model_path = (
            self.checkpoints_dir / "latest_model.tar"
        )
        assert (
            latest_model_path.exists()
        ), f"{latest_model_path} does not exist, can not load latest checkpoint."
 
        checkpoint = torch.load(latest_model_path.as_posix(), map_location="cpu")

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_score = float(checkpoint["best_score"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        self.model.to(self.device)

        print(
            f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch."
        )

    def _save_checkpoint(self, epoch, is_best_epoch=False):

        print(f"\t Saving the model checkpoint of epoch {epoch}...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict(),
        }

        state_dict["model"] = self.model.state_dict()
 
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
 
        torch.save(
            state_dict["model"],
            (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix(),
        )
 
        if is_best_epoch:
            print(f"\t :smiley: Found a best score in the epoch {epoch}, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

    def _is_best_epoch(self, score, save_max_metric_score=True):
        if save_max_metric_score and score >= self.best_score:
            self.best_score = score
            return True
        elif not save_max_metric_score and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _print_networks(models: list):
        print(
            f"This project contains {len(models)} models, the number of the parameters is: "
        )

        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                params_of_network += param.numel()

            print(f"\Model {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(
            f"The amount of parameters in the project is {params_of_all_networks / 1e6} million."
        )

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    def spec_audio_visualization(self, noisy, enhanced, clean, name, epoch):
        self.writer.add_audio(
            f"Speech/{name}_Noisy", noisy, epoch, sample_rate=16000
        )
        self.writer.add_audio(
            f"Speech/{name}_enhanced", enhanced, epoch, sample_rate=16000
        )
        self.writer.add_audio(
            f"Speech/{name}_Clean", clean, epoch, sample_rate=16000
        )
        noisy_stft = self.librosa_stft(y=noisy) 
        enhanced_stft = self.librosa_stft(y=enhanced)
        clean_stft = self.librosa_stft(y=clean)

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate([noisy_stft, enhanced_stft, clean_stft]):
            librosa.display.specshow(
                librosa.power_to_db(np.abs(mag)**2, ref = np.max(mag)),
                y_axis="fft",
                x_axis='time',
                ax=axes[k]
            )
        plt.tight_layout()
        self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

    def metrics_visualization( 
        self,
        noisy_list,
        clean_list,
        enhanced_list,
        metrics_list,
        epoch,
        name,
        ):

        stoi_mean = 0.0
        wb_pesq_mean = 0.0

        score_on_noisy_stoi = np.empty((0,len(clean_list)))
        score_on_enhanced_stoi = np.empty((0,len(clean_list)))

        score_on_noisy_pesq = np.empty((0,len(clean_list)))
        score_on_enhanced_pesq = np.empty((0,len(clean_list)))

        for metric_name in metrics_list:
            if metric_name == "STOI":
                for ref, est in zip(clean_list, noisy_list):
                    try:
                        score = stoi(ref, est, 16000, extended=False)
                        score_on_noisy_stoi = np.append(score_on_noisy_stoi,score)
                    except:
                        print(f"pass noisy {name} STOI calculation ")
                        pass

                for ref, est in zip(clean_list, enhanced_list):
                    try:
                        score = stoi(ref, est, 16000, extended=False)
                        score_on_enhanced_stoi = np.append(score_on_enhanced_stoi,score)
                    except:
                        print(f"pass enhanced {name} STOI calculation ")
                        pass

                
            mean_score_on_noisy_stoi = np.mean(score_on_noisy_stoi)
            mean_score_on_enhanced_stoi = np.mean(score_on_enhanced_stoi)

            self.writer.add_scalars(
                f"Validation/{metric_name}",
                {"Noisy": mean_score_on_noisy_stoi, "Enhanced": mean_score_on_enhanced_stoi},
                epoch,
            )

            stoi_mean = mean_score_on_enhanced_stoi

            if metric_name == "WB_PESQ":
                for ref, est in zip(clean_list, noisy_list):
                    try:
                        score = pesq(16000, ref, est, 'wb')
                        score_on_noisy_pesq = np.append(score_on_noisy_pesq,score)
                    except:
                        print(f"pass noisy {name} WB_PESQ calculation ")
                        pass
                for ref, est in zip(clean_list, enhanced_list):
                    try:
                        score = pesq(16000, ref, est, 'wb')
                        score_on_enhanced_pesq = np.append(score_on_enhanced_pesq,score)
                    except:
                        print(f"pass enhanced {name} WB_PESQ calculation ")
                        pass
    
            # Add mean value of the metric to tensorboard
            mean_score_on_noisy_pesq = np.mean(score_on_noisy_pesq)
            mean_score_on_enhanced_pesq = np.mean(score_on_enhanced_pesq)

            self.writer.add_scalars(
                f"Validation/{metric_name}",
                {"Noisy": mean_score_on_noisy_pesq, "Enhanced": mean_score_on_enhanced_pesq},
                epoch,
            )

            wb_pesq_mean = transform_pesq_range(mean_score_on_enhanced_pesq)

        return (stoi_mean + wb_pesq_mean) / 2

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            
            print(f"{'=' * 15} epoch {epoch} {'=' * 15}")
            print("[0 seconds] Begin training...")

            # [debug validation] Only run validation 
            # inference + calculating metrics + saving checkpoints
            if self.validation_only:
                self._set_models_to_eval_mode()
                metric_score = self._validation_epoch(epoch)

                if self._is_best_epoch(
                    metric_score, save_max_metric_score=self.save_max_metric_score
                ):
                    self._save_checkpoint(epoch, is_best_epoch=True)

                # Skip the following regular training, saving checkpoints, and validation
                continue

            # Regular training
            timer = ExecutionTime()
            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            # Regular save checkpoints
            if (
                self.save_checkpoint_interval != 0
                and (epoch % self.save_checkpoint_interval == 0)
            ):
                self._save_checkpoint(epoch)

            # Regular validation
            if epoch % self.validation_interval == 0:
                print(
                    f"[{timer.duration()} seconds] Training has finished, validation is in progress..."
                )
                self._set_models_to_eval_mode()
                metric_score = self._validation_epoch(epoch)

                if self._is_best_epoch(
                    metric_score, save_max_metric_score=self.save_max_metric_score
                ):
                    self._save_checkpoint(epoch, is_best_epoch=True)

            print(f"[{timer.duration()} seconds] This epoch is finished.")
            


    def _train_epoch(self, epoch):
        loss_total = 0.0
        mag_loss_total = 0.0
        comlex_loss_total = 0.0
        print("processing....")

        with tqdm(position=0, leave=True) as pbar:
            for i, (noisy, clean) in enumerate(tqdm(self.train_dataloader, bar_format='{l_bar}{bar:50}{r_bar}')):

                self.optimizer.zero_grad()
 
                noisy_mag , noisy_angle , _ , _ = self.torch_stft(noisy) # [B, F, T]
                noisy_mag = power_compression(noisy_mag,trans=False)
                noisy_c_stft = torch.polar(noisy_mag,noisy_angle)
                noisy_c_stft = torch.stack((noisy_c_stft.real, noisy_c_stft.imag),dim=1).transpose(2,3) 
                noisy_mag = noisy_mag.unsqueeze(1).transpose(2,3)

                clean_mag , clean_angle , _ , _ = self.torch_stft(clean) # [B, F, T]
                clean_mag = power_compression(clean_mag,trans=False)
                clean_c_stft = torch.polar(clean_mag,clean_angle)
                clean_c_stft = torch.stack((clean_c_stft.real, clean_c_stft.imag),dim=1) 

                noisy_mag = noisy_mag.to(self.device)
                clean_c_stft = clean_c_stft.to(self.device)
                noisy_angle = noisy_angle.to(self.device)
                clean_mag = clean_mag.to(self.device)

                enhanced_mag = self.model(noisy_mag)   # [B, 1, T, F=161]
                enhanced_stft = torch.stack((enhanced_mag.squeeze(1).transpose(2,3), noisy_angle),dim=1) 
                

                loss1 = self.loss_function(clean_mag, enhanced_mag)
                loss2 = self.loss_function(clean_c_stft, enhanced_stft)   
                loss = 0.5 * loss1 + 0.5 * loss2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
                self.optimizer.step()

                loss_total += loss.item()
                mag_loss_total += loss1.item()
                comlex_loss_total += loss2.item()
 
                pbar.set_description("loss for training %s" % loss.item() )    

            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)
            self.writer.add_scalar(f"Loss/mag", mag_loss_total / len(self.train_dataloader), epoch)
            self.writer.add_scalar(f"Loss/complex", comlex_loss_total / len(self.train_dataloader), epoch)
            print(f"Total loss: {loss_total}")
            pbar.close()


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_n_samples = int(self.visualization_config["n_samples"])
        visualization_metrics = ["WB_PESQ", "STOI"]

        loss_total = 0.0
        # mag_loss_total = 0.0
        # comlex_loss_total = 0.0
        loss_list = 0.0
        item_idx_list = 0 
        noisy_y_list = []
        clean_y_list = []
        enhanced_y_list = []
        validation_score_list = 0.0

        with torch.no_grad():
            for i, (noisy, clean, name) in tqdm(enumerate(self.valid_dataloader)):
                assert len(name) == 1, "The batch size for the validation stage must be one."
                name = name[0]                

                noisy_mag , noisy_angle , _ , _ = self.torch_stft(noisy) # [B, F, T]
                noisy_mag = power_compression(noisy_mag,trans=False)
                noisy_c_stft = torch.polar(noisy_mag,noisy_angle)
                noisy_c_stft = torch.stack((noisy_c_stft.real, noisy_c_stft.imag),dim=1).transpose(2,3)  # [B, 2, T, F]  
                noisy_mag = noisy_mag.unsqueeze(1).transpose(2,3)

                clean_mag , clean_angle , _ , _ = self.torch_stft(clean) # [B, F, T]
                clean_mag = power_compression(clean_mag,trans=False)
                clean_c_stft = torch.polar(clean_mag,clean_angle)
                clean_c_stft = torch.stack((clean_c_stft.real, clean_c_stft.imag),dim=1) 

                noisy_mag = noisy_mag.to(self.device)
                clean_c_stft = clean_c_stft.to(self.device)
                noisy_angle = noisy_angle.to(self.device)
                clean_mag = clean_mag.to(self.device)
        

                enhanced_mag = self.model(noisy_mag).squeeze(1).transpose(1,2)   # [B, F, T]
                enhanced_stft = torch.stack((enhanced_mag, noisy_angle),dim=1) 


                loss1 = self.loss_function(clean_mag, enhanced_mag)
                loss2 = self.loss_function(clean_c_stft, enhanced_stft)   
                loss = 0.5 * loss1 + 0.5 * loss2
       
                enhanced_mag = inverse_power_compression(enhanced_mag) # [B, F, T]
                decompressed_enhanced_stft = torch.polar(enhanced_mag, noisy_angle) # [B, F, T]

                decompressed_enhanced_stft = decompressed_enhanced_stft.detach().squeeze(0).cpu().numpy()
                noisy = noisy.detach().squeeze(0).cpu().numpy()
                clean = clean.detach().squeeze(0).cpu().numpy()   
            
                enhanced_audio = self.librosa_istft(decompressed_enhanced_stft)
 
                assert len(enhanced_audio) == len(noisy) == len(clean)
                loss_total += loss
 
                # Separated loss
                loss_list += loss
                item_idx_list += 1

                
                if item_idx_list <= visualization_n_samples:
                    self.spec_audio_visualization(noisy, enhanced_audio, clean, name, epoch)    
                
                noisy_y_list.append(noisy)
                clean_y_list.append(clean)
                enhanced_y_list.append(enhanced_audio)

            self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

            validation_score_list = self.metrics_visualization(
                noisy_y_list, clean_y_list, enhanced_y_list,
                visualization_metrics, epoch, name            
                )
            
            
            return validation_score_list

