

import os
from os.path import join


import numpy as np

import torch
from torch import device, nn
from torch._C import device
from torchinfo import summary
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import autocast, nn


from time import time, sleep

from typing import Union, Tuple, List

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.SwinUMambaD import get_swin_umamba_d_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs




from nnunetv2.training.loss.l2_loss import L2

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper



class nnUNetTrainerDepth_SwinUMambaD(nnUNetTrainer):

    """ Swin-UMamba$\dagger$ with Mamba-based decoder"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = 10
        self.num_epochs = 250

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True) -> nn.Module:

        model = get_swin_umamba_d_from_plans(
            plans_manager, 
            dataset_json, 
            configuration_manager,
            num_input_channels, 
            deep_supervision=enable_deep_supervision, 
            use_pretrain=True
        )

        print(model)
        print(configuration_manager.patch_size)
        #summary(model, input_size=[1, num_input_channels] + configuration_manager.patch_size)

        return model
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr, 
            weight_decay=self.weight_decay, 
            eps=1e-5,
            betas=(0.9, 0.999),
            )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    def _build_loss(self):

        loss = L2()
            
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss




    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        
        loss_here = np.mean(outputs_collated['loss'])

        # self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        # self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        print("LOSS HERE", loss_here)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        # Log the end time of the epoch
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Logging train and validation loss
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        # Log the duration of the epoch
        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        # Checkpoint handling for best and periodic saves
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        best_metric = 'val_losses'  # Example metric, adjust based on actual usage
        if self._best_ema is None or self.logger.my_fantastic_logging[best_metric][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging[best_metric][-1]
            self.print_to_log_file(f"Yayy! New best EMA MSE: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        # Increment the epoch counter
        self.current_epoch += 1

        
    def on_train_epoch_start(self):
        # freeze the encoder if the epoch is less than 10
        if self.current_epoch < self.freeze_encoder_epochs:
            self.print_to_log_file("Freezing the encoder")
            if self.is_ddp:
                self.network.module.freeze_encoder()
            else:
                self.network.freeze_encoder()
        else:
            self.print_to_log_file("Unfreezing the encoder")
            if self.is_ddp:
                self.network.module.unfreeze_encoder()
            else:
                self.network.unfreeze_encoder()
        super().on_train_epoch_start()


    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0,1.0], [0.25,0.25], [0.125,0.125], [0.0625, 0.0625]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales
    

    # https://github.com/Phyrise/nnUNet_translation/ For img2img Unet

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target'][0]
        print("TARGET SHAPE", target.shape)

        print(target)
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        #arget = torch.stack(target, dim=0).to(self.device, non_blocking=True)
        #print(target.shape, data.shape)


        '''if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target = torch.stack(target, 0)

        else:
            target = target.to(self.device, non_blocking=True)'''

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)[0]
            '''torch.save(data, "data")
            torch.save(output, "output")
            torch.save(target, "target")'''

            del data
            mse_loss = L2()
            l = mse_loss(output, target)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}