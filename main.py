import sys
sys.path.append('swin_umamba/')
import torch

import os

import glob

os.environ['nnUNet_results'] = 'results/'  
os.environ['nnUNet_raw'] = 'raw/'
os.environ['nnUNet_preprocessed'] = 'preprocessed/'



from swin_umamba.nnunetv2.run.run_training import run_training
from swin_umamba.nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
from swin_umamba.nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from swin_umamba.nnunetv2.run.run_training import run_training, get_trainer_from_args
from swin_umamba.nnunetv2.training.dataloading.utils import unpack_dataset

from typing import Union, Optional


import shutil, json, glob, os
from tqdm import tqdm 

import numpy as np


def run_unpacking(dataset_name_or_id: Union[str, int],
                  configuration: str, fold: Union[int, str],
                  trainer_class_name: str = 'nnUNetTrainer',
                  plans_identifier: str = 'nnUNetPlans',
                  pretrained_weights: Optional[str] = None,
                  num_gpus: int = 1,
                  use_compressed_data: bool = False,
                  export_validation_probabilities: bool = False,
                  continue_training: bool = False,
                  only_run_validation: bool = False,
                  disable_checkpointing: bool = False,
                  val_with_best: bool = False,
                  device: torch.device = torch.device('cuda')):

    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, use_compressed_data, device=device)
    # apply the unpacking to the proposed trainer;
    unpack_dataset(nnunet_trainer.preprocessed_dataset_folder)


if __name__ == '__main__':


    '''data_dir = 'data/imgs/'
    target_dir = 'data/gt/'



    print(os.environ)
    # example with 1 input modality
    list_datas = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    list_targets = sorted(glob.glob(os.path.join(target_dir, '*.npy')))

    print(len(list_datas), list_datas)
    print(len(list_targets), list_targets)


    dataset_id =  50 # /!\ we will use both the dataset_id and the dataset_id + 1 
    dataset_data_name = 'FSOut'
    dataset_target_name = 'DepthGT'

    # we will copy the datas
    # do not use exist_ok=True, we want an error if the dataset exist already
    dataset_data_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id:03d}_{dataset_data_name}') 
    os.makedirs(dataset_data_path, exist_ok = True)
    os.makedirs(os.path.join(dataset_data_path, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(dataset_data_path, 'labelsTr'), exist_ok = True)

    dataset_target_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id+1:03d}_{dataset_target_name}') 
    os.makedirs(dataset_target_path, exist_ok = True)
    os.makedirs(os.path.join(dataset_target_path, 'imagesTr'), exist_ok = True)
    os.makedirs(os.path.join(dataset_target_path, 'labelsTr'), exist_ok = True)


    def process_file(data_path, dataset_path):
        curr_npy = np.load(data_path)
        filename = os.path.basename(data_path)
        if not filename.endswith('_0000.npy'):
            filename = filename.replace('.npy', '_0000.npy')

        np.save(os.path.join(dataset_path, f'imagesTr/{filename}'), curr_npy)
        #curr_nifti.to_filename(os.path.join(dataset_path, f'imagesTr/{filename}'))

        data = curr_npy
        # Adjust the mask as needed for your specific use case. By default, the mask is set to 1 for the entire volume.
        # This will be used for foreground preprocessing, cf https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation_normalization.md
        data = np.ones_like(data)

        filename = filename.replace('_0000', '') #remove _0000 for masks

        np.save(os.path.join(dataset_path, f'labelsTr/{filename}'), data)
        #nib.Nifti1Image(data, mat).to_filename()


    #### without multithreading
    for data_path in tqdm(list_datas, total=len(list_datas)):
        process_file(data_path, dataset_data_path)

    for target_path in tqdm(list_targets, total=len(list_targets)):
        process_file(target_path, dataset_target_path)


    # /!\ you will need to edit this with regards to the number of modalities used;
    data_dataset_json = {
        "labels": {
            "label_001": "1", 
            "background": 0
        },
        "channel_names": {
            "0": "noNorm",
        },
        "numTraining": len(list_datas),
        "file_ending": ".npy"
    }
    dump_data_datasets_path = os.path.join(dataset_data_path, 'dataset.json')
    with open(dump_data_datasets_path, 'w') as f:
        json.dump(data_dataset_json, f)

    target_dataset_json = {
        "labels": {
            "label_001": "1",
            "background": 0
        },
        "channel_names": {
            "0": "noNorm",
        },
        "numTraining": len(list_targets),
        "file_ending": ".npy"
    }
    dump_target_datasets_path = os.path.join(dataset_target_path, 'dataset.json')
    with open(dump_target_datasets_path, 'w') as f:
        json.dump(target_dataset_json, f)


    if 'MPLBACKEND' in os.environ: 
        del os.environ['MPLBACKEND'] # avoid conflicts with matplotlib backend  
        
    print("OSSSSS 0", os.environ['nnUNet_raw'])    
    rrr = sorted(glob.glob(os.environ['nnUNet_raw']))
    print("RAW !!", rrr)

    
    extract_fingerprints([dataset_id])
    plan_experiments([dataset_id])
    preprocess([dataset_id])
    run_unpacking(dataset_id, configuration='2d', fold=0)

    print("DONE PART 1")
    extract_fingerprints([dataset_id+1])
    plan_experiments([dataset_id+1])
    preprocess([dataset_id+1])
    run_unpacking(dataset_id+1, configuration='2d', fold=0)

    # Define 2nd modality raw data as gt_segmentations of 1st modality
    nnunet_datas_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id+1:03d}_{dataset_target_name}') 
    nnunet_targets_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}') 

    list_targets = glob.glob(os.path.join(f"{dataset_target_path}/imagesTr", '*'))
    list_targets.sort()
    list_gt_segmentations_datas = glob.glob(os.path.join(f"{nnunet_targets_preprocessed_dir}/gt_segmentations", '*'))
    list_gt_segmentations_datas.sort()

    print(nnunet_targets_preprocessed_dir)

    for (preprocessed_path, gt_path) in zip(list_targets, list_gt_segmentations_datas):
        # here, gt_path is the path to the gt_segmentation in nnUNet_preprocessed.
        print(preprocessed_path, "->", gt_path) # ensure correct file pairing; 
        shutil.copy(src = preprocessed_path, dst = gt_path) # we use shutil.copy to ensure safety, but switching to shutil.move would be more efficient

    #Define 2nd modality preprocessed files as ground truth of 1st modality
    list_preprocessed_datas_seg_path = sorted(glob.glob(os.path.join(nnunet_targets_preprocessed_dir, 'nnUNetPlans_2d/*_seg.npy')))

    list_preprocessed_targets_path = sorted(glob.glob(os.path.join(nnunet_datas_preprocessed_dir, 'nnUNetPlans_2d/*.npy')))
    list_preprocessed_targets_path = [name for name in list_preprocessed_targets_path if '_seg' not in name]

    for (datas_path, targets_path) in zip(list_preprocessed_datas_seg_path, list_preprocessed_targets_path):
        print(targets_path, "->", datas_path)
        shutil.copy(src = targets_path, dst = datas_path) '''


    # multithreading in torch doesn't help nnU-Net if run on GPU
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'


    print("GPU !!!!!!!")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda')
    

    run_training(50, "2d", 0, "nnUNetTrainerDepth_SwinUMambaD", device=device)



#run_training("Dataset001_Depth",configuration= "2D", fold=0)
