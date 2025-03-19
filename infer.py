import sys
sys.path.append('swin_umamba/')

from swin_umamba.nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from swin_umamba.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


import os
os.environ['nnUNet_results'] = 'results/'  
os.environ['nnUNet_raw'] = 'raw/'
os.environ['nnUNet_preprocessed'] = 'preprocessed/'

from swin_umamba.nnunetv2.paths import nnUNet_results, nnUNet_raw

if __name__ == '__main__':


    # predict a bunch of files
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
        )
    predictor.initialize_from_trained_model_folder(
        join(os.environ['nnUNet_results'], 'Dataset050_FSOut/nnUNetTrainerDepth_SwinUMambaD__nnUNetPlans__2d/'),
        use_folds=(0, ),
        checkpoint_name='checkpoint_best.pth',
    )


    from swin_umamba.nnunetv2.imageio.depth_npy_reader_writer import DepthNpy2dIO
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('TkAgg')
    
    img, props =  DepthNpy2dIO().read_images([join(os.environ['nnUNet_raw'] , 'Dataset050_FSOut/imagesTr/Bucket_0_0000.npy')])


    
    ret = predictor.predict_single_npy_array(img, props, None, None, False)

    from swin_umamba.nnunetv2.training.loss.l2_loss import L2

    print(img.shape, ret.shape)
    #loss = L2()
    #print("L2 LOSS" , loss(input=ret, target=img[0, ...]))

    plt.imshow(ret[0,...])
    plt.show()

    plt.imshow(img[0,0,...])
    plt.show()