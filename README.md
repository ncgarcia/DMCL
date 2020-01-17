# Code for the paper [DMCL: Distillation Multiple Choice Learning for Multimodal Action Recognition](https://arxiv.org/abs/1912.10982)

## Overview
``codebase/utils.py``: essentially contains functions to deal with the infrastructure, like building filepaths and read lists of files. This file contains the paths to data and lists that should be modified to match your setup.   
``codebase/restorers.py``: functions to restore checkpoints.  
``codebase/parsers.py``: functions to parse tfrecords, i.e. to read tfrecords and build training and testing batches.  
``nets/resnet_official/``: forked from tensorflow resnet official repo. It is modified to reflect the model proposed in [A Closer Look at Spatiotemporal Convolutions for Action Recognition, CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf).   
``dmcl.py``: implements the method presented in the paper.

### Prerequisites
This work was developed using Python 3 and Tensorflow 1.12

## Data Pipeline
The input to this model are batches of frames of three modalities: RGB, optical flow, and depth. Each modality is associated with a different deep neural network.
This is how we implemented the data pipeline for this experiment.

1. write a json file that contains three lists of video_ids: training, validation, and testing. 

2. training tfrecords: write *one* tfrecord for each of the training video_ids.  
Each tfrecord, corresponding to a video_id, contains three lists: paths to *all* the RGB frames, paths to *all* the depth frames, and paths to *all* the optical flow frames.

3. test/val tfrecords: write *ten* tfrecords for each of the testing and validation video_ids.  
Each tfrecord contains three lists: paths to the RGB frames, paths to the depth frames, and paths to the optical flow frames.  
Each tfrecord corresponds to a clip of eight randomly sampled frames of a video. The indices are the same across modalities.  
These frames are samped once when writing these tfrecords and then re-used for all the experiments. <!-- , in order to guarantee consistency of validation and test sets.  -->
The final prediction for a validation/test video_id is the average of the predictions for each of the ten clips.   


Steps *1,2,3* are run only once.

For training: dmcl.py reads the list of training video_ids produced in *step 1*.
For each training step, it randomly samples `batch_sz` videos (default is 5). 
For each of these videos, it is randomly sampled a clip of eight frames. The indices of the sampled frames are the same across modalities.
Thus, one training batch is composed of `batch_sz` clips from different videos, each of eight frames.
Each modality network is served with the corresponding modality batch of samples.

For validation and testing: dmcl.py reads all the validation and testing video_ids produced in *step 1*.
It then proceeds to build the paths to each of the ten clips per video, produced in *step 3*. 
The frames of these clips are used as input to the corresponding modality network.

## Training
```
python dmcl.py --dset=nwucla --eval=cross_view --temp=2 
```
Other options are available, please check `./codebase/utils.get_arguments()`.  
Each experiment run will create two folders: one for logging, with checkpoints and a .txt file with the output, and one for saving checkpoints.  
At the end of the training, the model restores the checkpoint that had the best validation accuracy during training, and runs the test set.  

## Reference

**DMCL: Distillation Multiple Choice Learning for Multimodal Action Recognition** - *[PDF](https://arxiv.org/abs/1912.10982)*   
Nuno Cruz Garcia, Sarah Adel Bargal, Vitaly Ablavsky, Pietro Morerio, Vittorio Murino, Stan Sclaroff 
```
  @article{garcia2019dmcl,
  title={DMCL: Distillation Multiple Choice Learning for Multimodal Action Recognition},
  author={Garcia, Nuno C and Bargal, Sarah Adel and Ablavsky, Vitaly and Morerio, Pietro and Murino, Vittorio and Sclaroff, Stan},
  journal={arXiv preprint arXiv:1912.10982},
  year={2019}
}
```
