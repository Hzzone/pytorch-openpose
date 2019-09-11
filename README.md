## pytorch-openpose

This fork is a modified version of the original repo specialized for using OpenPose models on MPII-like data. 

### Model Download
Here is the procedure to download a pre-trained MPII OpenPose model:
```
mkdir model
wget -P model/ https://www.dropbox.com/s/fol13clg8jl80bg/pose_iter_146000.caffemodel.pt
```

This is the pre-trained model from [original CVPR 2017 OpenPose repo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation), converted using [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch).  
For reference, the original caffe model can be found [here](http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_146000.caffemodel). 

### Annotation Downloads
The only annotations that need to be downloaded are the MPII masks, which can be downloaded with the following commands.

```
wget http://posefs1.perception.cs.cmu.edu/Users/ZheCao/masks_for_mpii_pose.tgz
tar -xvf masks_for_mpii_pose.tgz -C python/
rm masks_for_mpii_pose.tgz
```

### Citation
Please cite these papers in your publications if it helps your research (the face keypoint detector was trained using the procedure described in [Simon et al. 2017] for hands):

```
@inproceedings{cao2017realtime,
  author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2017}
}

@inproceedings{simon2017hand,
  author = {Tomas Simon and Hanbyul Joo and Iain Matthews and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Hand Keypoint Detection in Single Images using Multiview Bootstrapping},
  year = {2017}
}

@inproceedings{wei2016cpm,
  author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Convolutional pose machines},
  year = {2016}
}
```
