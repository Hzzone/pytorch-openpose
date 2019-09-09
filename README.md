## pytorch-openpose

This is a modified version of the original repo specialized for body pose estimation using models trained on MPII-like data.e Estimation**, and the pytorch model is directly converted from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) caffemodel by [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch). You could implement face keypoint detection in the same way if you are interested in. Pay attention to that the face keypoint detector was trained using the procedure described in [Simon et al. 2017] for hands.

### Model Download

```
mkdir model
wget -P model/ https://www.dropbox.com/s/fol13clg8jl80bg/pose_iter_146000.caffemodel.pt
```

This is the pre-trained model from from the [original CVPR 2017 OpenPose repo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation), converted using [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch).  
For reference, the original caffe model can be found [here](http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_146000.caffemodel). 

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
