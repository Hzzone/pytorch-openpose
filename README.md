## pytorch-openpose

pytorch implementation of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) including **Body and Hand Pose Estimation**, and the pytorch model is directed converted from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) caffemodel by [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch). You could implement face keypoint detection in the same way if you are interested in.

### Model Download
* [dropbox](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0)

`*.pth` files are pytorch model, you could also download caffemodel file if you want to use caffe as backend.

### Todo list
- [x] convert caffemodel to pytorch.
- [x] Body Pose Estimation.
- [x] Hand Pose Estimation.
- [ ] Performance test.
- [ ] Speed up.

### Demo
#### Skeleton

<div align='center'>
<img src='images/skeleton.jpg'>
<div/>

#### Body Pose Estimation

<div align='center'>
<img src='images/body_preview.jpg'>
<div/>

### Hand Pose Estimation

<div align='center'>
<img src='images/hand_preview.png'>
<div/>
