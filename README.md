## [DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation](https://github.com/Reagan1311/DABNet)

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

#### Introduction

This project contains the code for:  [**DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation**](https://arxiv.org/pdf/1905.02423.pdf)  by [Gen Li](https://github.com/Reagan1311).

<p align="center"><img width="100%" src="./image/architecture.png" /></p>

As a pixel-level prediction task, semantic segmentation needs large computational cost with enormous parameters to obtain high performance. Recently, due to the increasing demand for autonomous systems and robots, it is significant to make a tradeoff between accuracy and inference speed. In this paper, we propose a novel Depthwise Asymmetric Bottleneck (DAB) module to address this dilemma, which efficiently adopts depth-wise asymmetric convolution and dilated convolution to build a bottleneck structure. Based on the DAB module, we design a Depth-wise Asymmetric Bottleneck Network (DABNet) especially for real-time semantic segmentation, which creates sufficient receptive field and densely utilizes the contextual information. Experiments on Cityscapes and CamVid datasets demonstrate that the proposed DABNet achieves a balance between speed and precision. Specifically, without any pretrained model and postprocessing, it achieves 70.1% Mean IoU on the Cityscapes test dataset with only 0.76 million parameters and a speed of 104 FPS on a single GTX 1080Ti card.

#### Installation
- Python 3.6; PyTorch 1.0; CUDA 9.0; cuDNN V7
- numpy, matplotlib
- Clone this repository.
```
git clone https://github.com/Reagan1311/DABNet
cd DABNet
```

#### Dataset
You need to download the two dataset——CamVid and Cityscapes, and put the files in the `dataset` folder with following structure.
```
├── camvid
|    ├── train
|    ├── test
|    ├── val 
|    ├── trainannot
|    ├── testannot
|    ├── valannot
|    ├── camvid_trainval_list.txt
|    ├── camvid_train_list.txt
|    ├── camvid_test_list.txt
|    └── camvid_val_list.txt
├── cityscapes
|    ├── gtCoarse
|    ├── gtFine
|    ├── leftImg8bit
|    ├── cityscapes_trainval_list.txt
|    ├── cityscapes_train_list.txt
|    ├── cityscapes_test_list.txt
|    └── cityscapes_val_list.txt           
```

#### Training

- For help on the optional arguments you can run: `python train.py -h`.
Basically, in the `train.py`, you can set the dataset, train_type (on train or ontrainval).
```
python train.py --dataset ${camvid, cityscapes} --train_type ${train, trainval} --max_epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}  --resume ${CHECKPOINT_FILE}
```
##### Here are some examples:
- train on Cityscapes dataset
```
python train.py --dataset cityscapes
```
- train on CamVid dataset (trainval)
```
python train.py --dataset camvid --train_type trainval --max_epochs 1000 --lr 1e-3 --batch_size 16
```
- During training course, every 50 epochs, we will record the mean IoU of train set, validation set and training loss to draw a plot, so you can check whether the training process is normal.

Loss vs Epochs            |  Val. Acc. vs Epochs
:-------------------------:|:-------------------------:
![alt text-1](https://github.com/Reagan1311/DABNet/blob/master/image/iou_vs_epochs.png)  |  ![alt text-2](https://github.com/Reagan1311/DABNet/blob/master/image/loss_vs_epochs.png)


#### Testing

- After training, the checkpoint will be saved at `checkpoint` folder.
```
python test.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}
```

#### Results

- quantitative result: [Cityscapes official evaluation](https://www.cityscapes-dataset.com/anonymous-results/?id=16896cc219a6d5af875f8aa3d528a0f7c4ce57644aece957938eae9062ed8070)

- qualitative segmentation examples:

<p align="center"><img width="100%" src="./image/DABNet_demo.png" /></p>

#### Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
 @article{wang2019lednet,
  title={LEDNet: A Lightweight Encoder-Decoder Network for Real-time Semantic Segmentation},
  author={Wang, Yu and Zhou, Quan and Liu, Jia and Xiong，Jian and Gao, Guangwei and Wu, Xiaofu, and Latecki Jan Longin},
  journal={arXiv preprint arXiv:1905.02423},
  year={2019}
}
```

#### Tips

- Limited by GPU resources, the project results need to be further improved...
- It is recommended to pre-train Encoder on ImageNet and then Fine-turning Decoder part. The result will be better.

#### Reference

1. [**Deep residual learning for image recognition**](https://arxiv.org/pdf/1512.03385.pdf)
2. [**Enet: A deep neural network architecture for real-time semantic segmentation**](https://arxiv.org/pdf/1606.02147.pdf)
3. [**Erfnet: Efficient residual factorized convnet for real-time semantic segmentation**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8063438)
4. [**Shufflenet: An extremely efficient convolutional neural network for mobile devices**](https://arxiv.org/pdf/1707.01083.pdf)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.0-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/aur/license/pac.svg
[lic-url]: https://github.com/xiaoyufenfei/LEDNet/blob/master/LICENSE
