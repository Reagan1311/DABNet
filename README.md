## [DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation](https://github.com/Reagan1311/DABNet)

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

#### Introduction

This project contains the code for:  [**DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation**](https://arxiv.org/pdf/1905.02423.pdf)  by [Gen Li](https://github.com/Reagan1311).

<p align="center"><img width="100%" src="./image/architecture.png" /></p>

As a pixel-level prediction task, semantic segmentation needs large computational cost with enormous parameters to obtain high performance. Recently, due to the increasing demand for autonomous systems and robots, it is significant to make a tradeoff between accuracy and inference speed. In this paper, we propose a novel Depthwise Asymmetric Bottleneck (DAB) module to address this dilemma, which efficiently adopts depth-wise asymmetric convolution and dilated convolution to build a bottleneck structure. Based on the DAB module, we design a Depth-wise Asymmetric Bottleneck Network (DABNet) especially for real-time semantic segmentation, which creates sufficient receptive field and densely utilizes the contextual information. Experiments on Cityscapes and CamVid datasets demonstrate that the proposed DABNet achieves a balance between speed and precision. Specifically, without any pretrained model and postprocessing, it achieves 70.1% Mean IoU on the Cityscapes test dataset with only 0.76 million parameters and a speed of 104 FPS on a single GTX 1080Ti card.

#### Dataset
You need to download two dataset, and put the files in the dataset folder with following structure.
- You can download [cityscapes](https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Note: please download [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).
- You can download [CityscapesScripts](https://github.com/mcordts/cityscapesScripts), and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py). It should have this basic structure.
```
├── camvid
     ├── train
     ├── test
     ├── val 
     ├── trainannot
     ├── testannot
     ├── valannot
     ├── camvid_trainval_list.txt
     ├── camvid_train_list.txt
     ├── camvid_test_list.txt
     └── camvid_val_list.txt
├── cityscapes
     ├── gtCoarse
     ├── gtFine
     ├── leftImg8bit
     ├── cityscapes_trainval_list.txt
     ├── cityscapes_train_list.txt
     ├── cityscapes_test_list.txt
     └── cityscapes_val_list.txt           
```

#### Installation
- Python 3.6.x. Recommended using [Anaconda3](https://www.anaconda.com/distribution/)
- Set up python environment

```
pip3 install -r requirements.txt
```

- Env: PyTorch_0.4.1; cuda_9.0; cudnn_7.1; python_3.6, 

- Clone this repository.

```
git clone https://github.com/xiaoyufenfei/LEDNet.git
cd LEDNet-master
```

- Install [Visdom](https://github.com/facebookresearch/visdom).
- Install [torchsummary](https://github.com/sksq96/pytorch-summary)
- Download the dataset by following the **Datasets** below.
- Note: For training, we currently support [cityscapes](https://www.cityscapes-dataset.com/) , aim to add [Camvid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)  and  [VOC](http://host.robots.ox.ac.uk/pascal/VOC/)  and  [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)  dataset


#### Training

- For help on the optional arguments you can run: `python train.py -h`
Basically, in the train.py, you can set the dataset, train_type (on train or ontrainval)

- train on Cityscapes dataset
```
python train.py --dataset cityscapes 
```
- train on Cityscapes dataset (trainval)
```
python train.py --dataset camvid --train_type trainval
```
- train on CamVid dataset (trainval)
```
python train.py --dataset camvid --train_type trainval --lr 1e-3 --batch_size 16
```


#### Resuming-training-if-decoder-part-broken


```
python main.py --savedir logs --name lednet --datadir path/root_directory/  --num-epochs xx --batch-size xx --decoder --state "../save/logs/model_best_enc.pth.tar"...
```

#### Testing

- the trained models of training process can be found at [here](https://github.com/xiaoyufenfei/LEDNet/save/). This may not be the best one, you can train one from scratch by yourself or Fine-tuning the training decoder with  model encoder pre-trained on ImageNet, For instance

```
more details refer ./test/README.md
```

#### Results

- Please refer to our article for more details.

|Method|Dataset|Fine|Coarse| IoU_cla |IoU_cat|FPS|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**LEDNet**|**cityscapes**|yes|yes|**70.6​%**|**87.1​%​**|**70​+​**|

qualitative segmentation result examples:

<p align="center"><img width="100%" src="./images/LEDNet_demo.png" /></p>

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
