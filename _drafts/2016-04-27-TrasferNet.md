---
layout: "post"
title: "Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network"
excerpt: "TrasferNet and other networks"
date: "2016-04-27 10:00:00"
---

얼마전 KIAS에서 열린 딥러닝 세미나를 다녀왔다. 세미나에서 가장 흥미롭고 재밌었던 발표는 포스텍의 한보형 교수님께서 발표한 이 논문이었다. 일단 발표를 정말 잘하셔서 더욱 흥미롭게 들은 것도 하나의 이유였고, 제시한 논문의 당위성도 공감되고 신기했기 때문이다.<br>
이 논문(그리고 이전 논문에서도) 의 주장은 segmentation을 할 때 annotation이 된 데이터를 얻는 것은 매우 힘들고 비용이 많이 들기 때문에 최소한의 annotation된 데이터를 이용해 semi-supervised learning을 하거나, 다른 데이터 셋에서 학습된 모델을 가지고 주어진 도메인의 데이터를 마치 transfer learning처럼 학습 하자는 것이다.

개요는 이정도로 설명하고 본 포스트는 KIAS 발표와 비슷하게 한보영 교수님 그룹에서 진행한 논문 3개를 한꺼번에 설명할 것이다. 이는 이전 논문에서 제시한 모델을 통해 새로운 모델을 제시하는 방법론을 취하기 있기 때문이고 물론 KIAS에서도 이렇게 발표됬기 때문이다..

# Learning Deconvolution Network for Semantic Segmentation
---
이 논문에서는 segmentation을 위해 DeconvNet이라는 네트워크 모델을 제시하였다. 사실 segmentation에 관한 지식도 거의 없고 이 논문으로 처음 이 분야에 대해 접한지라 Related work와 같은 세세한 부분은 약간씩 건너뛰고 이 논문에서 주장하는 내용만을 주로 담으려고 한다.

## Introduction
CNN의 등장으로 neural network 진영은 이미지 인식 분야에서 엄청난 성능 향상을 보이고 있다. 그 중 segmentation 분야에서는 FCN이라는 모델이 있는데(이하의 설명은 이 논문에서 주장하는 것이고 아직 FCN 논문을 읽어보지 못했다) 이 모델은 이미지의 모든 local region을 classify하여 coarse한 label mapd을 얻고, 이를 간단한 deconvolution을 통해 pixel-level labeling을 얻는다. 이 방식은 전체 이미지를 입력값으로 받을 수 있고 괜찮은 성능을 빠르게 얻을 수 있다는 장점이 있다. 하지만 FCN 방식의 segmentation은 크게 두가지의 문제가 있다.

이 모델은 고정된 크기의 receptive field를 가지고 있기 때문에 single-scale의 semantic만 처리할 수 있다. 그래서 receptive field보다 크거나 작은 물체들은 짤리거나 아예 labeling이 되지 않게 된다. (아래 그림 참고)<br>
다시 말해 local한 정보만 가지고 픽셀의 label을 예측하기 때문에 
그림 (a)와 같이 물체가 매우 클 경우(receptive field보다 클 경우) 하나의 물체를 여러 label로 예측하는 경우가 발생한다. 또한 하나의 scale만 처리하기 때문에 그림 (b)와 같이 작은 물체는 무시되어버린다. skip-아키텍쳐를 이용해서 문제를 해결하려는 시도가 있었지만 이 방법을 적용해도 눈에 띄는 성능 향상이 이루어지지는 않는다.
<div class="imgcap">
<img src="/assets/TransferNet/fcn_problem.png">
</div>
두번째로 label map이 매우 작은 크기이기 때문에 물체의 디테일한 구조들을 가끔 잃어버리기도 한다. FCN 논문에서 제시한 모델같은 경우 label map의 크기가 16x16이고 이를 한번의 deconv를 통해 segmentation 결과를 생성하기 때문에 좋은 결과를 내기 힘들다. 물론 최근에는 이 문제를 CRF와 결합하여 해결하는 경우도 있다고 한다.

논문에서는 위의 문제를 해결하기 위해 완전히 새로운 방식의 모델을 제시하고 이를 이용해 PASCAL VOC 2012 벤치마크에서 state-of-the-art 성능을 보였다고 주장하고 있다 (물론 그 때 당시).

## System Architecture
논문에서 제시한 모델(deconvnet)은 아래 그림과 같이 conv 네트워크와 deconv 네트워크로 구성되어 있다. conv 네트워크는 입력 이미지가 들어오면 feature extractor와 같은 역할을 해서 feature representation 형태의 벡터 혹은 행렬을 결과값으로 도출해 낸다 (일반적인 CNN 모델과 동일하다).<br>
반면 deconv 네트워크는 extract된 feature를 가지고 이미지와 같은 크기의 확률 맵을 생성한다. 이 확률 맵은 각 픽셀이 어떤 label에 속해있는지를 알려주는 역할을 한다.
<div class="imgcap">
<img src="/assets/TransferNet/deconvnet_overall.png">
</div>
conv 네트워크는 마지막 classification 레이어를 제거한 VGG-16 네트워크를 사용한다. deconv 네트워크는 conv 네트워크를 뒤집은 모양인 대신 pooling은 unpooling으로, conv는 deconv로 바꾸고 ReLU는 conv 네트워크와 같으며 conv 네트워크는 fowardprop시 activation 이미지의 크기가 점차 줄어들지만 deconv 네트워크는 반대로 unpooling과 deconv 레이어를 거치면서 커지는 양상을 보인다는 차이가 있다.

### Deconvolution Network
<div class="imgcap">
<img src="/assets/TransferNet/unpool_deconv.png">
</div>
#### Unpooling
pooling 레이어는 얕은 레이어에서 noisy한 activation들을 걸러주는 역할을 하고, 깊은 레이어에서는 robust한 activation만 남기는 역할을 한다. 하지만 이 작업들은 pooling 레이어의 receptive field에 있는 spatial한 정보들이 사라질 수 있기 때문에 segmentation이나 super-resolution과 같이 정확한 정보가 필요한 문제에 대해서는 문제가 된다.<br>
이 문제를 해결하기 위해 논문에서는 deconv 네트워크에 unpooling 레이어를 추가 하였다. unpooling 레이어는 
#### Deconvolution


# 레퍼런스
Learning Deconvolution Network for Semantic Segmentation [[arXiv](http://arxiv.org/abs/1505.04366)] [[project page](http://cvlab.postech.ac.kr/research/deconvnet/)]<br>
Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation [[NIPS](https://papers.nips.cc/paper/5858-decoupled-deep-neural-network-for-semi-supervised-semantic-segmentation.pdf)] [[project page](http://cvlab.postech.ac.kr/research/decouplednet/)]<br>
Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network [[arXiv](http://arxiv.org/abs/1512.07928)] [[project page](http://cvlab.postech.ac.kr/research/transfernet/)]
