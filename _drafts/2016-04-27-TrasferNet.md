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
논문에서 제시한 모델(DeconvNet)은 아래 그림과 같이 conv 네트워크와 deconv 네트워크로 구성되어 있다. conv 네트워크는 입력 이미지가 들어오면 feature extractor와 같은 역할을 해서 feature representation 형태의 벡터 혹은 행렬을 결과값으로 도출해 낸다 (일반적인 CNN 모델과 동일하다).<br>
반면 deconv 네트워크는 extract된 feature를 가지고 이미지와 같은 크기의 확률 맵을 생성한다. 이 확률 맵은 각 픽셀이 어떤 label에 속해있는지를 알려주는 역할을 한다.
<div class="imgcap">
<img src="/assets/TransferNet/deconvnet_overview.png">
</div>
conv 네트워크는 마지막 classification 레이어를 제거한 VGG-16 네트워크를 사용한다. deconv 네트워크는 conv 네트워크를 뒤집은 모양인 대신 pooling은 unpooling으로, conv는 deconv로 바꾸고 ReLU는 conv 네트워크와 같으며 conv 네트워크는 fowardprop시 activation 이미지의 크기가 점차 줄어들지만 deconv 네트워크는 반대로 unpooling과 deconv 레이어를 거치면서 커지는 양상을 보인다는 차이가 있다.

### Deconvolution Network
<div class="imgcap">
<img src="/assets/TransferNet/unpool_deconv.png">
</div>

#### Unpooling
pooling 레이어는 얕은 레이어에서 noisy한 activation들을 걸러주는 역할을 하고, 깊은 레이어에서는 robust한 activation만 남기는 역할을 한다. 하지만 이 작업들은 pooling 레이어의 receptive field에 있는 spatial한 정보들이 사라질 수 있기 때문에 segmentation이나 super-resolution과 같이 정확한 정보가 필요한 문제에 대해서는 문제가 된다.<br>
이 문제를 해결하기 위해 논문에서는 deconv 네트워크에 unpooling 레이어를 추가 하였다. unpooling 레이어는 위 그림처럼 pooling 레이어의 역연산을 통해 activation의 원본 크기를 재구성한다. 어떤식으로 unpooling이 일어나는지 살펴보면, pooling시에 activation의 최대값의 위치를 저장해놓고 (이를 switch variable이라 부른다) unpooling 할 때 switch variable 자리에 입력값을 그대로 집어넣어 activation의 크기를 원래대로 맞추는 것이다. 일단 이렇게 하면 크기는 복구가 되지만 switch variable의 자리 외에는 공백이 생기게 된다. 이 공백을 deconvoution 연산을 통해 메꾸는 것이다.

#### Deconvolution
말했듯이 unpooling 레이어를 거치면서 결과값의 크기는 커졌지만 activation map은 sparse한 상태이다. deconv 레이어는 마치 conv 레이어의 역연산처럼 동작하여 하나의 입력값을 이용해 여러개의 결과값을 만든다. 위 그림과 같이 deconv 레이어를 통과한 activation map은 커지고(enlarge), dense한 것을 볼 수 있다.

deconv 네트워크에서 얕은 레이어들은 물체의 전체적인 모양을 캡쳐하는 경향을 보이고 깊은 레이어들은 class-specific한 디테일들을 잡으려 한다 (마치 conv 네트워크의 반전된 형태와 같다). 그러므로 conv와 deconv 네트워크를 거치고 난 최종 결과값은 class-specific한 shape의 정보를 담고 있기 때문에 segmentation을 할 때 더 좋은 결과를 보일 수 있다.

### Analysis of Deconvolution Network
논문에서 제안한 알고리즘은 물체의 정확한 segmentation을 위해 중요하다. 간단한 deconv 연산만을 수행하는 다른 논문들은 coarse한 activation map을 생성하지만 논문의 알고리즘 (Deep DeconvNet)을 사용하면 dense한 pixel-wise 확률 맵을 생성하기 때문에 더 좋은 결과값을 보인다.
<div class="imgcap">
<img src="/assets/TransferNet/visual_deconv.png">
</div>
이 그림을 보면 DeconvNet이 내부적으로 어떻게 동작하는지 이해하기 쉽다. deconv 레이어들을 통과하면서 점점 coarse한 map에서 dense하고 디테일한 map으로 바뀐다. 또 얕은 레이어들은 위치나 모양, 영역등과 같이 물체의 전체적이고 개략적인 모습만 캡쳐하는 반면 깊은 레이어에서는 더 복잡한 패턴들을 발견하려는 경향을 보인다.<br>
또한 unpooling과 deconv 레이어가 하는 일도 다른 것을 확인 할 수 있는데, unpooling은 강한 activation의 위치를 기억하고 이를 이용하기 때문에 example-specific한 구조를 캡쳐해낸다. 그래서 더 높은 해상도에서 물체의 디테일한 구조를 재구성이 가능하다.<br>
반대로 deconv 레이어는 class-specfic한 형태를 캡쳐한다. 그래서 타겟 클래스와 연관이 높은 activation은 증폭되고 노이즈 activation 들은 점점 사라지게 된다. 이렇게 서로 다른 역할을 하는 레이어를 쌓아 정확한 segmentation 맵을 만들 수 있는 것이다.

위 예시 그림은 각 레이어에서 가장 표현이 잘된 activation을 골라서 보여준 것이라고 한다.<br>
(a)는 입력 이미지이며 (b)는 14x14 deconv 레이어를 통과한 결과값이다. (c)는 28x28 unpooling 레이어, (d)는 28x28 deconv 레이어, (e)는 56x56 unpooling, (f)는 56x56 deconv 레이어, (g)는 112x112 unpooling, (h)는 마지막 112x112 deconv 이고 (i)는 224x224 unpooling, (j)는 마지막 224x224 deconv 레이어이다. 아마 같은 크기의 map을 가진 레이어들은 중간에 몇 개씩 스킵한 것 같다.<br>
아까 설명하길 deconv 레이어는 class-specific한 형태만을 캡쳐한다고 하였는데 (h), (i), (j) 그림에서 왼쪽 상단의 노이즈들이 레이어를 거치면서 점차 사라지는 현상을 볼 수 있다.
<div class="imgcap">
<img src="/assets/TransferNet/fcn_vs_deconvnet.png">
</div>
나온 결과값을 FCN-8s과 비교하면 DeconvNet이 더 자세한 activation map을 그린다.

### System Overview
논문은 semantic segmentation을 instance-wise segmentation으로 바꾸어 해결한다. 이 말은 전체 이미지가 아닌 물체를 포함할 가능성이 있는 sub-image (instance라 부른다) 를 입력으로 받고 pixel-wise 예측값을 결과값으로 낸다는 것이다. 그래서 최종적인 segmentation 결과는 각각의 sub-images에서 뽑은 proposal들을 네트워크에 넣어 예측 맵을 얻고 이를 종합해서 최종 원본 이미지의 segmentation 맵을 만들게 된다.<br>
이렇게 하면 생기는 장점은 다양한 스케일에서 segmentation을 돌리기 때문에 물체의 자세한 디테일을 식별 할 수 있다. 또한 예측할 탐색 공간(search space)을 줄이기 때문에 학습시 메모리를 줄일 수 있다.

## Training
DeconvNet은 VGG-16보다 두배나 깊기 때문에 그만큼 parameter의 수도 많다. 하지만 parameter의 수가 많으면 데이터의 수도 많아야 하는데 PASCAL 데이터셋은 그만큼 많지는 않다. 그래서 논문에서는 여러가지 방법들을 추가해서 네트워크를 학습 시켰다고 한다.

원래 VGGNet은 BN을 포함하고 있지 않지만 논문에서는 conv, deconv 레이어 뒤에 BN 레이어를 추가했다. 또한 관측해본 바로 BN을 넣지 않으면 나쁜 local optimum에 빠지기 때문에 BN이 매우 중요하다고 한다.

BN을 사용하면 local optima를 탈출하는데 도움이 되기는 하지만 정말 많은 parameter의 수에 비해 학습 데이터가 적기 때문에 DeconvNet을 사용하는 의미가 없게 된다. 그래서 논문에서는 두 단계의 학습을 통해 이 문제를 해결하였다고 한다. 이 학습 방법을 간단하게 요약하면 아래와 같다.

1. 쉬운 예제들로 먼저 학습을 한다. 쉬운 예제들은 ground-truth annotation 된 데이터들을 가지고 물체가 중앙에 오게 이미지를 자르는 과정을 통해 만든다. 이 방법을 사용하면 물체들의 크기나 위치를 제한 시켜서 탐색 공간을 줄이고 이를 통해 적은 데이터로도 효과적으로 학습이 될 수 있다.

2. 논문을 보고 잘 이해가 안가긴 하는데 아마 원본 이미지에서 sub-image (proposal)를 추출 한 뒤, 이 sub-image가 물체의 ground-truth와 어느정도 겹치면 이 sub-image를 학습 데이터로 사용한 다는 것 같다.

정리하자면 내 생각으로는 일차적으로 원본 이미지를 임의로 수정해서 학습에 용이하게 만든 후 학습시키고, 마지막으로 원본 이미지를 instance-wise 하게 segmentation을 돌린다는 의미 같다.

## Inference
Instance-wise 하게 돌리면 문제가 몇몇 proposal들은 위치가 잘못 되거나(misalignment) 혹은 배경때문에 잘못된 예측을 할 우려가 있다. 그래서 논문에서는 aggregation 단계에서 이런 노이즈들을 줄이는 방법을 제시한다. score map에서 pixel-wise하게 각각 클래스에 대해 **평균이나 최대값**을 구하는 방법을 취하면 충분히 효과적으로 robust한 결과를 얻을 수 있다고 한다. (자세하게는 적지 않도록 하겠다)<br>
그리고 클래스의 조건부 확률 분포 맵은 위에서 얻어진 score map을 가지고 softmax에 넣어서 도출해 낸다 (일반적인 Neural net 분류 문제 풀이 방식과 유사하다). 마지막으로 여기다가  fully-connected CRF를 적용시킨다는데 CRF에 대해 잘 모르기도 하고, 논문을 읽어도 무슨말인지 잘 모르겠다..

논문에서는 DeconvNet은 FCN과 상호보완적인 특성을 가지고 있다고 한다. DeconvNet은 물체의 자세한 디테일을 찾는데 장점이 있는 반면 FCN은 물체의 전체적인 모양을 잡는데 특화되어 있기 때문이다. 그리고 instance-wise 예측은 다양한 스케일의 물체를 다루는데 효과적이지만, FCN은 이미지 내부의 컨텍스트를 찾는데 장점이 있다고 한다.

어쨌거나 FCN과 DeconvNet은 상호보완적인 관계이니 논문에서는 두 방식을 앙상블해서 적용시켰다. 앙상블은 간단하게 두 방식으로  클래스 조건부 확률 분포 맵을 만들고 두 이미지의 mean 값을 CRF에 넣어서 마지막 segmentation 결과를 만드는 방식으로 적용한다.

## Eexperiments
DeconvNet의 디테일한 부분은 간단하게 요약만 하고 넘어가겠다.
<div class="imgcap">
<img src="/assets/TransferNet/deconvnet_summarize.png">
</div>

- 위 이미지는 DeconvNet의 전체적인 구조를 나타낸다. 참고로 네트워크는 약 252M (2억5천만개)의 parameter를 가진다..
- 데이터셋은 PASCAL VOC 2012을 사용하였고 다른 추가 데이터는 사용하지 않았다.
- 네트워크의 hyperparameter의 설정은 논문을 읽어보면 될 것 같고, 네트워크 학습은 Titan X 12GB로 6일 정도 걸렸다고 한다.
- 테스트 단계에서 물체의 proposal(sub-image)를 생성할 때 EdgeBox 알고리즘을 사용한다. 관련 논문을 보지 않아서 어떤식으로 동작하는지는 잘 모르겠지만 이 논문에 의하면 2000개 정도의 proposal을 만들고, objectness score를 통해 상위 50개의 proposal을 골라낸다. 실험을 통해 이 정도의 proposal만 이용해도 충분한 성능을 낼 수 있다고 한다. 각 proposal에서 클래스 확률 맵을 만들고 종합하는 과정은 위에서 서술한 것과 같다.

<div class="imgcap">
<img src="/assets/TransferNet/deconvnet_pascal.png">
</div>
<div class="imgcap">
<img src="/assets/TransferNet/deconvnet_ex1.png">
</div>

- PASCAL VOC 2012 데이터셋에서 실험한 성능은 위 표와 같다. EDeconvNet은 FCN과 앙상블한 버전인데 deconvnet만 사용한 결과는 DeepLab-CRF보다 근소하게 밀리지만 앙상블한 알고리즘은 state-of-the-art의 성능을 낸다.
- 위 그림 에서 볼 수 있는 것처럼 instance-wise 예측을 하면 좋은 성능을 낼 수 있다. 아래 그림에서 proposal의 수를 늘리면 다양한 스케일의 물체를 검출 할 수 있기 때문에, 적은 proposal에서 검출되지 않은 사람 물체도 검출 된다.

## Summary
논문에서 주장하는 DeconvNet의 장점은 다음과 같다.

- Coarse-to-fine 구조로 물체를 재구성하는 방식을 취해 segmentation이 정확하다.
- Instance-wise 예측을 사용하면 고정된 크기의 receptive field 만 사용할 수 있는 제한을 없앨 수 있으므로 물체의 다양한 스케일에 유연하게 대처할 수 있다.
- FCN과 앙상블하면 (EDeconvNet) 두 알고리즘의 상호보완적인 관계에 의해 성능 향상이 일어난다.
- EDeconvNet은 PASCAL VOC 2012 데이터셋에서 state-of-the-art의 성능을 보인다.

여기에 내 의견과 궁금증들을 좀 더 덧붙여보면,

- 논문에서 언급을 한건데 내가 캐치를 못한건지, 아니면 언급이 안되어있는건지는 모르겠는데 VGG-16 네트워크를 사용할 때 pre-trained 된 모델을 사용한걸까? Caffe로 구현된 걸 보면 사용한 것 같긴 하지만  원래 VGG-16 네트워크는 BN을 사용하지 않았을텐데 이 네트워크의 weight들을 그대로 들고와서 BN을 적용시켜도 괜찮은지 궁금하다.
- 네트워크의 parameter가 너무 많다. 그러면 당연하게도 학습에도 너무 오래걸리고 데이터가 훨씬 더 많이 필요하다. 하지만 segmentation 데이터셋은 classification 문제보다 데이터셋이 적은데, 이 논문은 이를 해결하기 위해 두 단계의 학습 프로세스를 제시하였다. 근데 이 방법은 좀 매끄럽지 못하고 부자연스러운 것 같다. parameter를 줄일 수 있는 방법은 없을까?
- 구현시의 의문점. 하나의 이미지에 여러 물체들이 들어있을 경우 모든 물체를 식별하기 위해서는 classification이 아닌 detection 영역인거 같은데 classification으로 처리할 수 있는지 궁금하다. EdgeBox를 사용하면 바운딩 박스에 하나의 instance만 들어가게 되어서 상관없는 것일까? (사실 이 의문은 아래 논문들을 읽을 때도 계속 드는 생각인데 아직 해결하지 못하였다 ㅠㅠ)

# Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation
---
두 번째로 소개할 논문은 DeconvNet을 활용하여 아직 완전히 Transferable한 네트워크는 아니지만 작은 데이터들을 가지고 semi-supervised한 segmentation 문제를 푸는 방법을 제시하고 있다.

## Introduction
이 논문 이전에 segmentation을 할 때 필요한 ground-truth를 만드는데 너무 많은 자원이 필요하기 때문에 이를 해결할 수 없을 까 해서 semi 혹은 weakly-supervised 방식으로 segmentation을 접근하는 방법론이 제시되었다. 하지만 이 논문에서는 기존 방식들은 ad-hoc (어떻게 번역해야 할지 잘 모르겠다) 방식에 의존하고 수렴이 보장되지 않는 단점을 가지고 있다고 한다. 또한 구현방법이 tricky하고 알고리즘 자체가 복잡하다.

이 논문에서는 DecoupledNet이라는 새로운 방법론을 제시한다. 이 네트워크는 DeconvNet 기반으로 중간에 브릿지 레이어를 가지고 있어서 segmentation 네트워크에 class-specific한 정보들을 전달에 용이하도록 수정한 형태이다 (자세한 내용은 후술하겠다). 학습할 때는 두 개의 네트워크 따로 학습하게 되며 이런 decoupling한 구조는 segmentation할 때 탐색 공간을 매우 줄일 수 있어서 비교적 적은 데이터로도 학습이 가능하고 따로 후처리 과정등을 하지 않아도 inference시 좋은 성능을 낼 수 있다.

## Overview
<div class="imgcap">
<img src="/assets/TransferNet/decoupled_overview.png">
(논문 이미지 참 잘만든다..)
</div>
DecoupledNet은 classification 네트워크, segmentation 네트워크와 두 네트워크를 연결해주는 연결(bridge) 레이어로 구성되어 있다. 네트워크의 구조만 보면 연결 레이어의 유무이외에는 DeconvNet과 매우 흡사한 형태이고 각 네트워크의 역할도 비슷하다 (약간 다르다).

다시 설명하자면 classification 네트워크는 입력 이미지를 문자 그대로 "분류" 하며 segmentation 네트워크는 pixel-wise segmentation 한다. 여기서 약간 다른점이 있는데 이는 segmentation을 할 때 **각** label마다 진행 한다는 것이다 (자세한 내용은 후술).<br>
하지만 이러한 구조는 classification과 segmentation 작업 사이에 연결성을 잃어버릴 위험이 있기 때문에 논문에서는 연결 레이어를 하나 추가해서 문제를 보완한다. 이 레이어를 추가함으로 인하여 **두 네트워크를 각각 다른 목적 함수(objective function)를 사용**해서 학습시킬 수 있고 동시에 두 네트워크를 융합해서 최종 목표인 segmentation을 효과적으로 진행 할 수 있다.

네트워크의 학습은 image-level annotation (e.g 개, 고양이..)를 이용해서 classification 네트워크를 먼저 학습시키고, 그 후 적은 양의 pixel-level annotation 데이터를 이용해서 연결 레이어와 segmentation 네트워크를 학습시킨다. 그래도 데이터의 양이 적긴하지만 이는 augmentation을 이용해서 해결한다.

종합해보자.

- DecoupledNet은 classification과 segmentation을 **분리(decouple)** 시킨다. 그래서 각 네트워크를 따로 학습시킬 수도 있고 심지어 classification 네트워크는 pre-trained 네트워크를 갖다써도 상관없다. 적은 양의 데이터로 segmentation 네트워크와 연결 레이어만 학습시키면 된다.
- 연결 레이어는 class-speific한 맵을 구축하고 
- 두 네트워크를 각각 학습시킬 수 있기 때문에 전체적인 학습 프로시저가 간단하다.

## Architecture

### Classification Network
간단하게 말하자면 VGGNet이랑 똑같지만 notation를 정리하기 위해서 약간의 설명을 하려고 한다.<br>
이 네트워크는 입력 이미지 $\boldsymbol{x}$를 받고 score 벡터 $S(\boldsymbol{x};\theta) \in R^L$ 를 도출한다. 이 score 벡터를 이용해 loss값을 아래와 같이 계산 할 수 있다.

$$
min _{\theta _c}\sum _i e _c(\boldsymbol{y} _i, S)
$$

이 때, $\boldsymbol{y} _i \in \\{ 0, 1 \\}^L $ 는 i번째 이미지의 ground-truth label이고 $e(\cdot)$ 는 score 벡터 $S(\cdot)$ 와 ground-truth와의 loss 함수(sigmoid cross-entropy)을 의미한다. 여기에 VGG-16 네트워크를 사용했고 DeconvNet과 다르게 fc 레이어도 사용한다. 

### Segmentation Network
이 네트워크는 연결 레이어로부터 class-speific한 activation map $\boldsymbol{g}^l _i$를 받아 **두 개의 채널**로 구성된 segmentation 맵 $M(\boldsymbol{g}^l _i; \theta _s)$ 를 내고 이 값을 softmax 함수에 집어넣는다. 맵은 $M _f(\cdot), M _b(\cdot)$ 으로 구성되어 있는데 각각 전경과 배경 맵을 나타낸다. 이 맵을 이용해서 per-pixel regression을 돌리고 아래 식을 최소화하는 방향으로 네트워크를 학습한다.

$$
min _{\theta _s}\sum _i e _s(\boldsymbol{z}^l _i, M(\boldsymbol{g}^l _i, \theta _s))
$$

여기서 $\boldsymbol{z}^l _i$ 는 카테고리 $l$ 에 관한 **binary ground-truth** segmentation 마스크를 의미하고, $e(\cdot)$ 은 loss 함수(softmax) 이다.

중요한 점! DeconvNet과 달리 DecoupledNet은 segmentation 네트워크에서 binary 분류를 수행 한다. 무슨말이냐면 DeconvNet은 이 픽셀이 예를 들어 개인지 고양이인지 판별하기 때문에 가능한 모든 카테고리에 대해 정보를 가지고 있게 된다. 하지만 DecoupledNet은 segmentation 단계에서 이 픽셀이 물체인지 아닌지만 판단하도록 바뀌었다. 이렇게 바꾸면 장점으로 **parameter의 개수가 매우 감소**한다. 왜냐면 결과값의 채널이 모든 클래스의 개수에서 2개로 줄어들었기 때문인데, 이 성질을 통해 적은 pixel-wise annotation만으로 네트워크를 학습 시킬 수 있다.

### Bridging Layer
segmentation 네트워크를 통해 특정 클래스의 segmentation 마스크를 생성하기 위해서는 입력값으로 class-specific한 정보와 물체의 모양을 생성하기 위한 공간적인 정보가 필요하다. 그래서 segmentation 네트워크 이전에 연결 레이어를 집어넣어 class-specific한 activation 맵 $\boldsymbol{g}^l _i$ 를 생성한다. 이 때 activation map은 각 label $l \in L _i$ 마다 주어져야 한다.

물체의 공간적인 정보를 넣기 위해 classification 네트워크에서 fc 레이어 이전 레이어에서 빼온 결과값을 이용한다. DecoupledNet은 이 결과값을 pool5 레이어에서 가져오는데 이는 conv나 pooling 레이어의 결과값은 공간적 정보를 저장하는데 효과적이기 때문이라고 한다 (사실 fc 레이어는 공간 정보가 없다). 논문에서는 이 값을 $\boldsymbol{f} _{\text{spat}}$ 라고 명명하였다.

$\boldsymbol{f} _{\text{spat}}$ 의 activation이 물체의 모양을 생성하는데 중요한 정보를 가지고 있지만 이미지가 갖고 있는 모든 label에 대한 정보들을 가지고 있다. 그래서 논문에서는 $\boldsymbol{f} _{\text{spat}}$ 에서 class-specific한 정보를 식별해야 한다고 한다. 이는 class-speific한 saliency 맵을 만들어 활용 할 수 있다.

<div class="imgcap">
<img src="/assets/TransferNet/saliency.png"><br>
모든 이미지에서 이렇게 식별하기 쉬운 saliency 맵을 만들어내진 않는다..
</div>
Saliency 맵을 만드는건 간단하게 설명하면 각 class label에 대한 score 벡터를 뽑아내고 여기서 원하는 레이어까지 backprop을 한다. 이 때 일반적인 backprop와 달리 score 벡터의 derivate 초기값을 원하는 label에 해당하는 인덱스에 1로 준다. 이렇게 하면 다른 정보들은 제거되고 원하는 label에 의한 공간 정보값을 뽑아낼 수 있다 (위 그림 참고).

어쨌든 salency 맵을 논문에서는 $\boldsymbol{f}^l _{\text{cls}}$ 라 notate 하고 있으며, class-speific한 activation 맵 $\boldsymbol{g}^l _i$ 는 이 $\boldsymbol{f} _{\text{spat}}$ 와 $\boldsymbol{f}^l _{\text{cls}}$ 를 채널 방향으로 concat하고 이를 두 개의 fc 레이어 (연결 레이어)에 fowardprop 시켜 얻을 수 있다. 그래서 결과적으로 segmentation 네트워크에 들어오는 activation 맵은 공간적인 정보(pool5) 와 class-speific한 정보(sailency) 를 모두 가질 수 있게 되는 것이다.

# 레퍼런스
Learning Deconvolution Network for Semantic Segmentation [[arXiv](http://arxiv.org/abs/1505.04366)] [[project page](http://cvlab.postech.ac.kr/research/deconvnet/)]<br>
Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation [[NIPS](https://papers.nips.cc/paper/5858-decoupled-deep-neural-network-for-semi-supervised-semantic-segmentation.pdf)] [[project page](http://cvlab.postech.ac.kr/research/decouplednet/)]<br>
Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network [[arXiv](http://arxiv.org/abs/1512.07928)] [[project page](http://cvlab.postech.ac.kr/research/transfernet/)]
