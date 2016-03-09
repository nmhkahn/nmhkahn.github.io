---
layout: "post"
title: "Case study of Convolutional Neural Network"
excerpt: "AlexNet, VGGNet, NIN, GoogLeNet, ResNet..."
date: "2016-02-09 16:00:00"
---

### Contents
- AlexNet
- VGGNet
- Network In Network
- GoogLeNet
- PReLU-net
- ResNet

## AlexNet (2012)
Alex Krizhevsky의 이름을 따서 작명된 AlexNet은 2012년 ILSVRC에서 압도적인 winner가 된 네트워크이다. 이 네트워크 이후 ZFNet, NIN, VGGNet, GoogLeNet, ResNet등 다양한 뉴럴넷 기반의 모델들이 ILSVRC 혹은 다른 데이터셋에서 outperform한 결과를 보이게 되는데, AlexNet은 이 돌풍을 열게 한 선두주자라 말할 수 있을 것이다.  이번 포스트에서는 AlexNet이 소개된 논문을 정리하여, AlexNet에서 어떠한 방법론을 사용하여 LeNet을 발전시켰는지 살펴 볼 예정이다.

### Datasets 
평균 이미지를 입력 이미지에 빼어 zero-centered된 이미지를 입력 이미지로 사용하였다.
(자세한 사항은 [Neural Network](/NN) 참조)

### ReLU Nonlinearlity
Vanishing gradient 문제를 해결하기 위해 sigmoid 혹은 tanh 대신 ReLU activation 함수를 사용하였다.<br>
[Neural Network](/NN) 참조

### Multiple GPU 사용
GPU(GTX 580, 3GB VRAM)의 메모리 부족 문제로 두개의 GPU를 병렬로 사용하였다고 한다. 
지금은 GPU의 VRAM이 크다면 잘 사용하지 않는 방법이다.

### Local Response Normalization
<div class="imgcap">
<img src="/assets/Casestudy-CNN/alex-norm1.png">
</div>

Generalization 에러를 줄이기 위해 정규화를 한다.<br>
$a_{x,y}^i$ 는 2D 이미지에서 (x, y) 위치에서 $i$ 번째 커널(필터)을 의미한다. $k, n, \alpha, \beta$ 는 hyper-parameter로 이 논문에서는 $k = 2, n = 5, \alpha = 10^{-4}, \beta = 0.75$ 로 설정했다.<br>
이 정규화는 같은 spatial 위치에 있는 $n$ 만큼 adjust한 필터들의 square-sum을 이용하여 정규화 하는 것이다. 예를들어 $n = 5$ 라면, (2, 3)번째 픽셀에 위치하는 5번째 필터는 그 위치의 3~7번째 필터에 해당하는 결과값을 이용하여 정규화 한다.<br>
참고로 지금은 성능상 큰 이점이 없어서 잘 사용하지는 않는다.

#### Overlapping Pooling
기존의 pooling 레이어는 stride와 필터의 크기를 같게 하는데, 이 논문에서 stride는 3, 필터의 크기는 2로 정하며 pooling 레이어가 overlapped 하게 구성하였다. 이 결과 top-1, top-5를 0.4% 0.3%정도 성능향상이 일어났다고 한다.

### Architecture

<div class="imgcap">
<img src="/assets/Casestudy-CNN/alex-arch1.png">
</div>

[227x227x3]	INPUT<br>
[55x55x96]  CONV1		: 96@ 11x11, s = 4, p = 0<br>
[27x27x96]	MAX POOL1	: 3x3, s = 2<br>
[27x27x96]	NORM1		:<br>
[27x27x256]	CONV2		: 256@ 5x5, s = 1, p = 2<br>
[13x13x256] MAX POOL2	: 3x3, s = 2<br>
[13x13x256] NORM2		:<br>
[13x13x384] CONV3		: 384@ 3x3, s = 1, p = 1<br>
[13x13x384] CONV4		: 384@ 3x3, s = 1, p = 1<br>
[13x13x256] CONV5		: 256@ 3x3, s = 1, p = 1<br>
[6x6x256]   MAX POOL3	: 3x3, s = 2<br>
[4096]		FC6			: 4096 neurons<br>
[4096]		FC7			: 4096 neurons<br>
[1000]		FC8			: 1000 neurons<br>

참고로 예를 들어 설명하자면 만약 입력 이미지가 [227x227x3]이라 가정하고 conv1의 필터는 [11x11]이 96개이고 stride는 4라고 하자.<br>
이 때 conv1을 거치고 나온 출력 volume은 [55x55x96]이다. 이는 (227-11)/4/1 + 1에 의해 55가 나오게 되는 것이고, 96은 필터의 수와 동일하다. parameter의 수는 11*11*3*96으로 35K 정도이다.

conv1을 나온 출력값을 [3x3], stride가 2인 pooling 레이어에 적용하면 출력값은 [27x27x96]으로 나온다. 이는 (55-3)/2 + 1에 의해 27이 나오기 때문이며, parameter의 수는 0이다.

### Details of Learning
AlexNet에서 사용한 수치적인 디테일들은 다음과 같다.

- SGD를 사용하고 mini-batch 사이즈는 128
- 모멘텀은 0.9, weight decay(L2)는 0.0005 (weight decay를 사용하니 **트레이닝 에러**가 줄었다고 한다.)
- weight 초기화는 zero-mean gaussian * 0.01
- dropout 확률은 0.5
- learning rate는 0.01, validation error의 성능향상이 멈추면 10배 감소시킨다.
- 7개의 CNN을 앙상블하여 18.2% -> 15.4%로 향상시켰다.

### Reducing Overfitting
Overfitting을 해결하기 위해 강력한 data augmentation과 dropout 기법을 사용하였다.

#### Data Augmentation
원본 이미지의 augmentation은 CPU에서 동작하고, 그 동안 GPU는 이전 batch를 트레이닝한다. 이미지 augmentation은 크게 두가지로 다음과 같다.

- 원본 이미지는 256x256인데, 이를 224x224의 패치가 되게 랜덤하게 추출한다. 이렇게 추출하면 한 개의 원본 이미지로 2048가지의 경우의 수가 나온다. 상하 대칭으로도 이미지를 똑같은 방법으로 추출한다. 테스트 단계에서는 원본, 상하반전 이미지에서 5개씩 224x224 패치를 추출하여 softmax의 평균을 내어 추측한다. 이 때 5개의 패치는 4개의 코너와 중앙에서 추출한다.
- 두번째로 PCA를 통해 테스트 이미지의 RGB 채널 강도를 변화시키는 augmentation을 진행한다.

#### Dropout
[Neural Network](/NN) 참조 

---

## VGGNet(2014)
VGGNet은 2014년 ILSVRC에서 GoogLeNet과 함께 높은 성능을 보인 네트워크이다. 또한 간단한 구조, 단일 네트워크에서 좋은 성능등을 이유로 여러 응용 분야에서 기본 네트워크로 많이 사용되고 있다.

### Architecture
<div class="imgcap">
<img src="/assets/Casestudy-CNN/vgg-arch1.png">
</div>

VGGNet은 AlexNet과 마찬가지로 224x224의 이미지 크기를 입력으로 사용한다. 입력되는 이미지는 AlexNet과는 약간 다른 전처리를 거치는데, 트레이닝 셋 전체의 RGB 채널 평균 값을 입력 이미지의 각 픽셀마다 substract하여 입력을 zero-centerd되게 한다.

기존의 conv-net들은 대체적으로 큰 필터(커널)을 사용하였다. 예를 들어 AlexNet은 11x11, s=4인 필터를 사용하였고 ZFNet은 7x7, s=2인 필터를 사용했다.<br>
하지만 VGG는 기존의 네트워크들과는 다르게 모든 레이어에서 가장 작은 크기의 필터를 사용한다(3x3, s=1 p=1). 이 때 3x3 필터로 구성된 레이어 2개를 쌓으면 5x5와 동일한 성능을, 7x7은 레이어 3개를 쌓으면 같은 성능을 보인다.<br>
그럼 VGGNet은 왜 3x3 필터를 사용하는 걸까? 논문에 의하면 작은 크기의 필터를 사용하면 두가지 장점이 있다고 한다.

**여러 개의 ReLU non-linear를 사용할 수 있다.** 큰 필터로 구성된 하나의 레이어를 작은 필터의 여러 레이어로 나누었기 때문에 ReLU non-linearlity가 들어갈 곳이 더 많아진다. 이는 decision function이 더 discriminative하다는 의미와 같다.

**학습해야할 weight의 수가 많이 줄어든다.** 비교를 위해 3x3 conv 레이어 3 개와 7x7 conv 레이어 1 개가 있다고 가정하자. 이 때 두 레이어 모두 C개의 채널을 가진다.<br>
두 레이어의 parameter의 개수를 계산해보자. 7x7 conv 레이어의 parameter의 개수는 $7^2 C^2 = 49C^2$이다. 3개의 3x3 conv 레이어는 $3*(3^2 C^2) = 27C^2$ 이다. 당연하게도 학습해야할 parameter의 개수가 적다면 regularization 측면에서 큰 이점을 보인다.

다른 실험을 위해 논문에서는 모델 C 네트워크에 1x1 conv 레이어도 사용하였다. 1x1 크기의 필터는 입력을 그대로 출력에 보내는 듯 하지만 그렇진 않다. 1x1 크기의 필터는 채널들끼리 dot product을 해서 채널들을 linear transformation 하는 역할을 한다. 이 때 입력과 출력의 채널은 같게 설정한다. 네트워크에 1x1 conv 레이어를 쌓으면 ReLU가 들어갈 공간이 많아져 더 discriminative한 네트워크가 된다. 

VGGNetwork의 추가적인 특징은 다음과 같다.

- 2x2 Max-pool (s=2)인 레이어를 pooling 레이어로 사용하였다.
- fc는 4096-4096-1000이며, 마지막 레이어는 softmax를 사용한다.
- 모든 레이어는 ReLU non-linearlity를 사용한다.
- AlexNet에서 사용한 LRN은 사용하지 않는다. 실험 결과 성능향상은 잘 안되지만 메모리와 컴퓨팅 시간을 많이 잡아먹어서 사용하지 않는다고 한다.

### Details of Learning
대부분의 경우 AlexNet과 비슷하다.

- SGD를 사용, mini-batch 사이즈는 256
- 모멘텀은 0.9, Weight decay는 $5*10^{-4}$
- fc 2개 레이어에 dropout (p = 0.5)
- learning rate는 AlexNet과 같다. 실험 결과 트레이닝 단계에서 3번 정도 감소 했다고 한다.

weight 초기화는 AlexNet과 많이 다르다. 단순히 gaussian 분포를 통한 랜덤 초기화가 아니라 상대적으로 작은 네트워크 A를 AlexNet과 같은 초기화 한 후 트레이닝 한다. 그리고 트레이닝 된 작은 네트워크를 기반으로 다른 큰 네트워크를 fine-tuning 하는 기법을 사용한다.<br>
이 때 첫 4개의 cov 레이어와 마지막 3개의 fc의 weight를 pre-train된 네트워크의 weigh로 사용하였으며, 나머지 레이어의 weight는 AlexNet과 동일하게 랜덤 초기화 하였다 (단, 바이어스는 0으로 초기화).

### Data augmentation
VGGNet도 AlexNet이 시행한 두가지 augmentation 방법을 기본적으로 따른다. 224x224의 입력 이미지를 만들기 위해 원본 이미지에서 224x224의 이미지 패치를 랜덤하게 잘라내(crop) 만들며 상하 반전, RGB 컬러 sift등의 방법은 같다. 하지만 두 네트워크의 큰 차이점은 VGGNet은추가적으로 image-rescaling을 한다는 것이다.

#### Image Rescaling
원본 이미지를 잘라내기 전에 먼저 이미지를 가로, 세로가 원본과 동일한 비가 되도록 scaling(isotropically-rescaled) 하고 이 re-scale된 이미지에서 학습할 224x224 크기의 이미지들을 추출한다.<br>
이 때 re-scale된 이미지의 크기(w, h중 작은 값)을 $S$ 라고 하자. 이 때 $S$는 224보다 크거나 같아야 한다는 조건을 만족해야만 한다. 만약 $S=224$ 라면 scale된 이미지 전체를 입력 이미지로 사용한다는 의미이며, $S > 224$라면 scale된 이미지의 일부분만 입력 이미지로 사용한다는 뜻이다 (물체의 일부분만 입력 이미지로 들어간다).

이 때 $S$를 설정하는 방법은 두가지가 있다.

- Single-scale learning
- Multi-scale learning

**Single-scale learning**은 $S$값을 고정된 크기로 정하는 것이다. 본 논문에서는 $S=256, 384$ 두가지 값으로 설정하였다. 네트워크를 훈련 시킬 때 먼저 $S=256$ 크기의 이미지를 통해 일차적으로 훈련하고 $S=384$ 크기의 네트워크를 학습시킬 때는 학습 속도를 올리기 위해 $S=256$인 네트워크의 초기값을 이용하고 이 때 더 작은 learning rate($10^{-3}$)을 사용하여 학습을 진행한다.

**Multi-scale learning**은 $S$ 값을 $[S _{min}, S _{max}]$ 사이에서 랜덤하게 선택하고 이 값을 통해 이미지를 scale한다. VGGNet은 $S _{min} = 256, S _{max} = 512$를 사용한다.<br>
이 방법을 사용하면 원본 이미지에 있는 물체는 랜덤한 크기를 가질 수 있으므로 더 다양한 입력 variation을 줄 수 있다 (scale jittering).<br>
네트워크의 훈련시 Single-scale learning과 비슷하게 먼저 $S=384$인 Single-scale learning 네트워크를 학습 시키고, 학습된 네트워크의 초기값을 적용하여 fine-tuning 한다.

### 참고 사항
<div class="imgcap">
<img src="/assets/Casestudy-CNN/vgg-arch3.png">
</div>

위 그림은 VGG-16의 Architecture, 요구 메모리, parameter 개수를 나타낸 것이다. (참고 - [CS231n](cs231n.stanford.edu/syllabus.html))<br>
VGGNet은 하나의 이미지당 fowardprop 단계에서 약 93MB정도의 메모리가 필요하며, backprop단계를 합하면 거의 200MB정도의 메모리가 필요하다. 또한 총 parameter의 개수는 약 138M 정도 이다.

한가지 눈여겨 볼점은 대부분의 메모리는 앞쪽 conv 레이어에서 발생하고, 대부분의 parameter는 fc 레이어에서 증가한다는 것이다.

---

## Network In Network (2014)
Network In Network (NIN)는 VGGNet, AlexNet과 같이 ILSVRC에서 우수한 성능을 입증한 네트워크는 아니다 (논문에서도 CIFAR-100, MNIST, SVHN과 같은 데이터셋에서만 성능을 검증하였다). 그럼에도 불구하고 NIN을 소개하려는 이유는 NIN에서 사용한 방법론들이 GoogLeNet에서 사용되었기 때문이다.

### Basic Concept of NIN
일반적인 CNN은 conv 레이어와 pooling 레이어로 구성된다. 이 때 conv 레이어는 각 필터의 수용 영역(receptive field)에 선형 필터를 dot product 시키고 (convolution), ReLU와 같은 non-linearlity를 거쳐서 출력값을 만들어 낸다. 이 때 CNN의 conv 필터를 Generalized linear model (GLM)이라고 부른다고 한다. 이 논문에서는 만약 GLM을 다른 nonlinear 함수로 바꾼다면 모델의 성능을 더 향상 시킬 수 있을 것이라 주장하고, **mlpconv 레이어 구조**를 제시하였다.

<div class="imgcap">
<img src="/assets/Casestudy-CNN/nin-arch1.png">
</div>

위 그림에서 왼쪽은 일반적인 CNN에서 사용하는 선형 컨볼루션 레이어이고 오른쪽은 논문에서 제시한 mlpconv 레이어이다. 두 레이어 모두 입력 레이어에서 local한 receptive 영역을 이용하여 출력 벡터를 만들어낸다. 하지만 mlpconv 레이어는 local한 patch에서 fc 레이어와 non-linearlity로 구성된 MLP 구조를 거친 후 출력 벡터를 만든다는 차이를 보인다. feature map을 만들 때 일반적인 레이어와 같이 필터의 window를 sliding한 다는 점은 동일하다. 결론으로 NIN은 이 mlpconv 레이어를 여러층으로 쌓은 네트워크라고 보면 되겠다.

NIN은 mlpconv 이외에 **(global) average pooling** 방식을 사용한다. 기존의 CNN에서는 conv 레이어들을 거친 후 마지막에 classification을 위해 몇개의 fc 레이어들을 섞어서 사용한다. 이 방식은 VGGNet에 의하면 fc 레이어에서 급격하게 parameter의 수가 늘어나는 문제가 생기며, NIN은 fc 레이어의 문제점을 average pooling를 이용하여 해결하였다.

### Details of NIN
<div class="imgcap">
<img src="/assets/Casestudy-CNN/nin-arch2.png">
</div>

위 그림은 NIN 구조의 전체 모습이다. 구성된 NIN은 3개의 mlpconv 레이어와, 한 개의 average pooling 레이어로 구성된 것을 볼 수 있다.

### MLP Convolution Layers
만약 분포에 대해 미리 아는 정보가 없다면, feature들을 추출할 때 universal function을 사용하는게 나을 것이다. Radial basis 네트워크나 MLP는 잘 알려진 universal approximator인데 논문에서는 이 중 MLP를 다음과 같은 장점 때문에 MLP를 이용하였다.

- MLP는 back-prop을 통하여 훈련시킬 수 있기 때문에 CNN 구조와 호환이 잘 된다.
- MLP는 그 자체로 깊은 구조가 될 수 있다.

때문에 기존의 GLM 구조를 MLP로 대체한 mlpconv 레이어를 제안하였다. 이 때 mlpconv 레이어는 아래와 같은 방식으로 공식화 할 수 있다.

<div class="imgcap">
<img src="/assets/Casestudy-CNN/nin-formula.png">
</div>

이 때 $n$은 MLP에서 레이어의 개수를 의미한다. $max(..)$에서 알 수 있듯이 non-linearlity는 ReLU를 사용하였다.

위 공식은 일반적인 conv 레이어에 Cascaded Cross Channel Pooling (CCCP)라는 방법을 적용한 것과 같다. 각 pooling 레이어는 입력된 feature map을 weighted linear combination한 뒤 이를 ReLU에 전달하는 것이고, 그 결과값을 다음 레이어에 전파시킨다. 이 때 CCCP 방법은 1x1 conv 레이어와 동일하다. 다시 말해 만약 하나의 mlpconv 레이어가 3개의 레이어를 가진 MLP 레이어로 구성된다면, 이는 conv 레이어에서 나온 결과값을 1x1 conv 레이어와 ReLU 레이어에 전파시키는 과정을 3번 반복한 것과 같다는 의미이다. (아래 그림 참조)

<div class="imgcap">
<img src="/assets/Casestudy-CNN/CCCP.png">
</div>

mlpconv 레이어를 사용하면 local한 범위에서 더 좋은 abstraction을 할 수 있다는 장점이 있다.

### Average Pooling
위에서도 언급했지만 전통적인 CNN은 낮은 레이어(입력 이미지와 가까운 레이어)는 conv 레이어를 사용하고, classification을 위해 마지막 conv 레이어의 출력값을 vectorize 하여 fc 레이어에 전달하는 방식을 사용한다. 이는 feature의 추출은 conv 레이어를 이용하여 진행하고, 추출된 feature들을 전통적인 방식을 이용하여 classify 하는 것으로 볼 수 있다.

이 방식의 문제는 fc 레이어는 overfit 문제에 매우 취약하며 네트워크의 일반화 성능에 악영향을 미칠 수 있다. 그래서 dropout과 같은 방법들이 제시되어 overfit 문제를 어느정도 해결하기도 하였다. 하지만 논문에서는 dropout과 같이 fc 레이어의 overfit 문제를 회피하는 방식이 아닌, fc 레이어를 average pooling 레이어로 대체하는 전략을 제시하였다.

<div class="imgcap">
<img src="/assets/Casestudy-CNN/average_pooling.png">
</div>

average pooling은 마지막 mlpconv 레이어에서 classification 해야 할 수만큼 feature map을 만드는 것에서부터 시작한다. 그 후 각 map의 average를 계산하여 결과 값을 바로 softmax 레이어에 전달한다. 예를 들어 데이터셋이 CIFAR-10이라면, 마지막 mlpconv 레이어에서 10개의 feature map을 추출하고, 각 map을 average하여 추출한 벡터값을 10개의 softmax 뉴런에 one-to-one으로 전달하는 것이다.

이 방식을 사용하면 다음과 같은 이점이 있다고 한다.

- fc 레이어보다 훨씬 자연스럽다. mlpconv 레이어를 이용하여 feature들을 추출한 뒤, 이 추출된 feature들의 map을 각각 종합하여 classification에 사용하기 때문에 feature들과 category간의 관계를 강화하는 역할을 한다. 반면에 fc 레이어는 블랙박스라고 일컬어질만큼 내부에서 어떤 일이 일어나는지 알기 어렵다.
- parameter에 대해 걱정할 필요가 없다. fc 레이어는 매우매우 많은 parameter 때문에 overfit 문제가 항상 도사리고 있지만, average pooling은 parameter가 없다.
- average pooling은 spatial한 정보를 합하는 방식이기 때문에 입력 이미지의 spatial 변환에 robust 하다고 한다.

<div class="imgcap">
<img src="/assets/Casestudy-CNN/nin-exp.png">
</div>

average pooling은 regularizer의 역할을 한다고 볼 수도 있다. 위 그래프는 CIFAR-10 데이터를 이용하여 generalization error를 나타낸 것이다. 실험 결과를 통해 average pooling이 어느정도 regularizer의 역할을 한다고 볼 수 있을 것이다.

논문에서는 전통적인 CNN 구조에 fc 레이어 대신 ap 레이어를 사용한 실험도 진행하였는데, 이 때는 fc with dropout > ap > fc without dropout 순으로 성능이 좋았다.

### NIN Structure
NIN은 mlpconv 레이어를 3층 쌓고, 마지막에 average pooling 레이어를 한 층 쌓아서 만든 네트워크이다. 이 때 전통적인 네트워크들 처럼 subsample을 위한 pooling 레이어를 mlpconv 레이어 사이에 추가할 수도 있다. 다른 방법들은 AlexNet에서 사용한 것과 대부분 유사하기 때문에 모두 언급하지는 않을 예정이다.

### Conclusion
NIN은 1x1 필터를 이용한 mlpconv 레이어에 의해 **더 나은 local abstraction**을 보인다. 또한 average pooling을 사용하여 **parameter의 수를 매우 줄일 수** 있으며 이는 **overfit 문제를 어느정도 해결** 한다는 장점이 있다.

하지만 조금 의문인점은 VGGNet의 실험 결과에서 conv1 레이어를 사용한 모델 C 네트워크와 conv3 레이어를 사용한 모델 D 네트워크를 비교했을 때 모델 D가 더 좋은 성능을 보인 것으로 나타났다. 물론 모델 B와 C를 비교하면 성능향상이 일어났다고 주장할 수 있지만, 너무 깊은 레이어가 아닌 이상에야 conv1이 아닌 conv3을 사용하는게 더 낫지 않을까 싶다.

---

## GoogLeNet (2014)

---

## PReLU-net (2015)

### Parametric Rectifiers
<div class="imgcap">
<img src="/assets/Casestudy-CNN/prelu1.png">
</div>

PReLU-net은 기존에 많이 사용되던 ReLU 함수를 변형한 Parametric Rectifiers (PReLU)를 사용한다. PReLU는 LReLU와 비슷하지만 $y < 0$일 때의 기울기를 학습을 통해 조정해 나가는 점에서 약간의 차이를 보인다. 참고로 논문에 의하면 LReLU는 ReLU와 성능면에서 큰 차이를 보이지 않는다고 한다. ([Neural Network](/NN) 참조)

$$f(y _i) = 
\begin{cases}
	y _i & \text{if $y _i > 0$} \\\
	a _i y _i & \text{if $y _i < 0$}
\end{cases}
$$

PReLU를 사용한다면 약간의 parameter가 증가한다는 단점이 있다. 만약 channel-wise PReLU를 사용하면 channel개수만큼 parameter가 증가하며, channel-shared는 레이어 개수만큼 증가한다. 이는 전체 parameter 개수에 비하면 무시해도 될 만큼의 증가이기 때문에 overfit에 대한 우려는 하지 않아도 된다.

**channel-wise PReLU**는 channel 마다 다른 기울기 parameter를 가지며, **channel-shared PReLU**는 channel끼리 parameter를 공유하는 것이다.

### Optimization
PReLU의 parameter $a$는 back-prop을 이용하여 학습이 가능하며, 다른 weight들과 같이 체인룰에 의해 gradient를 구할 수 있다. 아래 공식은 channel-wise PReLU의 gradient를 구하는 공식이다.

$$\frac{\partial \varepsilon}{\partial a _i} = \sum _{y _i}\frac{\partial \varepsilon}{\partial f(y _i)}\frac{\partial f(y _i)}{\partial a _i}$$

그리고 $a_i$에 의한 gradient는 다음과 같다.

$$\frac{\partial f(y _i)}{\partial a _i} = 
\begin{cases}
	0 & \text{if $y _i > 0$} \\\
	y _i & \text{if $y _i < 0$}
\end{cases}$$

channel-shared PReLU의 gradient를 구하는 공식은 아래와 같다.

$$\frac{\partial \varepsilon}{\partial a} = \sum _{i}\sum _{y _i}\frac{\partial \varepsilon}{\partial f(y _i)}\frac{\partial f(y _i)}{\partial a}$$

$a _i$는 0.25로 초기화하고, 값의 업데이트는 모멘텀 방식을 사용하지만 다른 weight들의 업데이트와는 달리 weight decay는 사용하지 않는다. 이는 weight decay가 $a _i$의 값을 0 으로 만드는 경향이 있어 ReLU와 비슷하게 되어버리기 때문이며, regularization 없이도 $a _i$는 1 이상으로 증가하지는 않는다. 재밌는 점으로 값의 범위를 제한하지 않아서 학습에 의해 non-monotonic한 함수로 될 가능성도 존재한다는 점이다.

### Comparison Result
<div class="imgcap">
<img src="/assets/Casestudy-CNN/prelu-exp1.png" style="max-width:49%;">
<img src="/assets/Casestudy-CNN/prelu-exp2.png" style="max-width:49%;">
</div>

논문에서는 위와 같은 네트워크의 조건 하에 ReLU, channel-wise PReLU, channel-shared PReLU를 비교하는 실험을 진행하였다.<br>
PReLU가, 그 중에서도 channel-wise PReLU가 매우 우수한 성능을 보임을 확인 할 수 있다. 네트워크 표에서 함수의 coefficient를 기록해 놓은 것을 보면 흥미로운 점을 발견 할 수 있다.

- conv1의 coefficient는 0보다 매우 크다. 대체적으로 conv1은 엣지나 텍스쳐를 detect하는 Gabor-like적인 특성을 띄고 있기 때문에 postive와 negative한 경우 모두 반응을 보인 것을 볼 수 있다.
- channel-wise의 경우 레이어가 깊어질 수록 coefficient의 값이 감소하는 경향을 보인다. 이는 레이어가 깊어질 수록 activation이 non-linear해지는 의미이다. 다시 말해 초기 레이어에서는 더 많은 정보를 가지고 있으려 하지만, 레이어가 깊어질수록 분별에 초점을 맞추려는 경향을 보인다.

## Initalization of Filter
ReLU를 사용하는 네트워크는 sigmoid-like 네트워크보다 weight들을 학습시키기 용이하다. 하지만 여전히 초기화를 제대로 하지 않으면 학습이 잘 되지 않을 가능성이 존재 한다. PReLU-net은 깊은 rectifier 네트워크를 학습시킬 때의 문제점을 해결 할 수 있는 robust한 초기화 방법을 제안하였다.

[Neural network](/NN)에서도 한번 언급한 내용이지만 weight 초기화에 대해 다시 한번 짚고 넘어가도록 하자.<br>
최근까지 CNN에서 weight의 초기화는 가우시안 분포를 이용하는 경우가 많았다. 하지만 고정된 표준편차를 사용한다면 (예를 들어 0.01) 8 레이어 이상의 매우 깊은 네트워크에서는 제대로 수렴이 되지 않는 문제가 발생한다. 문제를 해결 하기 위해서 VGGNet은 작은 네트워크에서 먼저 학습시키고, 학습된 weight를 큰 네트워크에 초기값으로 전달하는 방식을 취했지만 이 전략은 학습 시간이 증가하고 local optima에 빠질 수 있다는 문제점이 있다. GoogLeNet에서는 메인이 되는 softmax classifier 이외에 보조적인 classifier를 두어 수렴하는데 도움 되도록 네트워크를 구성하기도 하였다.

[Glorot et al., 2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) 는 Xavier 초기화라는 방법을 제안하였다. 이 방법은 뉴런의 수에 따라 값을 적절하게 스케일링 한다는 특징을 가진다. 하지만 Xavier 초기화는 activation이 linear하다는 가정을 가지고 만들어졌기 때문에 ReLU-like를 사용하는 네트워크에서는 잘 작동하지 않는다는 단점이 있다.

그래서 PReLU-net은 Xavier 초기화를 변형하여 ReLU-like 네트워크에서 잘 작동이 되도록 초기화 방법을 제안했으며, 30 레이어 이상의 매우 깊은 네트워크에서도 잘 수렴이 된다고 한다.

작성중..

---

## ResNet (2015)
MSRA에서 만든 ResNet은 2015년 ImageNet의 Classification, Detection, Localization 부문에서 모두 1위를 차지했으며, 매우매우 깊은 레이어를 자랑하는 네트워크이다. 레이어의 수에 관한 설명은 아래 슬라이드 그림으로 대체하겠다...

<div class="imgcap">
<img src="/assets/Casestudy-CNN/deeeep.png">
</div>

ILSVRC의 winning 네트워크들의 추세를 봐도 알수 있는 사실이지만 네트워크의 레이어를 층층이 쌓아서 깊게 구현하면 더 좋은 성능을 낸다. 하지만 레이어를 깊게 쌓는 것이 항상 좋은 결과를 낼 까? 네트워크를 깊게 쌓으면 gradient vanishing/exploding 현상이 발생할 수 있기 때문에 네트워크는 학습의 초기 단계부터 saturated되어 버릴 우려가 있다. 하지만 이 문제는 BN, Xavier 초기화(PReLU-net 참조) 등을 이용하면 수십개의 레이어까지는 해결이 된 상태이다.

하지만 네트워크가 더 깊어지면 **degradation** 이라 불리는 문제가 발생한다. 네트워크는 깊어지는데 정확도는 saturated 되는 현상이다. 사실 이는 overfit을 생각하면 당연하다고 생각 할 수 있지만 놀랍게도 degradation은 overfit에 의한 것이 아닌 애초에 **트레이닝 에러 자체가 높아지는** 현상이다. 아래 그림은 degradation의 예시를 보여준다.

<div class="imgcap">
<img src="/assets/Casestudy-CNN/degradation1.png">
</div>

그래프의 왼쪽은 트레이닝 에러, 오른쪽은 테스트 에러를 나타낸다. 두 네트워크는 레이어의 갯수만 달리한 일반적인 네트워크이다. 깊은 네트워크가 테스트 에러 뿐만 아니라 트레이닝 에러까지 높은 것을 확인 할 수 있다. 이 현상은 CIFAR-10에서만 일어나는 문제가 아닌 다양한 데이터셋에서도 공통적으로 일어나는 현상이다. ImageNet 데이터셋에서의 결과를 나타낸 아래 그래프에서도 깊은 네트워크가 더 나쁜 성능을 보인다.

<div class="imgcap">
<img src="/assets/Casestudy-CNN/degradation2.png">
</div>

이에 대한 해결책으로 VGGNet에서 학습 시킨 방법과 같이 비교적 얕은 네트워크를 이용하여 일차적으로 학습시키고 이를 깊은 네트워크의 초기값으로 주어 학습시키는 방법을 사용할 수 있지만, 항상 최적이지도 않고 시간도 오래 걸리는 단점이 있다. 반면 ResNet은 degradation 문제를 **deep residual learning** 이라는 학습법으로 해결하여 무려 152개의 레이어를 쌓은 깊은 네트워크를 만들 수 있게 되었다.

### Deep Residual Learning

<div class="imgcap">
<img src="/assets/Casestudy-CNN/plain.png" style="max-width:49%; height:250px;">
<img src="/assets/Casestudy-CNN/residual.png" style="max-width:49%; height:250px;">
</div>

일반적인(Plain) 네트워크는 위와 같은 레이어 구조를 가진다. 이 때 두 레이어를 거친 후 매핑된 결과는 $H(x)$로 표현하며 아래와 같이 표현 할 수도 있다.

$$ H(x) = F(x, \lbrace W_i \rbrace) $$

여기서 위 네트워크는 2개의 레이어를 가지고 있기 때문에

$$ F = W _2 \sigma (W _1 \boldsymbol{x}) $$

이다. $x$는 입력 벡터이며 $\sigma$는 ReLU activation을 의미한다. 식을 간단히 쓰기 위해서 바이어스는 생략하였다.

residual 네트워크는 일반적인 네트워크와는 달리 몇개의 레이어 (여기에서는 2개의 레이어)를 건너 뛴 shortcut을 활용한 것이 특징이다. $H(x)$는 $H(x) = F + x$ 으로 표현할 수 있다.


- AlexNet 처럼 data augementation
- BN (no dropout)
- Xavier Init

작성중..

---

### 레퍼런스
[ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)<br>
[Network In Network](http://arxiv.org/abs/1312.4400)<br>
[Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/abs/1409.1556)<br>
[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852)<br>
[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)<br>
[Deep Residual Learning ICCV15 slides](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)
