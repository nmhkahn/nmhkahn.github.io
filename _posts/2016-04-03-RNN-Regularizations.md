---
layout: "post"
title: "RNN Regularizations"
excerpt: "Regularization methods used in RNN"
date: "2016-04-03 06:00:00"
---

Feedfoward neural net은 dropout, BN등과 같이 강력한 regularizer 기법들을 적용해서 overfit을 해결하곤 한다. 하지만 RNN은 feedfoward net에서 사용한 regularizer를 그대로 쓰게되면 생기는 여러 문제점들로 인해서 아직까지 이 분야에 대한 발달이 더딘편이기도 하고, 정리도 잘 되어있지 않고 있다. 그래서 이번 포스트에서 최근 나온 몇편의 논문들을 소개해서 RNN에서 어떤식으로 regularizer들이 적용될 수 있는지에 대한 정리를 하고자 한다.

참고로 논문들의 Introduction과 Related work에 공통적으로 등장하는 RNN이나 LSTM에 관한 자세한 설명은 생략하도록 하겠다.

## Recurrent Neural Network Regularization
---
이 논문은 dropout을 RNN에 적용시키면 잘 동작하지 않는(성능이 떨어지는) 현상에 대한 해결방안을 제시하였다.

### RNN and LSTM
논문에서는 먼저 RNN과 LSTM에 대한 수학적인 정의를 한 뒤 자신들의 방법들을 제시하였는데 일단 포스트의 첫 논문이니 이 방식을 그대로 따라가 보도록 하겠다.

그 전 notation을 정리하자. 모든 state는 n차원이고 $h _t^l \in \mathbb{R}$는 timestep $t$에서 $l$번째 레이어의 hidden state를 의미한다. 또한 이 논문에서는 편의성을 위해 $ T _{n,m} : \mathbb{R^n} \rightarrow \mathbb{R^m}$ 라고 affine transform을 정의하였다. 쉽게 말해서 $Wx + b$를 의미하는 것이라고 보면 된다.
그럼 RNN은 $h _t^{l-1}, h _{t-1}^l \rightarrow h _t^l$ 의 상태 전이 함수로 표현가능하며, vanilla RNN의 $h _t^l$ 은 다음과 같다.

$$
h _t^l = \sigma(T _{n,n}h _t^{l-1} + T _{n,n}h _{t-1}^l)
$$

LSTM은 gradient vanishing문제를 해결하기 위해 (exploding 문제는 별도로 clipping 기법을 활용하여 해결 가능하다) 사용되는 RNN의 변형버전으로 아래와 같은 수식을 통해 표현된다.

<div>
\begin{aligned}
\text{LSTM: } h _t^{l-1}, h _{t-1}^l, c _{t-1}^l \rightarrow h _t^l, c _t^l \\

\begin{pmatrix}
i \\ f \\ o \\ g \\
\end{pmatrix} = 

\begin{pmatrix}
sigm \\ sigm \\ sigm \\ tanh \\
\end{pmatrix} T _{2n, 4n} 

\begin{pmatrix}
h _t^{l-1} \\ h _{t-1}^l \\
\end{pmatrix} \\
\end{aligned}

\begin{aligned}
c _t^l &= f \odot c _{t-1}^l + i \odot g \\
h _t^l &= o \odot tanh(c _t^l)
\end{aligned}
</div>

### LSTM with Dropout
논문에서 제시한 방법 dropout을 non-recurrent한 연결에만 적용하는 것이다. 
<div class="imgcap">
<img src="/assets/RNN-Reg/p1-dropout.png" style="max-height:400px">
</div>
위 그림에서 점선은 dropout이 적용된 연결이고 실선은 dropout이 적용이 되지 않은 것이다. 제시한 방법을 곱씹어보면 이전 timestep에서 온 정보는 dropout 하지 말고, **현재 timestep에서 들어온 입력값 혹은 이전 레이어의 값만 dropout** 하는 의미이다. LSTM의 수식으로 표현해보면 다음과 같다.

<div>
\begin{aligned}
\text{LSTM: } h _t^{l-1}, h _{t-1}^l, c _{t-1}^l \rightarrow h _t^l, c _t^l \\

\begin{pmatrix}
i \\ f \\ o \\ g \\
\end{pmatrix} = 

\begin{pmatrix}
sigm \\ sigm \\ sigm \\ tanh \\
\end{pmatrix} T _{2n, 4n} 

\begin{pmatrix}
\boldsymbol{D}(h _t^{l-1}) \\ h _{t-1}^l \\
\end{pmatrix} \\
\end{aligned}

\begin{aligned}
c _t^l &= f \odot c _{t-1}^l + i \odot g \\
h _t^l &= o \odot tanh(c _t^l)
\end{aligned}
</div>

현재 timestep에서 이전 레이어인 $h _t^{l-1}$ 에 dropout을 적용했는데 과거 timestep의 정보를 지우지 않게 하기 위해 recurrent한 연결 (과거 timestep)에 dropout을 적용하지 않은 것으로 보인다. 다시 정리하면, 일반적인 dropout을 적용한다면 먼 과거의 정보를 잃어버려 학습하는데 어려움을 갖지만, **non-recurrent한 연결만 dropout을 적용하면 과거 중요 정보를 희생하지 않아도 regularization을 사용**할 수 있다.

### Experiments
먼저 Penn Tree Bank(PTB) 데이터셋에서 단어 단위의 prediction을 진행하였다. PTB는 929k개의 학습 단어, 73k개의 validation 단어와 82k개의 테스트 단어로 이루어져있으며, vocabulary에는 10k의 단어가 있다. 비교를 위해서 논문은 medium LSTM, large LSTM 나누었는데 두 LSTM은 모두 두 개의 레이어를 35 step만큼 펼쳐놓은 네트워크이다. 또한 hidden state는 모두 0으로 초기화했으며, 현재 minibatch의 마지막 hidden state는 다음 minibatch에서 첫 hidden state가 되게 하였다 (minibatch 크기는 20).
<div class="imgcap">
<img src="/assets/RNN-Reg/p1-ptb.png" style="max-height:400px">
</div>
medium LSTM은 레이어 당 650개의 neuron으로 구성되어 있으며 weight들은 [-0.05,0.05] 범위에서 uniform하게 초기화 했다. 또한 non-recurrent한 연결에는 50%의 확률로 dropout을 적용했다. LSTM의 학습은 39번 epoch만큼 진행했는데 초기 learning rate는 1으로, 6 epoch이후 매 epoch마다 decay를 0.8만큼 적용하였다. 추가로 gradient exploding을 방지하기 위해 gradient의 norm(minibatch 크기로 normalize한 값)이 5보다 크다면 clip하였다.

large LSTMd은 각 레이어가 1500개의 neuron을 가지고 있고, weight는 [-0.04,0.04]의 범위에서 초기화 하였다. dropout의 확률은 65%로 meidum보다는 조금 크게 설정하였고 초기 learning rate는 1, decay는 14 epoch이후 매 epoch마다 1/1.15만큼 적용하며 총 55 epoch를 학습하였다. clipping은 값이 10보다 작을 때 진행하였다고 한다.

비교를 위해서 non-regularized 네트워크도 학습시켰으며, cv를 통해 최적의 네트워크를 찾으려고 했으나 overfit문제 때문에 상대적으로 작은 네트워크를 구성할수 밖에 없었다고 한다. 레이어마다 200개의 neuron에 20 step을 펼쳐놓은 형태이며, 초기화는 [-0.1,0.1], 학습은 4 epoch 이후 0.5로 decay했으며 총 20 epoch를 학습하였다.
<div class="imgcap">
<img src="/assets/RNN-Reg/p1-speech.png" style="max-height:400px">
</div>
Icelandic 음성 데이터셋으로 비교한 음성 인식의 경우 non-regularized LSTM보다 성능이 좋은 것을 확인 할 수 있다. 참고로 학습 데이터의 accuracy는 더 낮은데 이는 학습시 dropout에 의해 noise를 같이 학습하기 때문에 학습 데이터에는 덜 정확하지만 generalization이 더 좋아져서 validation에 대해서는 좋은 accuracy를 보이는 것으로 볼 수 있다.
<div class="imgcap">
<img src="/assets/RNN-Reg/p1-translation.png" style="max-height:400px">
</div>
영어에서 프랑스어로 번역한 기계 번역은 regularized LSTM이 좋은 성능을 보이는 것은 맞지만, phrase-based LIUM SMT (Schwenk et al., 2011) 보다는 좋지 않은 성능을 보인 것으로 나타났다.
<div class="imgcap">
<img src="/assets/RNN-Reg/p1-captioning.png" style="max-height:400px">
</div>
[Vinyals et al., 2014] 에서 제시된 캡션 생성 모델에 dropout을 적용한 결과이다. dropout을 한 single model이 하지 않은 ensemble 모델과 비슷한 성능을 내는 것으로 나타났다.

## RNNDROP: A Novel Dropout for RNNs In ASR
---
CNN에서 사용하던 dropout 방식을 RNN에 그대로 사용하면 각 hidden node 들은 매우 자주($p = 0.5$라면 2 timestep마다) 정보를 잊어버리게 되기 때문에 과거의 메모리들이 쉽게 없어질 가능성이 높다. 잦은 메모리의 리셋은 RNN 계열의 모델에서 가장 큰 무기를 잃어버리는 것과 마찬가지이기 때문에 오히려 좋지 못한 성능이 나오게 되는 결과를 초래하게 된다. 논문에서도 deep한 양방향 LSTM을 위에서 서술한 방식으로 dropout을 적용하니 성능이 매우 나빠지는 것을 관측 할 수 있었다고 한다.

### RNNDROP
이 문제를 해결하기 위해 논문은 RNNDROP 이라는 방법을 제안하였다.<br>
<div class="imgcap">
<img src="/assets/RNN-Reg/dropout.png" style="max-height:400px">
</div>
기존의 dropout 방식은 매 timestep 마다 랜덤한 dropout 마스크를 씌우는 것이라고 볼 수 있다.<br>
이 때 검은색 노드는 dropout에 의해 제거된 노드들이고 점선은 제거된 노드들에 의해 나오는 weight들을 나타낸다. 그림에서는 timestep이 달라지면 dropout되는 노드들도 달라진다.

<div class="imgcap">
<img src="/assets/RNN-Reg/rnndrop.png" style="max-height:400px">
</div>
반면 RNNDROP은 학습 시퀀스의 맨 처음에 dropout 마스크를 생성하고 그 뒤 시퀀스들은 첫 시퀀스에서 생성된 마스크를 고정시켜서 사용하는 것이다. 그림은 두 개의 시퀀스에서 각각 RNNDROP을 적용시킨 것인데 (t-1, t, t+1) timestep 모두 같은 dropout 마스크를 사용하는 것을 볼 수 있다.<br>
RNNDROP은 dropout을 시퀀스 레벨에서 사용하는 것으로도 볼 수 있으며 하나의 학습 시퀀스에서 하나의 dropout 마스크를 사용하기 때문에 DNN에서 하나의 학습 이미지에서 dropout을 사용하는 것과 비슷한 방식이다. 

수식으로 RNNDROP을 표현하면 LSTM에서 $c$는 아래와 같이 표현된다

$$
c _t \leftarrow m _u \odot c _t
$$

이 때, $m$는 Bernoulli(p) 분포에 의해 독립적으로 생성된 dropout 마스크 벡터를 의미하고 $\odot$은 element-wise 곱을 의미한다. 참고로 $m$은 위에서 서술한 것과 마찬가지로 학습 예제 하나 당 한번 생성되며, timestep $t$에 종속되지 않는다.<br>
BPTT를 이용해서 RNN을 학습할 때는 제거된 노드에서 나오는 gradient를 0으로 치환하는 방식으로 구현할 수 있으며, 학습이 끝난 후에는 기존 dropout과 비슷하게 $1-p$ 값을 곱해서 rescale 한다 (사실 논문이나 구현마다 $p$를 dropout 하는 확률로 하는 경우도 있고 dropout 하지 않는 확률로 하는 경우도 있어서 헷갈리는데 아마 논문에서는 dropout 하는 확률을 $p$라고 둔 것 같다).<br>
논문에서는 실험을 양방향 LSTM에서 진행하였고 (DBLSTM) 양방향 LSTM의 경우 dropout 마스크를 양쪽에서 생성하는 방식을 사용하여 구현한다.

### Experiments
논문에서는 DBLSTM 음성 모델인 TIMIT 음소 인식 데이터와 Wall Street Journal 음성 인식 데이터를 이용하여 평가를 하였다.

TIMIT 실험은 [[A. Graves,. 2013]](http://www.cs.toronto.edu/~graves/asru_2013.pdf) 논문과의 비교를 위해 RNNDROP을 적용한 점을 제외하면 동일한 세팅을 이용해서 실험하였다고 한다 (관심 있으면 이 논문을 참고해보자...).
<div class="imgcap">
<img src="/assets/RNN-Reg/TIMIT1.png" style="max-height:400px">
</div>
이 데이터셋으로 실험된 기존의 연구 결과들을 종합하면 위 표와 같다. dropout을 non-recurrent한 부분만 적용시킨 DBLSTM의 경우 dropout을 사용하지 않고 weight noise를 준 경우보다 성능이 좋지 못하다.
<div class="imgcap">
<img src="/assets/RNN-Reg/TIMIT2.png" style="max-height:400px">
</div>
반면 RNNDROP을 적용한 경우 평균 16.92%로 기존 state-of-the-art보다 약 6% 정도 성능이 향상된 것을 확인 할 수 있다. DBLSTM을 5개 레이어에 각 레이어마다 500 메모리 블록을 사용하도록 디자인 하면 16.29%로 조금 더 성능이 향상된다고 한다.

WSJ 실험의 경우 DNN으로 적용한 모델보다 약 10% 정도 WER이 향상된 결과를 보인다고 한다.
<div class="imgcap">
<img src="/assets/RNN-Reg/WSJ.png" style="max-height:400px">
</div>

## Batch Normalized Recurrent Neural Networks
---
Neural Net을 학습할 때 weight의 초기화를 "잘" 하거나, 정규화를 하면 모델의 수렴 속도를 향상시켜 더 빨리 모델을 학습시킬 수 있다. 하지만 whitening과 같은 방법은 계산량이 많기 때문에 Batch Normalization과 같은 방법을 사용하여 학습 속도도 줄이고 더불어 regularization를 사용한 효과를 내기 때문에 최근의 CNN 모델들은 거의 대부분 이 BN을 사용한다.<br>
논문에서는 정규화 방법을 RNN에 적용하여 학습 속도를 감소시킬 수 있음을 보인다. 

### Batch Normalization
이번 포스트에서는 BN이나 LSTM을 다루는 글이 아니기 때문에 논문의 제안 방법을 설명하기 위한 정도로만 그치려고 한다. 대신 BN의 경우 수식의 이해가 조금 필요한 부분이 있어 이 부분은 약간의 수식을 덧붙여서 설명할 예정이다.

Minibatch $X$가 주어지면 sample mean과 variance는 다음과 같이 구할 수 있다 (이 때 $k$는 feature를 의미한다).

<div>
\begin{aligned}
\bar{\boldsymbol{X} _k} &= \frac{1}{m} \sum _{i=1}^m \boldsymbol{X} _{i, k} \\
\sigma _{k,t}^2 &= \frac{1}{m} \sum _{i=1}^m (\boldsymbol{X} _{i, k} - \bar{\boldsymbol{X} _k})^2
\end{aligned}
</div>

여기서 $m$은 minibatch 크기이며, 구한 mean과 variance를 가지고 normalize를 하면 다음과 같다.

$$
\hat{\boldsymbol{X}} = \frac{\boldsymbol{X} _{k,t} - \bar{\boldsymbol{X}} _{k,t}}{\sqrt{\sigma _{k,t}^2 + \epsilon}}
$$

$\epsilon$은 numerical stability를 위해 더해주는 매우 작은 값이다.

...

### Batch Normalization for RNNs
먼저 BN을 Feedfoward network에 적용시킨 것처럼 RNN에 적용시키면 다음과 같은 식으로 적용할 수 있을 것이다.

$$
\boldsymbol{h} _t = \phi(BN(\boldsymbol{W} _h \boldsymbol{h} _{t-1} + \boldsymbol{W} _x \boldsymbol{x} _t))
$$

논문에서는 PennTreeBank 데이터셋을 이용하여 실험을 진행하였으며, 네트워크는 250-D의 룩업 테이블, 각 250개의 hidden unit을 가지는 3개의 레이어로 구성되어 있고 50-D의 softmax 레이어를 맨 위에 추가한 형태이다.<br>
BN은 위 공식의 형태로 적용시켰는데 이는 매 timestep마다 mean과 variance를 250 features를 이용해서 구한다는 의미이다. 추론 단계에서도 매 timestep 마다 정보를 저장했지만, $\gamma$와 $\beta$ 는 timestep에 관계없이 같은 값을 사용하였다.

기타 세부적인 구현사항은 다음과 같다.

- 룩업 테이블은 unit gaussian으로 초기화 하였으며, 다른 weight들은 Glorot이 제시한 초기화 방법을 사용하였다 (Xavier init).
- SGD+momentum을 사용했으며 learning rate는 [0.0001, 1] 사이를 random search를 이용해서 성능 측정을 했고 momentum은 [0.5, 0.8, 0.9, 0.95, 0.995]를, 배치 크기는 [32, 64, 128]을 이용했다. 실험은 20 epoch동안 진행하고 전체 실험은 52번 반복한다.

모든 실험에서 BN을 사용한 경우가 베이스 RNN 모델보다 같거나 낮은 성능을 보였으며, learning rate가 높을 때는 좀 예외로 베이스라인은 발산하지만 BN 모델은 계속 학습할 수 있었다고 한다.
<div class="imgcap">
<img src="/assets/RNN-Reg/BN1.png" style="max-height:400px">
</div>
그래프를 관찰해도 BN을 적용한 경우 성능이 좋지 못한 것을 확인 할 수 있는데, 논문에서는 아마 매 timestep마다 새로운 statistics를 적용시키거나 혹은 같은 $\gamma$, $\beta$를 사용해서 gradient exploding이나 vanishing이 일어나지 않을까 하는 추측을 한다.

그래서 위 방법 대신에 논문에서는 input-to-hidden 노드에만 BN을 적용시키는 방법을 제안하였다.

$$
\boldsymbol{h} _t = \phi(\boldsymbol{W} _h \boldsymbol{h} _{t-1} + BN(\boldsymbol{W} _x \boldsymbol{x} _t))
$$

사실 이 아이디어는 Recurrent Neural Network Regularization 사용한 것과 비슷한 아이디어인데, 이 논문에서는 dropout 대신 BN을 적용시켰다는 점이 조금 다를뿐이다.

### Frame-wise and Sequence-wise Normalization
char-rnn와 같은 경우 미래의 frame (timestep)을 참고할 필요 없이 다음 문자를 예측하는게 목표이기 때문에 매 timestep마다 normalization을 다음과 같이 계산 할 수 있다.

$$
\hat{\boldsymbol{X}} = \frac{\boldsymbol{X} _{k,t} - \bar{\boldsymbol{X}} _{k,t}}{\sqrt{\sigma _{k,t}^2 + \epsilon}}
$$

논문에서는 이 방식을 frame-wise normalization이라 부른다.

반면 음성 인식과 같은 문제에서는 전체 시퀀스를 참고해야 하는 경우가 대부분이다. 또한 시퀀스들은 다양한 길이를 가지고 있기 때문에 minibatch를 사용하면 대부분의 경우 가장 긴 시퀀스에 맞추기 위해 0으로 패딩을 넣는 작업을 한다. 이 방법때문에 frame-wise를 사용하면 패딩되지 않는 frame이 감소하면 statistics estimate가 매우 나쁘기 때문에 다른 방법을 사용해야만 한다. 따라서 이 경우에는 mean과 variance를 시간과 batch 두 축을 이용해서 계산한다.

<div>
\begin{aligned}
\bar{\boldsymbol{X} _k} &= \frac{1}{n} \sum _{i=1}^m \sum _{t=1}^T \boldsymbol{X} _{i, t, k} \\
\sigma _{k,t}^2 &= \frac{1}{n} \sum _{i=1}^m \sum _{t=1}^T (\boldsymbol{X} _{i, t, k} - \bar{\boldsymbol{X} _k})^2
\end{aligned}
</div>

이 때 $T$는 각 시퀀스의 길이, $n$은 minibatch에서 unpadded frame의 총 개수를 의미하고, 이 방식을 sequence-wise normalization이라 한다.


### Experiments
Speech task는 WSJ 데이터셋을 활용하였다. BL 모델은 각 250개의 unit을 가진 5개의 양방향 LSTM 레이어를 쌓은 구조에 3546-D의 softmax 출력 레이어를 가지고 있다. 모든 weight들은 Xavier 초기화 방법을 사용하였고 biase들은 0으로 초기화 하였다. BN 모델은 BL 모델에 sequence-wise 정규화 방법을 사용한 구조이다. 두 네트워크 모두 SGD + momentum를 사용하였으며, 1e-4의 learning rate와 0.9의 mu 값을 사용하고 minibatch 크기는 24로 고정하였다.

언어 모델링은 PTB 데이터셋을 이용해서 실험을 진행하였는데, 네트워크의 구조는 Recurrent Neural Network Regularization와 동일하다 (자세한 설명은 서술하지 않겠다).

<div class="imgcap">
<img src="/assets/RNN-Reg/BN2.png" style="max-height:400px">
</div>

<div class="imgcap">
<img src="/assets/RNN-Reg/BN3.png" style="max-height:400px">
</div>

위 그래프는 speech task에서 BL과 BN 네트워크를 비교한 것이다. BN 모델이 더 빨리 수렴하지만 더 overfit 하다. 표는 [A. Graves,. 2013] 논문의 방식과 비교한 것으로 가장 좋은 성능만 추려서 표기한 것이다.

<div class="imgcap">
<img src="/assets/RNN-Reg/BN4.png" style="max-height:400px">
</div>

<div class="imgcap">
<img src="/assets/RNN-Reg/BN5.png" style="max-height:400px">
</div>

PTB 데이터셋에 대해서도 비슷한 결과를 (더 overfit하지만) 관측 할 수 있다.

두 실험 모두 BN 모델이 학습은 더 빨리 되지만 overfit이 심해지는 것을 공통적으로 확인 할 수 있다. 대신 overfit 현상은 speech 실험에서 조금 덜한데 아마 이는 학습 데이터셋이 크기 때문일 수도 있고 혹은 frame-wise 정규화가 덜 효과적일 수도 있다는 의견을 논문에서는 내고 있다. 또한 언어 모델링은 한 번에 하나의 문자만 예측하지만 speech는 전체 시퀀스를 예측해야하는 실험 특성상 이런 결과가 나온 것이 아닌가 하고 생각한다.

BN은 feedfoward network에서는 높은 learning rate에서도 잘 동작하지만 논문에서는 BN을 input-to-hidden 연결부위에만 적용시켰다. 그래서 높은 learning rate를 사용하게 되면 normalize되지 않은 부분 때문에 잘 동작하지 않는다고 한다.

## 레퍼런스
[Recurrent Neural Network Regularization](http://arxiv.org/abs/1409.2329)<br>
[RNNDROP: A Novel Dropout for RNNs In ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)<br>
[Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378)
