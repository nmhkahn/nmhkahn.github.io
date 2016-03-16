---
layout: "post"
title: "RNN Regularization"
excerpt: "Regularization method used in RNN (LSTM)"
date: "2016-03-15 10:00:00"
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
i \\
f \\
o \\
g \\
\end{pmatrix} = 
\begin{pmatrix}
sigm \\
sigm \\
sigm \\
tanh \\
\end{pmatrix} T _{2n, 4n} 
\begin{pmatrix}
h _t^{l-1} \\
h _{t-1}^l \\
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
\text{LSTM: } h _t^{l-1}, h _{t-1}^l, c _{t-1}^l \rightarrow h _t^l, c _t^l \\\
\begin{pmatrix}
i \\
f \\
o \\
g \\
\end{pmatrix} = 
\begin{pmatrix}
sigm \\
sigm \\
sigm \\
tanh \\
\end{pmatrix} W _{2n, 4n} 
\begin{pmatrix}
\boldsymbol{D}(h _t^{l-1}) \\
h _{t-1}^l \\
\end{pmatrix} \\
\end{aligned}

\begin{aligned}
c _t^l &= f \odot c _{t-1}^l + i \odot g \\
h _t^l &= o \odot tanh(c _t^l)
\end{aligned}
</div>

현재 timestep에서 이전 레이어인 $h _t^{l-1}$ 에 dropout을 적용했는데 과거 timestep의 정보를 지우지 않게 하기 위해 recurrent한 연결 (과거 timestep)에 dropout을 적용하지 않은 것으로 보인다. 다시 정리하면, 일반적인 dropout을 적용한다면 먼 과거의 정보를 잃어버려 학습하는데 어려움을 갖지만, **non-recurrent한 연결만 dropout을 적용하면 과거 중요 정보를 희생하지 않아도 regularization을 사용**할 수 있다.

### Experiments
논문에서는 언어 모델링, 음성 인식, 기계 번역과 이미지 캡션 생성 분야에 대해 실험을 진행하였다.<br>




## 레퍼런스
[Recurrent Neural Network Regularization](http://arxiv.org/abs/1409.2329)<br>
[RNNDROP: A Novel Dropout for RNNs In ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)<br>
[Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378)
