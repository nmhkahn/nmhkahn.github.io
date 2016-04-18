---
layout: "post"
title: "Some Special Distributions"
excerpt: "수리통계 수업때 배운 확률 분포 정리"
date: "2016-04-17 10:00:00"
---

## Contents
- Bernoulli Distribution
- Binomial Distribution
- Geometric Distribution
- Negative binomial Distribution
- Multinomial Distribution
- Hypergeometric Distribution
- Poisson Distribution

## Bernoulli Distribution
베르누이 분포는 이산확률분포로 성공/실패 두 가지 상황만 나오는 경우에 사용하는 분포이며, 성공/실패 경우와 같이 나올 수 있는 경우의 수는 mutually exclusive해야 한다.<br>
랜덤 변수 $X$ 가 베르누이 시행이라 가정하면 $X$ 를 다음과 같이 정의할 수 있다.

$$
X(\text{성공}) = 1 \quad \text{and} \quad X(\text{실패}) = 0
$$
 
수학적 편의성을 위해서 성공은 1, 실패는 0 으로 정의했으며 이 시행에서 이산 확률 분포(pmf) 는 아래와 같다.

$$
p(x) = p^x (1-p)^{1-x}, \\ x = 0, 1
$$

이 때 우리는 pmf $p(x)$ 를 베르누이 분포라고 부른다. 그리고 베르누의 분포에서 $X$ 의 기대값은

<div>
\begin{aligned}
\mu = E(X) &= (0)(1-p) + (1)(p) \\
		   &= p
\end{aligned}
</div>

으로, 분산은

<div>
\begin{aligned}
\sigma^2 = Var(X) &= p^2(1-p) + (1-p)^2p \\ 
			      &= p(1-p)
\end{aligned}
</div>

으로 증명 가능하다. 사실 평균과 분산은 위와 같이 정의를 이용해서 구할 수도 있고 mgf를 이용해서 구할 수도 있는데 일단 베르누이 분포의 mgf는 아래와 같다.

<div>
\begin{aligned}
M(t) = E(e^{tx}) &= \sum _{x=0,1} e^{tx} p^x (1-p)^{1-x} \\
			     &= pe^t + (1-p)
\end{aligned}
</div>

mgf를 이용해 평균과 분산을 구하는 방법은 1차 모멘트가 평균, 2차 모멘트를 이용해서 분산을 구할 수 있으니 알아서 해보시라 (..).

## Binomial Distribution
이항 분포는 서로 iid인 베르누이 분포 $X _i$ 를 $n$ 번 시행했을 때의 분포를 의미한다. 다시 말하면 베르누이 분포는 성공/실패 두 경우만 나오는 실험을 한 번만 진행한 것이고 이항 분포는 위 실험을 여러 번 진행한 것과 같다. (ex. 동전을 10번 던졌을 때 앞면이 4번 나올 확률)

$$
p(x) = \binom{n}{x}p^x(1-p)^{n-x} \\ , x = 0, 1, 2, ... , n
$$

이항 분포의 pmf는 위와 같다. pmf인지 따져보기 위해 pmf의 합이 1이 되는지는 이항 정리를 이용하여 증명 할 수 있다.

$$
(a+b)^n = \sum \binom{n}{x} a^{n-x}b^x
$$

이므로 $a = 1-p, b = p$ 로 두면 $\sum p(x) = 1$ 임을 쉽게 증명할 수 있다. 그리고 비슷하게 이항 분포의 mgf도 이항 정리를 이용해 구하면 된다.

<div>
\begin{aligned}
M(t) &= E(e^{tx}) \\
	 &= \sum^n _{x=0} e^{tx} \binom{n}{x}p^x (1-p)^{n-x} \\
	 &= \sum \binom{n}{x} (pe^t)^x (1-p)^{n-x} \quad \text{(이항정리)} \\
	 &= \{(1-p) + (pe^t)\}^n \\
	 &= (pe^t + q)^n \ , q = (1-p)
\end{aligned}
</div>

아무튼 mgf를 이용하면 평균과 분산을 구할 수 있고 아래와 같다.

$$
E(x) = np, \quad Var(x) = np(1-p)
$$

이항 분포에 대한 또다른 성질로 $X _1 \sim B(n _1, p _1), \\ X _2 \sim B(n _2, p _2)$ 일 경우, $X _1 + X _2 \sim B(n _1 + n _2, p _1 + p _2)$ 와 같다는 점이다. 이 성질은 $X _1, X _2$가 서로 독립이기 때문에 두 분포를 더한 분포의 mgf는 각 분포의 mgf의 합과 같다는 성질을 이용하면 증명 할 수 있다.

## Geometric Distribution
기하 분포는 $x$ 번째 시행에서 첫번째 성공이 일어날 확률을 나타내는 분포이다. 다시 말하면 $x-1$ 번째 까지 계속 실패하다 $x$ 번째에서 성공하는 확률이라 보아도 무방하다. pmf는

$$
p(x) = p(1-p)^{x-1}
$$

이며, $W \sim Geo(p)$ 일 때 $mgf _W (t) = (1-qe^t)^{-1}e^tp, \\ t < -log(q)$ 와 같고 이를 이용해

$$
E(W) = \frac{1}{p}, \quad Var(W) = \frac{q}{p^2}
$$

임을 유도 할 수 있다. 참고로 기하 분포에서 $x = 1, 2, ...$ 부터 시작하는데 이항 분포는 성공할 확률이 0, 1, ... 이지만 기하 분포는 k 번째 시행일 때 처음 성공하는 경우를 나타내기 때문에 1 부터 시작하는 것이다.

## Negative Binomial Distribution
음이항 분포는 $x$ 번째 시행에서 $r$ 번째 성공할 확률을 나타내고, 기하 분포를 일반화시킨 분포라고 생각하면 된다. pmf는

$$
p(x) = \binom{x-1}{r-1}p^{r-1}(1-p)^{(x-1)-(r-1)}p
$$

이며, $x$ 번째 시행에서 $r$ 번째 성공할 확률이므로 이 말은 $x-1$ 시행까지 $r-1$ 번 성공한다는 의미와 같다. 때문에 위 식에서 $p$ 텀을 제외한 식은 이와 같으며, 마지막 $p$ 는 $x$ 번째 시행에서 성공할 확률을 뜻한다. 음이항 분포를 나타낼 때는 $X \sim NB(r, p)$ 와 같이 나타낸다.

만약 $X \sim NB(r, p)$ 면, $M(t) = [pe^t(1-qe^t)^{-1}]^r, \\ t < -log(q), q = 1-p$ 이며

$$
E(X) = \frac{r}{p}, \quad Var(X) = \frac{rq}{p^2}
$$

와 같다. 또한 기하 분포와 비슷하게 $X _1 + X _2 \sim N(r _1 + r _2, p)$ 이다. 이는 기하 분포와 마찬가지로 mgf를 이용해서 증명이 가능하고 평균은

$$
E(X) = E(z _1 + z _2 + ... + z _r) = rE(z) = r\frac{1}{p}
$$

로, 분산은 $Var(X) = rVar(z)$ 임을 이용해 평균과 비슷하게 유도 할 수 있다.
