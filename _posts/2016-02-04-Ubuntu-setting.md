---
layout: "post"
title: "Linux Ubuntu 15.10 초기 설정"
excerpt: "Linux Ubuntu 15.10"
date: "2016-02-03 14:00:00"
---

## 한글 키보드 설정
Fcitx를 사용하였다.  
참고 - [우분투 14.04, 'Fcitx'로 한영 전환.](http://egloos.zum.com/nemonein/v/5229390)

## Python 패키지 설치
기본적인 python 패키지인 numpy, scipy, matplotlib을 맨땅에서 깔면 많은 오류가 발생하게 된다. 물론 apt-get을 이용하면 알아서 dependency를 잡아주니 편리하지만, 최신 버전을 지원하지 않을 수도 있고 가장 큰 문제는 virtualenv를 사용하지 못한 다는 것이다. 

그래서 pip을 통하여 파이썬 패키지들을 설치해보고자 한다.  
하지만 위 패키지들을 설치하기 전 먼저 apt-get을 이용하여 다음과 같은 denpendency들을 미리 설치해야 한다.  

**Numpy**

```shell
$ sudo apt-get install python-dev
```

**Scipy**

```shell
$ sudo apt-get install gfortran libatlas-base-dev
```

**Matplotlib**

```shell
$ sudo apt-get install libfreetype6-dev libpng-dev ligjpeg8-dev
```

위 dependency들을 설치하고, pip을 통해 numpy, scipy, matplotlib를 깔면 된다.

## Jekyll 설치
다음과 같이 설치한다.  
(참고 - [운영체제별 지킬 설치와 사용](http://vjinn.github.io/install-jekyll))

```shell
$ sudo apt-get install curl
$ gpg --keyserver hkp://keys.gnupg.net --recv-keys D39DC0E3
$ curl -sSL https://get.rvm.io | bash -s stable
$ source ~/.rvm/scripts/rvm
$ rvm install ruby
$ sudo apt-get install nodejs
$ gem install jekyll
```

내 경우는 rvm 으로 ruby를 깔면서 이 stackoverflow의 질문자와 비슷한 오류가 났었다. - [Ruby RVM apt-get update error](http://stackoverflow.com/questions/23650992/ruby-rvm-apt-get-update-error)


구글링 해본 결과 RVM이 apt-get의 repo를 긁어오는데 이 때 404 에러가 나는 repo들 때문에 오류가 생긴다는 것 같다. (리눅스에 잘 몰라서 맞는 설명인지는 잘 모르겠다.)  
해결법은 다음과 같다.

먼저 apt-get의 update를 해본다. 이 때 에러만 출력 하기 위해 grep을 사용한다

```shell
$ sudo apt-get update | grep Failed
```

```shell
Failed to fetch http://kr.archive.ubuntu.com/ubuntu/dists/wily-updates/main/binary-amd64/Packages  Hash Sum mismatch
Failed to fetch http://kr.archive.ubuntu.com/ubuntu/dists/wily-updates/universe/binary-amd64/Packages  Hash Sum mismatch
Failed to fetch http://kr.archive.ubuntu.com/ubuntu/dists/wily-updates/main/binary-i386/Packages  Hash Sum mismatch
Failed to fetch http://kr.archive.ubuntu.com/ubuntu/dists/wily-updates/universe/binary-i386/Packages  Hash Sum mismatch
Some index files failed to download. They have been ignored, or old ones used instead.
```

내 경우는 위와 같은 에러가 났었다. 이제 찾을 수 없는 repo를 삭제 해보자.  
repo 삭제는 /etc/apt/soures.list에서 가능하다.

```shell
$ sudo vi /etc/apt/sources.list
```

위 예시에서는 (잘은 모르겠지만..) wily-updates부분에서 문제가 생긴 것을 볼 수 있고, sources.list 파일에서 wily-updates가 있는 부분을 죄다 주석처리 해준다. (다른 저장소 문제일 수도 있다.)  

해결 완료. 다시 다음과 같이 입력하면 제대로 설치되는 것을 확인 할 수 있다.

```shell
$ rvm install ruby
```

참고로 jekyll을 설치하고 serve 하면 다음과 같은 오류가 날 수 있다.

```shell
Dependency Error: Yikes! It looks like you don't have jekyll-paginate or one of its dependencies installed. In order to use Jekyll as currently configured, you'll need to install this gem. The full error message from Ruby is: 'cannot load such file -- jekyll-paginate' If you run into trouble, you can find helpful resources at http://jekyllrb.com/help/! 
```

별 다른 오류는 아니고, config.yml에서 `gem: jekyll-paginate`와 같은 기타 denpendcy를 다음과 같이 설치해주면 된다.

```shell
$ gem install jekyll-paginate
```
