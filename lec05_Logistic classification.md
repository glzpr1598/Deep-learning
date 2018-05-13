# (Binary) Classification

둘 중 하나를 고르는 것.
ex) 이메일 spam(1) or ham(0), 페이스북 피드 show or hide, 신용카드 진짜(0) or 가짜(1)

기존의 Linear regression(H(x) = Wx + b)은 결과값이 0 또는 1인 데이터에는 적합하지 않다. 0에서 1로 바뀌는 지점이 x데이터 범위에 따라 바뀔 수 있고, H(x)가 1보다 커질 수도 있다.

![](img\lec05-1.png)

# Logistic hypothesis

결과값이 항상 0~1 사이인 함수가 필요

$$
g(z) = 
\frac{1}{1+e^{-z}}
$$
![](img\lec05-2.png)

이러한 형태의 함수를 logistic function 또는 sigmoid function이라 한다.

따라서, logistic hypothesis는
$$
H(X) = 
\frac{1}{1+e^{-W^TX}}
$$

# Cost function

Logistic hypothesis
$$
H(X) = 
\frac{1}{1+e^{-W^TX}}
$$
의 cost function을 기존 방식대로
$$
cost(W, b) = 
\frac{1}{m} 
\sum_{i=1}^m 
(H(x_i)-y_i)^2
$$
를 통해 구하면

![](img\lec05-3.png)

의 형태가 되어 cost function으로 적합하지가 않다.(시작점에 따라 목표점이 달라진다.)

따라서 logistic hypothesis에 적합한 cost function은
$$
cost(W) =
\frac{1}{m}\sum C(H(x),y)
$$

$$
C(H(x),y) = 
\begin{cases}
-log(H(x))  & \text{: y=1} \\
-log(1-H(x)) & \text{: y=0}
\end{cases}
$$

이다.

y=1인 경우, -log(H(x))의 그래프는

![](img\lec05-4.png)

의 형태가 되어 가설 H(x)도 1에 가까울수록 cost가 0에 가까워진다.
하지만, 가설이 잘못되어 H(x)가 0에 가까워지면 cost가 무한대로 발산한다.

반대로 y=0인 경우, -log(1-H(x))의 그래프는

![](img\lec05-5.png)

의 형태가 되어 가설 H(x)도 0에 가까울수록 cost가 0에 가까워진다.
하지만, 가설이 잘못되어 H(x)가 1에 가까워지면 cost가 무한대로 발산한다.

따라서, cost function은
$$
cost(W) =
\frac{1}{m}\sum C(H(x),y)
$$

$$
C(H(x),y) = 
\begin{cases}
-log(H(x))  & \text{: y=1} \\
-log(1-H(x)) & \text{: y=0}
\end{cases}
$$

가 되고 하나의 식으로 나타내면
$$
C(H(x),y)=ylog(H(x))-(1-y)log(1-H(x))
$$
결론적으로 cost function은
$$
cost(W)=
-\frac{1}{m}
\sum
ylog(H(x))-(1-y)log(1-H(x))
$$
가 되고, 이를 그래프로 그리면

![](img\lec05-6.png)

의 형태가 되어 우리가 원하는 cost function 모양이 된다.

# Gradient descent algorithm

$$
cost(W)=
-\frac{1}{m}
\sum
ylog(H(x))-(1-y)log(1-H(x))
$$

$$
W:=W-\alpha
\frac{\partial}{\partial W}cost(W)
$$
