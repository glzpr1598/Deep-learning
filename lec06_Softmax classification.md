# Logistic regression

$$
H_L(X)=WX
$$

$$
z = H_L(X)
$$

$$
g(z) = \frac{1}{1+e^{-z}}
$$

$$
H_R(X) = g(H_L(X))
$$

이를 그림으로 나타내면

![](img\lec06-1.PNG)



# Multinomial classification

여러 등급으로 분류하는 방법

![](img\lec06-2.PNG)

위의 경우 logistic classification을 3번 하여 분류할 수 있다.

![](img\lec06-3.PNG)

![](img\lec06-4.PNG)

위의 결과를 softmax를 사용해 각각 0~1 사이이고 합이 1인 형태로 만든다.(확률)

![](img\lec06-5.PNG)

따라서 hypothesis는 XW + b를 softmax한 값이 된다.

결과를 예측하기 위해서는 'one-hot' 인코딩을 이용한다.
('one-hot' 인코딩 : 하나만 1로 표현하는 방법)

![](img\lec06-6.PNG)



# Cost function

![](img\lec06-7.PNG)

S는 예측값, L은 실제값

![](img\lec06-8.PNG)

즉,
$$
cost(W) =
-\frac{1}{m}\sum ylog(H(x))
$$
