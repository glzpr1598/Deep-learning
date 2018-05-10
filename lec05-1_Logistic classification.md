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
