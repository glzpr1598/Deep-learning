# Linear regression(선형 회귀 분석)

주어진 데이터를 선형 관계로 모델링 하는 회기 분석 기법



# Hypothesis(가설)

$$
H(x) = Wx + b
$$

W : Weight
b : bias



# Cost function

$$
cost(W, b) = 
\frac{1}{m} 
\sum_{i=1}^m 
(H(x_i)-y_i)^2
$$

(H(x) - 실제값) 제곱의 평균



# 목표

Cost function을 최소화하는 W, b의 값을 찾는 것