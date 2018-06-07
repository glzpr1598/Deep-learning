# Learning rate

### learning rate가 너무 큰 경우

꼭지점에 다다르지 못하는 overshooting 현상이 발생한다.

![](img\lec07-1.PNG)

### learning rate가 너무 작은 경우

너무 조금씩 움직여서 꼭지점에 다다르지 못한다. 또한 아주 작은 굴곡이 있으면 그 지점을 꼭지점으로 판단할 수 있다.



#  Preprocessing(선처리)

x1 데이터에 비해 x2 데이터의 변화량이 클 경우, gradient descent 과정이 정상적으로 일어나지 않을 수 있다. 이 경우, 데이터를 선처리해야 한다.

![](img\lec07-2.PNG)

### Standardization

nomalization의 한 종류이다.
$$
x'_j=\frac{x_j-\mu_j}{\sigma_j}
$$
tensorflow에서는

```python
x_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
```



# Overfitting

머신 러닝의 가장 큰 문제점.
학습 데이터에 너무  딱 맞게 모델링을 한다.
예외 데이터가 있는 경우 선형에서 많이 벗어난 모델을 만든다.

![](img\lec07-3.PNG)

위 그림에서 왼쪽이 일반적인 모델이지만, 오른쪽은 너무 주어진 데이터에만 맞춰져있다.(overfitting)



## 해결방법

- 더 많은 데이터를 학습시킨다.
- 데이터의 features의 개수를 줄인다.
- Regularzation(일반화)



## Regulariztion

구부러진 선을 펴지도록 한다. -> weight의 수를 줄인다.

![](img\lec07-4.PNG)

regulariztion strength가 0이면 일반화를 하지 않는것이고, 커질수록 일반화를 많이 하는 것이다.

![](img\lec07-5.jpeg)

tensorflow에서는,

```python
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
```



# Training set, testing set

데이터를 모두 학습시키면 테스트를 할 수 없기 때문에, 학습시키는 데이터(training set)와, 테스트를 하기 위한 데이터(testing set)를 나누는 것이 좋다.

![](img\lec07-6.PNG)

Validation은 상수를 정하기 위한 데이터(learning rate, regulariztion strength 등)



# Online learning

데이터가 많은 경우, 데이터를 나눠서 학습시키는 방법.
첫 데이터를 학습하여 모델링을 한 뒤에, 데이터를 추가로 학습시키면 처음부터 새로 모델링을 하는 것이 아닌 추가한 데이터만 학습하여 모델링을 하는 방법.

