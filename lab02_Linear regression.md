

```python
import tensorflow as tf

#### 그래프 구현 ####
# x, y 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# W, b 지정
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Variable : tensorflow가 학습하면서 변경하는 값

# 함수 정의
hypothesis = W * x_train + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_maen : 평균

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


#### 그래프 실행 ####
# 세션 생성
sess = tf.Session()
# Vaiable을 사용하기 위해서는 Variable을 초기화 해야함
sess.run(tf.global_variables_initializer())

for step in range(2001): # range() 숫자 리스트를 만들어주는 함수
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
```

    0 8.790969 [0.103765] [-1.0807647]
    100 0.029329404 [1.1988969] [-0.45216367]
    200 0.01812378 [1.156358] [-0.3554387]
    300 0.011199392 [1.1229117] [-0.2794072]
    400 0.0069205486 [1.0966198] [-0.21963955]
    500 0.0042764884 [1.075952] [-0.17265685]
    600 0.002642607 [1.059705] [-0.13572405]
    700 0.0016329704 [1.0469337] [-0.1066914]
    800 0.0010090781 [1.0368942] [-0.08386929]
    900 0.0006235482 [1.0290022] [-0.06592878]
    1000 0.00038531589 [1.0227984] [-0.0518261]
    1100 0.00023809828 [1.0179214] [-0.04073979]
    1200 0.00014712977 [1.0140879] [-0.0320251]
    1300 9.0917434e-05 [1.0110744] [-0.02517482]
    1400 5.6181943e-05 [1.0087055] [-0.01978973]
    1500 3.4716806e-05 [1.0068434] [-0.01555652]
    1600 2.1452855e-05 [1.0053796] [-0.0122289]
    1700 1.3257205e-05 [1.0042288] [-0.00961312]
    1800 8.192062e-06 [1.0033244] [-0.00755688]
    1900 5.062591e-06 [1.0026133] [-0.00594048]
    2000 3.1284533e-06 [1.0020542] [-0.00466991]



```python
# placeholder 이용

import tensorflow as tf

#### 그래프 구현 ####
# x, y 데이터(placeholder)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# W, b 지정
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# Variable : tensorflow가 학습하면서 변경하는 값

# 함수 정의
hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# reduce_maen : 평균

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


#### 그래프 실행 ####
# 세션 생성
sess = tf.Session()
# Variable을 사용하기 위해서는 Variable을 초기화 해야함
sess.run(tf.global_variables_initializer())

for step in range(2001): # range() 숫자 리스트를 만들어주는 함수
    # sess.run을 실행한 결과를 변수에 넣어서 바로 출력 가능
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [2.1, 3.1, 4.1]})
    if step % 100 == 0:
        print(step, cost_val, W_val, b_val)
```

    0 66.522514 [-2.177921] [0.68312865]
    100 0.060190637 [0.7157184] [1.7461791]
    200 0.03719416 [0.776546] [1.6079633]
    300 0.022983735 [0.8243449] [1.4993052]
    400 0.014202554 [0.8619191] [1.4138904]
    500 0.008776331 [0.8914557] [1.3467466]
    600 0.005423231 [0.9146742] [1.2939655]
    700 0.0033512365 [0.93292624] [1.2524744]
    800 0.0020708574 [0.9472739] [1.2198588]
    900 0.0012796733 [0.9585525] [1.19422]
    1000 0.0007907629 [0.9674184] [1.1740656]
    1100 0.00048863835 [0.97438794] [1.1582221]
    1200 0.00030194796 [0.9798667] [1.1457679]
    1300 0.00018658699 [0.98417324] [1.1359777]
    1400 0.00011530064 [0.9875587] [1.1282818]
    1500 7.1247916e-05 [0.99022] [1.1222321]
    1600 4.4028915e-05 [0.9923119] [1.1174768]
    1700 2.7207703e-05 [0.99395645] [1.1137384]
    1800 1.6812211e-05 [0.9952492] [1.1107997]
    1900 1.0388664e-05 [0.99626553] [1.1084892]
    2000 6.4187793e-06 [0.9970645] [1.106673]



```python
# 테스트
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
```

    [6.0919952]
    [2.6022696 4.5963984]

