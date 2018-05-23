

```python
#### cross_entropy, one_hot, reshape ####

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

# 16개의 데이터를 통해 동물의 종(0 ~ 6)을 나타내는 데이터
xy = np.loadtxt('data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)
# (101, 16) (101, 1)

nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6, shape = (?, 1)

# Y를 one hot 형태로 변환. rank가 1 늘어남.
Y_one_hot = tf.one_hot(Y, nb_classes)  # shape = (?, 1, 7)
print("one_hot :", Y_one_hot)

# rank를 다시 줄여줌. 우리가 원하는 형태.
# [[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], ...]
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # shape = (?, 7)
print("reshape :", Y_one_hot)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)  # softmax

# Cross entropy cost
# 기존 코드 
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# 를 함수로 만들어 놓은 것.
# hypothesis가 아닌 logits을 인자로 받음!!
cost_i = tf.nn.softmax_corss_entropy_with_logits(logits=logits, 
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 예측값
prediction = tf.argmax(hypothesis, 1)
correct_prediction = 
```

    (101, 16) (101, 1)
    one_hot : Tensor("one_hot_3:0", shape=(?, 1, 7), dtype=float32)
    reshape : Tensor("Reshape_2:0", shape=(?, 7), dtype=float32)
    
