

```python
import tensorflow as tf

# 3번의 시험 점수와 최종 점수 데이터(5명)
x1_data = [73., 93., 90., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x1*w1 + x2*w2 + x3*w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "\nCost:", cost_val, "\nPrediction:", hy_val, "\n")
```

    0 
    Cost: 3356.8647 
    Prediction: [106.94714  118.45445  122.108444 132.09079   88.23723 ] 
    
    10 
    Cost: 24.438105 
    Prediction: [157.95248 179.79726 182.75632 197.89073 135.03453] 
    
    20 
    Cost: 24.28379 
    Prediction: [158.08583 179.99275 182.93346 198.0826  135.19069] 
    
    ...
    
    1980 
    Cost: 9.1149235 
    Prediction: [155.176   181.8727  182.3218  197.36577 137.70447] 
    
    1990 
    Cost: 9.070958 
    Prediction: [155.16467 181.87999 182.31941 197.36298 137.71423] 
    
    2000 
    Cost: 9.027144 
    Prediction: [155.1534  181.8873  182.31705 197.3602  137.72401] 


​    


```python
# Matrix 이용

import tensorflow as tf

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], 
          [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.,], [180.], [196.], [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "\nCost:", cost_val, "\nPrediction:\n", hy_val, "\n")
```

    0 
    Cost: 2891.175 
    Prediction:
     [[102.9861  ]
     [127.94099 ]
     [124.15584 ]
     [133.67604 ]
     [ 99.633316]] 
    
    10 
    Cost: 0.9687721 
    Prediction:
     [[150.54678]
     [185.0998 ]
     [180.47829]
     [195.00961]
     [143.22981]] 
    
    20 
    Cost: 0.9389981 
    Prediction:
     [[150.69362]
     [185.27072]
     [180.64957]
     [195.19601]
     [143.35895]] 
    
    ...
    
    1980 
    Cost: 0.53472686 
    Prediction:
     [[151.13693]
     [184.96422]
     [180.78139]
     [195.3252 ]
     [142.92819]] 
    
    1990 
    Cost: 0.53354025 
    Prediction:
     [[151.13864]
     [184.96306]
     [180.78188]
     [195.32573]
     [142.92651]] 
    
    2000 
    Cost: 0.53237635 
    Prediction:
     [[151.14032]
     [184.96188]
     [180.78238]
     [195.32625]
     [142.92484]] 


​    
