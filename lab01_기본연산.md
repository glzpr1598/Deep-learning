```python
import tensorflow as tf
```




```python
# 버전 확인
tf.__version__
```

'1.8.0'




```python
# Hello 출력
# constant : 그래프에 노드를 추가하는 연산자
hello = tf.constant("Hello, Tensorflow!")
sess = tf.Session()
print(sess.run(hello))
```

b'Hello, Tensorflow!'



```python
# Computational graph
```


```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # 암묵적으로 tf.float32
node3 = tf.add(node1, node2)
```


```python
print("node1:", node1, "node2:", node2) # ,로 구분 시 공백 자동 삽입
print("node3:", node3)
```

node1: Tensor("Const_1:0", shape=(), dtype=float32) node2: Tensor("Const_2:0", shape=(), dtype=float32)
node3: Tensor("Add:0", shape=(), dtype=float32)



```python
sess = tf.Session()
print("sess.run([node1, node2]):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))
```

sess.run([node1, node2]): [3.0, 4.0]
sess.run(node3): 7.0



```python
# Placeholder
# 미지수로 놓고 값을 넣고 싶을 때 사용
```


```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # tf.add(a, b)의 축약형

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
```

7.5
[3. 7.]



```python
# Ranks, Shapes, and Types

# Rank : 몇 차원 배열인지
# Shape : 배열에 몇 개가 들어있는 지
'''
Rank Shape
0    [] 
1    [D0]
2    [D0, D1]
'''

# Type
'''
tf.float32 *많이 사용
tf.float64
tf.int8
tf.int16
tf.int32 *많이 사용
tf.int64
'''
```




```python
3 # rank 0, shape []
[1, 2, 3] # rank 1, shape[3]
[[1, 2, 3], [4, 5, 6]] # rank 2, shape [2, 3]
[[[1, 2, 3]], [[7, 8, 9]]] # rank 3, shape [2, 1, 3]
```

[[[1, 2, 3]], [[7, 8, 9]]]


