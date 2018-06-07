# Hypothesis

변수가 1개인 경우,
$$
H(x) = Wx + b
$$
변수가 여러개인 경우,
$$
H(x_1, x_2, x_3) = 
w_1x_1 +
w_2x_2 +
w_3x_3 +
b
$$



# Cost function

$$
cost(W, b) = 
\frac{1}{m} 
\sum_{i=1}^m 
(W(x_{1i}, x_{2i}, x_{3i})-y_i)^2
$$



# Matrix

$$
w_1x_1 + w_2x_2 + w_3x_3 + ... +  w_nx_n
$$

$$
\begin{pmatrix}
x_1 & x_2 & x_3
\end{pmatrix}
\begin{pmatrix}
w_1 \\
w_2 \\
w_3 \\
\end{pmatrix} =
(x_1w_1 + x_2w_2 + x_3w_3)
$$

$$
H(X) = XW
$$
instance(데이터)가 많은 경우 행렬을 이용하면 훨씬 편하다.
$$
\begin{pmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
x_{31} & x_{32} & x_{33} \\
x_{41} & x_{42} & x_{43} \\
x_{51} & x_{52} & x_{53} \\
\end{pmatrix}
\begin{pmatrix}
w_1 \\
w_2 \\
w_3 \\
\end{pmatrix} =
\begin{pmatrix}
x_{11}w_1 + x_{12}w_2 + x_{13}w_3 \\
x_{21}w_1 + x_{22}w_2 + x_{23}w_3 \\
x_{31}w_1 + x_{32}w_2 + x_{33}w_3 \\
x_{41}w_1 + x_{42}w_2 + x_{43}w_3 \\
x_{51}w_1 + x_{52}w_2 + x_{53}w_3 \\
\end{pmatrix}
$$
### W(weight)의 크기 결정

##### Linear regression의 경우 출력이 하나이다. (H(x)의 열이 하나)

   X   *   W   =   H(X)
(5, 3)   (?, ?)     (5, 1)

-> W = (3, 1)

##### 출력이 2개인 경우

(n, 3) * (?, ?) = (n, 2)

-> W = (3, 2)



# 정리

### 이론
$$
H(x) = Wx + b
$$
### 구현(TensorFlow)
$$
H(X)=XW
$$

