# Hypothesis 단순화

$$
H(x) = Wx
$$

$$
cost(W) = 
\frac{1}{2m} 
\sum_{i=1}^m 
(W(x_i)-y_i)^2
$$

(미분을 편하게 하기 위해 2를 나눔)



# cost(W) 그래프

|  x   |  y   |
| :--: | :--: |
|  1   |  1   |
|  2   |  2   |
|  3   |  3   |

W = 1, cost(W) = 0
W = 0, cost(W) = 4.67
W = 2, cost(W) = 4.67 

W = 1을 중심으로 한 2차 함수 그래프가 됨.



# Gradient descent 알고리즘

기울기가 낮은 쪽을 따라가면서 목표점에 이르는 알고리즘
$$
cost(W) = 
\frac{1}{2m} 
\sum_{i=1}^m 
(W(x_i)-y_i)^2
$$

$$
W:=
W-
\alpha
\frac{\partial}{\partial W}
cost(W)
$$

$$
W:=
W-
\alpha
\frac{1}{m}
\sum_{i=1}^m 
(W(x_i)-y_i)x_i
$$

W - (a * 현재 기울기) 만큼 W를 움직이면서 이를 계속 반복(a는 상수)
기울기가 양수이면 왼쪽으로, 기울기가 음수이면 오른쪽으로 이동



# Convex function

![](img\lec03-1.png)

위의 cost function에서는 시작점에 따라서 목표지점이 달라지게 된다.

![](img\lec03-2.png))

위의 cost function에서는 시작점에 상관없이 목표지점이 항상 같다.
이를 convex function이라고 하는데, cost function을 정의할 때는 convex function이 되도록 정의해야 한다.