## 1. Anaconda 설치

Anaconda : 데이터 과학에 필요한 패키지(Numpy, SciPy, IPython, Matplotlib, Jupyter 등)를 포함하는 플랫폼. 파이썬이 자동으로 설치됨.

다운로드 링크 : https://www.anaconda.com/download/



## 2. TensorFlow 설치

참조 : https://www.tensorflow.org/install/install_windows

1. 설치한 Anaconda Prompt를 관리자 권한으로 실행

    

2. tensorflow란 이름으로 conda 환경 생성

```
C:> conda create -n tensorflow pip python=3.5 
```



3. tensorflow 환경 활성화

```
C:> activate tensorflow
(tensorflow)C:>    # 앞에 (tensorflow)가 생김
```



4. TensorFlow 설치

- CPU 버전

```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow
```

- GPU 버전(지원 환경을 충족해야 사용 가능)

```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu
```



### 테스트

1. 파이썬 쉘 호출

```
C:> python
```

2. Hello, TensorFlow! 출력

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```



## 3. Jupyter Notebook에서 개발

1. Ananconda Prompt 실행
2. tensorflow 환경 활성화

```
C:> activate tensorflow
```

3. 원하는 디렉토리에서 Jupyter Notebook 실행

```
(tensorflow) C:> jupyter notebook
```

4. New - Python [conda env:tensorflow]
5. 코드 작성



### import tensorflow 에러 발생하는 경우

1. Ananconda Prompt에서 다음 명령 실행

```
(tensorflow)C:> conda install nb_conda
```

2. Python [conda evn:tensorflow] 로 파일 생성

