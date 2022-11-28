# <핸즈온 머신러닝 2판 - Chapter.5>

## [5.1] 서포트 벡터 머신(SVM) 
* 선형이나 비선형 분류, 회귀, 이상치 탐색에도 사용가능한 다목적 머신러닝 모델

* SVM의 기본 아이디어는 라지 마진 분류로 나타낸다.
    * 각 클래스 사이에 가장 폭이 넓은 도로를 찾는 것
    * 도로 경계에 위치한 샘플을 서포트 벡터라고 한다.

* SVM은 특성의 스케일에 민감하다. 따라서 사이킷런의 StandardScaler를 사용해보자

* 하드 마진 분류
    * 모든 샘플이 도로 바깥쪽에 올바르게 분류
    * 데이터가 선형적으로 구분될 수 있어야 작동함
    * 이상치에 민감하다.

![하드 마진](https://ifh.cc/g/Vgwc7O.png)

* 소프트 마진 분류
    * 하드 마진에서 좀 더 유연한 형태의 모델
    * 샘플이 도로 중간에 있거나, 반대쪽에 있는 경우인 마진 오류 사이에 적절한 균형을 잡아야한다.


* 사이킷런의 SVM 모델에서는 하이퍼 파라미터를 지정할 수 있는데, 그 중 C의 값을 조절하면 아래 그림과 같은 결과를 얻는다.

![마진](https://ifh.cc/g/v6FOxc.jpg)

* 왼쪽 모델의 경우 마진 오류가 많지만 일반화가 잘 됨

* 오른쪽 모델의 경우 마진 오류가 적지만 일반화가 왼쪽에 비해 부족함

* 과대적합이라면 C를 감소시켜서 모델을 규제할 수 있다.

```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocesing import StardardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris() #iris 데이터 로드
X = iris["data"][: ,(2, 3)] #꽃잎 길이와 꽃잎 너비
y = (iris["target"] == 2) #Iris-Virginica

svm_clf = Pipeline([
     ("scaler", StandardScaler()), 
     ("linear_svc", LinearSVC(C=1, loss="hinge")), #힌지손실 함수 사용 
     ])

svm_clf.fit(X, y)
```

```python
svm_clf.predict([[5.5, 1.7]]) #길이가 5.5이고 너비가 1.7인 것에 대한 예측
```

* LinearSVC 클래스 대신
```python
SVC(kernel="linear", C=1) #으로 대체 가능

SGDClassifier(loss="hinge", alpha = 1(m*C)) # m은 샘플 수
# 확률적 경사하강법을 적용하여 대체 가능
```

## [5.2] 비선형 SVM 분류
* 비선형 데이터셋을 다루는 한 가지 방법은 다항 특성과 같은 특성을 추가하는 것

![비선형 데이터](https://ifh.cc/g/rkyvbS.png) 

* PolynimialFeatures와 StadardScaler, LinearSVC를 연결하여 사용해보자

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#make_moos는 두 개의 반달 모양 데이터셋이다.
X, y = make_moons(n_samples = 100, noise = 0.15)
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()), 
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

polynomial_svm_clf.fit(X,y)
```
### 다항식 커널
* SVM을 사용할 때 커널 트릭(Kernel trick)을 사용하여 실제로는 특성을 추가하지 않으면서 다항식 특성을 추가한 것과 같은 결과를 얻을 수 있다. 

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([ 
    ("scaler", StandardScaler()), 
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5)) 
    #모델이 과대적합 -> 차수를 줄여야함
    #모델이 과소적합 -> 차수를 늘려야함
    #coef0는 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절
    ])
    
poly_kernel_svm_clf.fit(X, y)
```

![결과](https://ifh.cc/g/fX57d1.jpg)

### 유사도 특성
* 비선형 특성을 다루는 다른 기법은 각 샘플이 특징 랜드마크와 얼마나 닮았는지 측정하는 유사도 함수로 계산한 특성을 추가하는 것이다

### 가우시안 RBF 커널
* 유사도 특성을 많이 추가하는 것과 비슷한 결과를 얻을 수 있다. 

* 아래 코드는 가우시안 RBF 커널을 사용한 SVC모델 

```python
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma = 5, C = 0.001))
])
rbf_kernel_svm_clf.fit(X, y)
```

![RBF 커널](https://ifh.cc/g/RmYhvm.jpg)

### 계산 복잡도
|모델|계산복잡도|
|------|---|
|LinearSVC|$O(m × n)$|
|SGDClassifier|$O(m × n)$|
|SVC|$O(m^2 × n)$ ~ $O(m^3 × n)$|

## [5.3] SVM 회귀

* 앞서 말한것처럼 SVM은 선형, 비선형 분류뿐만 아니라 선형, 비선형 회귀에도 사용이 가능

* SVM 회귀는 제한된 마진 오류 안에서 도로 안에 가능한 많은 샘플이 들어가도록 학습한다. 
    * 도로의 폭은 하이퍼 파라미터로 조절한다.

![SVM 회귀](https://ifh.cc/g/waKpO6.png)
* 왼쪽은 마진을 크게, 오른쪽은 마진을 작게하여 만듦
* 이 모델은 $\varepsilon$에 민감하지 않다
    * 마진 안에서는 훈련 샘플이 추가되어도 모델의 예측에는 영향이 없기 때문

```python
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epslion = 1.5)
svm_reg.fit(X, y)
```

![2차 다항커널 SVM회귀](https://ifh.cc/g/wXhafj.jpg)
* 규제(C)를 다르게 하여 적용한 모습

```python
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel = "poly", degree = 2, C = 100, epsilon = 0.1)
svm_poly_reg.fit(X, y)
```

* SVR은 SVM의 회귀 버전, LinearSVR은 LinearSVM의 회귀 버전
    * LinearSVR은 필요한 시간이 훈련 세트의 크기에 비례해서 선형적으로 늘어남. 하지만 SVR은 훈련 세트가 커지면 훨씬 느려진다.
    * LinearSVC : SVC에서 선형 커널 함수로 가정해 커널을 진행하는 것
        *  선형 커널로 분류할 때 더 빠르게 실행한다는 장점이 있다.
        *  규제에 편향을 포함하기 때문에, StandardScaler를 수행해야 한다. 
        *  훈련보다 샘플이 적다면, 성능을 높이기 위해 dual 매개변수를 False로 지정해야 한다.

    * LinearSVC는 kernel이라는 파라미터를 받지 않는다. 그 이유는 이미 선형 커널(Linear Kernel)로 가정하기 때문
    * SVC에서 선형 커널 함수로 가정해 커널을 진행하는 것

## [5.4] SVM 이론

* 선형 SVM 분류기 모델의 예측
    * $W^TX^+b < 0$ 일때 0으로 예측
    * $W^TX^+b >= 0$ 일때 1으로 예측

* 선형 SVM 분류기를 훈련한다는 것은 마진 오류를 하나도 발생시키지 않거나(하드 마진), 제한적인 마진 오류를 가지면서(소프트 마진) 가능한 마진을 크게하는 $W$와 $b$를 찾는 것이다. 

![image](https://user-images.githubusercontent.com/83739271/204321259-6fc5a843-0bb3-4438-bdef-0a55faf7bfdc.png)
* 가중치 벡터가 작을수록 마진은 커진다. 마진을 크게하기 위해 $W$를 최소화하려고 한다.

![image](https://user-images.githubusercontent.com/83739271/204321605-c776a529-edd5-428e-a237-38dba6b18774.png)

![image](https://user-images.githubusercontent.com/83739271/204321770-73163074-afbe-4da5-bbd5-b9019627a74e.png)
* 소프트 마진 분류기의 목적 함수를 구성하기 위해 각 샘플에 대해 슬랙함수를 도입
    * 슬랙함수는 $i$번째 샘플이 얼마나 마진을 위반할지 정한다.

</br>

* 하드 마진과 소프트 마진 문제에는 모두 선형적인 제약 조건이 있는 볼록 함수의 이차 최적화 문제이다.
    * 이런 문제를 콰드라틱 프로그래밍(QP) 문제라고 한다.

![image](https://user-images.githubusercontent.com/83739271/204325035-3859776e-d1a0-4597-a12b-2dca9f08af2e.png)

* 힌지 손실 함수
    * SVM 알고리즘을 위한 손실 함수 
    * $t<1$이면 기울기는 -1, $t>1$이면 기울기는 0
    * SGDClassifier에서 loss 매개변수에 hinge로 지정하면 선형 SVM 문제가 된다.
        * SGDClassifier에서 $t=1$일때 -1을 사용함.

- - - - -

* SVM
    * SVC : Classification 에 사용되는 SVM 모델을 의미
    * SVR : Regression 에 사용되는 SVM 모델을 의미

* SVC vs SVR
    * SVM을 회귀에 적용하는 방법은 SVC와 목표를 반대로 가지는 것이다. 즉, 마진 내부에 데이터가 최대한 많이 들어가도록 학습

* SVC


<div align="center">

![image](https://user-images.githubusercontent.com/83739271/204327536-767d81b4-e238-452b-8350-e44e700b269b.png)

![image](https://user-images.githubusercontent.com/83739271/204326307-035f44e5-c5df-4092-b338-ecd261b7bb65.png)

</div>

* boundary의 결정은 위의 그림과 같이, 각 분포로부터 margin을 최대화하는 것을 목표로 결정하게 된다. 즉 boundary의 선에 margin을 덧붙여서 최대한 점에 닿지 않도록 margin을 키우는 것이다

<div align="center">

![image](https://user-images.githubusercontent.com/83739271/204326573-e5bd7e31-6b82-4371-9d5f-caeefb0f6590.png)

</div>

* SVC의 경우 Margin안에 포함된 점들의 error를 기준으로 model cost를 계산한다. 거기에 반대방향으로 분류된, 즉 바운더리를 넘어서 존재하는 점들의 error만을 추가한다.  그림을 보면 아래 노란 부분에 포함된 점들과 boundary의 중심에 위치한 직선과의 error를 계산하는 것이다.

* SVR

<div align="center">

![image](https://user-images.githubusercontent.com/83739271/204326978-989e5dd8-2a5b-43f7-a15b-a82fd0bbc088.png)
   
</div>

* 이와 달리 SVR은 일정 Margin의 범위를 넘어선 점들에 대한 error를 기준으로 model cost를 계산한다. 아래 그림으로 본다면 margin 바깥에 위치한 점들과 차이를 계산하는 빨간선이 곧 한 점에 있어서 error로 계산되는 것이다. 

* 결론적으로 SVM은 각 cost를 최소화하는 boundary를 찾아내는 작업이다.
