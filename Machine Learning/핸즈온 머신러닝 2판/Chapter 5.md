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

