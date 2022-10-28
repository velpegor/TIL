# <핸즈온 머신러닝 2판 - Chapter.3>

## [3.2] SGDClassifier
확률적 경사 하강법을 이용하여 선형 모델을 구현

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)
```
[Tip]

SGDClassifier는 훈련하는 데 무작위성을 사용. 따라서 결과를 재현하기 위해서 random_state 매개변수를 지정해야함


## [3.3] 성능 측정 방법

### 교차 검증
* 교차 검증이란 쉽게 말해 데이터를 여러 번 반복해서 나누고 여러 모델을 학습하여 성능을 평가하는 방법이다

* 사이킷런에서는 교차 검증을 더 쉽게 할 수 있는 API인 cross_val_score()을 제공한다.

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train, cv = 3, scoring = "accuracy")
```

* cross_val_score(estimator, X, y, scoring=평가지표, cv=교차 검증 폴드 수) 

* 장점 
    * 모든 데이터셋을 훈련 시킬 수 있다. 
    * 모든 데이터셋을 평가에 활용할 수 있다.

* 단점
    * Iteration 횟수가 많기 때문에 모델 훈련/평가 시간이 오래 걸린다.

### 오차 행렬
* 분류기의 성능을 평가하는 더 좋은 방법은 Confusion Matrix(오차 행렬)를 조사하는 것이다

* 예를 들어 숫자 5의 이미지를 3으로 잘못 분류한 횟수를 세는 것이다.

* 오차 행렬을 만들려면 우선 실제 타겟과 비교할 수 있도록 예측값을 만들어야 함. 
    * cross_val_predict() : k-겹 교차 검증을 수행하지만 평가 점수를 반환하지 않고 각 test fold에서 얻은 예측을 반환한다. 

```python
# 5가 맞는지 아닌지 분류하는 예시

from sklearn.model_selection import cross_val_predict

y_train_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3) 

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_train_predict)
```
* 오차 행렬의 행은 실제 클래스를 나타내고 열은 예측한 클래스를 나타낸다. 


||음성|양성|
|---|---|---|
|음성|TN|FP|
|양성|FN|TP|


>첫번째 행은 '5 아님' 이미지에 대한 것이다.<br/>
TN : 5가 아닌 이미지를 정확하게 분류<br/>
FP : 5가 아닌 이미지를 5라고 잘못 분류



>두번째 행은 '5' 이미지에 대한 것이다<br/>
FN : 5인 이미지를 5가 아닌 이미지라고 잘못 분류<br/>
TP : 5인 이미지를 5라고 정확하게 분류

* Precision : TP / (TP + FP) / 실제 True, 예측 True

* Recall : TP / (TP + FN) / 실제 False, 예측 True

```python
from sklearn.metrics import precision_socre, recall_score
precision_score(y_train_5, y_train_predict)
recall_score(y_train_5, y_train_predict)
```

* F1-score : 2TP / (2TP + FN + FP) / Precision과 Recall의 조화 평균

```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_predict)
```

#### ROC 커브 곡선
* 거짓양성비율(FPR)에 대한 진짜양성비율(TPR)의 곡선
    * FPR : 양성으로 잘못 분류된 음성 샘플의 비율


* 민감도(재현율)에 대한 1-특이도 그래프

```python
from sklearn.metrics import roc_auc_score
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                            method="decision_function") 
roc_auc_score(y_train_5, y_scores)
```

## RandomForestClassifier
* 배깅(bagging) : 같은 알고리즘으로 여러개의 분류기를 만드는 알고리즘

* 배깅의 대표적인 알고리즘이 랜덤포레스트이다.

* 랜덤 포레스트는 결정 트리를 기반으로 하는 알고리즘입니다. 

* 랜덤 포레스트는 여러 개의 결정 트리 분류기가 배깅을 기반으로 각자의 데이터를 샘플링 하여 학습을 수행한 후에 최종적으로 보팅을 통해 예측 결정을 하게 됩니다.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

forest_clf = RandomForestClassifier(random_state = 42)
y_prod_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = "predict_proba" )
```

## [3.4] 다중 분류
* 둘 이상의 클래스를 구별할 수 있다
* SGDClassifier, RandomForest와 같은 일부 알고리즘들은 여러 개의 클래스를 직접 처리할 수 있는 반면, Logistic, SVM 같은 알고리즘은 이진 분류만 가능하다
* 이진 분류기를 여러개 사용해 다중 클래스를 분류하는 기법도 존재한다. ex) 특정 숫자 하나만 구분하는 숫자별 이진 분류기 10개 
* 클래스가 N개라면 분류기는 N(N-1)/2 개 필요하다


### 분류 서포트 벡터 머신 (SVC)
<br/>

```python
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svc_clf.predict([some_digit]) #some_digit는 숫자 5인 MNIST 데이터
```
* 위 코드는 타겟 클래스(y_train_5) 대신 0~9까지 원래 타겟 클래스(y_train)을 사용해 SVC를 훈련시켰다.

```python
some_digit_scores = svm_clf.decision_function([some_digit])
```
* 위 코드를 통해 클래스마다 해당하는 Score를 반환할 수 있다.

### OvO (OneVsOneClassifier), OvR(OneVsRestClassifier)

* 사이킷런에서 OvO, OvR를 강제하려면 OneVsOneClassifier, OneVsRestClassifier를 사용할 수 있다. 

* OvO는 1대1 이진 분류기를 만드는 방법이다.

* OvR은 하나 대 나머지를 비교하는 방법이다. One-versus-All(OvA)라고 불리기도 한다 
    * 각 클래스에 대한 점수를 비교하여 가장 높은 값을 선택하는 방식이다.

```python
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
```

### 스케일 조정

* 입력 스케일 조정을 통해 정확도를 높일 수 있다.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64)) #스케일러를 통해 float 형식의 값이 생기므로 float로 전환해줄 필요가 있다
```

## [3.5] 에러 분석

### 오차 행렬 분석
* cross_val_predict() 함수를 통해 예측을 만들고 이전처럼 confusion_matrix를 생성해낸다

```python
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv = 3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()
```
* 오차 행렬을 통해 어떤 클래스가 분류가 잘 안되는지 확인할 수 있다. 
* 이를 통해 분류기의 성능 향상 방안에 대한 통찰이 가능하다. 예를 들어 성능이 나오지 않는 클래스에 대해 데이터를 더 확보하여 학습시킬 수 있다.
* 또한 더 적합한 모델을 탐색할 수 있다 
    * 위 코드에서 MNIST에 대한 이미지 분류기로 SGDClassifier를 사용했는데, 선형 분류기는 픽셀에 가중치를 할당하고 새로운 이미지에 대해 단순히 픽셀 강도의 가중치 합을 클래스의 점수로 계산한다. 따라서 3과 5는 몇 개의 픽셀만 다르기 때문에 모델이 쉽게 혼동한다

## [3.6] 다중 레이블 분류

* 지금까지는 각 샘플이 하나의 클래스에만 할당되었다. 하지만 분류기마다 여러개의 클래스를 출력해야 할 때도 있다. 
    * ex) 사람이 여러명 포함된 이미지

```python
#두 개의 타겟 레이블이 포함된 y_multilabel 학습
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

## [3.7] 다중 출력 분류

* 다중 레이블 분류에서 한 레이블에 대한 출력 값이 다중 클래스가 될 수 있도록 일반화 한 것이다
    * ex ) MNIST에서 픽셀의 강도를 담은 배열로 출력을 한다. 분류기의 출력이 다중 레이블(픽셀 당 한 레이블)이고 각 레이블의 값은 여러개 가진다(0~255까지의 픽셀 강도).

```python
noise = np.random.randint(0, 100, (len(X_train), 784)) #len(X_train) x 784 행렬에서 0~100 랜덤 수 생성 / 784 = 28x28(이미지 크기)
X_train_mod = X_train + noise #shape 60000, 784
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

some_index = 0  # 0번 인덱스 

plt.subplot(121); plot_digit(X_test_mod[some_index])  # 잡음 추가된 이미지
plt.subplot(122); plot_digit(y_test_mod[some_index])  # 원본 이미지

plt.show()
```
