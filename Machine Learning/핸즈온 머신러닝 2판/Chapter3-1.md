# <핸즈온 머신러닝 2판 - Chapter.3 ~ 3.3>

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
