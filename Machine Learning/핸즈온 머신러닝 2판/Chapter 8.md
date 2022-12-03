# <핸즈온 머신러닝 2판 - Chapter.8>

## [8.1] 차원의 저주
* 훈련 샘플 각각이 많은 특성을 가지는데 이런 많은 특성은 훈련을 느리게 할 뿐만 아니라, 좋은 솔루션을 찾기 어렵게 만든다. 이런 문제를 종종 차원의 저주라고 한다.
* 훈련 세트의 차원이 클수록 과대적합 위험이 커진다.
* 차원의 저주를 해결하는 해결책 중 하나는 훈련 샘플의 밀도가 충분히 높아질 때까지 훈련 세트의 크기를 키우는 것

## [8.2] 차원 축소를 위한 접근 방법

### [8.2.1] 투영
* 모든 훈련 샘플이 고차원 공간 안의 저차원 부분공간에 놓여 있다.

<img width="804" alt="image" src="https://user-images.githubusercontent.com/83739271/205434838-cb2a29f3-f4fe-4bed-bb8c-a9fecedf87c7.png">

* 모든 샘플이 2차원 공간에 가깝게 배치되어 있다.
* 위 사진을 2차원 부분 공간에 수직으로 투영하여 2D 데이터셋을 얻을 수 있다.

</br>

<img width="806" alt="image" src="https://user-images.githubusercontent.com/83739271/205434881-9b72ae78-65f6-4ff0-8f2b-f24f9f871b9e.png">

* 이러한 투영이 언제나 최선의 방법은 아니다. 스위스 롤 데이터셋처럼 부분 공간이 뒤틀리거나 휘어 있기도 하다.

</br>

<img width="812" alt="image" src="https://user-images.githubusercontent.com/83739271/205434918-5ae79c9b-850e-47b7-95db-c95a8984065c.png">

### [8.2.2] 매니폴드 학습
* 스위스롤은 2D 매니폴드의 예시이다.
* 2D 매니폴드는 고차원 공간에서 휘어지거나 뒤틀린 2D 모양이다.
* 많은 차원 축소 알고리즘이 훈련 샘플이 놓여있는 매니폴드를 모델링하는 식으로 작동한다. 이를 매니폴드 학습이라고 한다.
* 이는 대부분 실제 고차원 데이터셋이 더 낮은 저차원 매니폴드에 가깝게 놓여 있다는 매니폴드 가정 또는 매니폴드 가설에 근거한다.

</br>
* 요약하자면 모델을 훈련시키기 전에 훈련 세트의 차원을 감소시키면 훈련 속도는 빨라지지만 항상 더 낫거나 간단한 솔루션이 되는 것은 아니다.

## [8.3] PCA
* 주성분 분석(PCA)는 차원 축소 알고리즘이다. 먼저 데이터에 가장 가까운 초평면을 정의한 다음, 데이터를 이 평면에 투영시킨다.

### [8.3.1] 분산 보존
* 먼저 올바른 초평면을 선택해야한다.

<img width="1113" alt="image" src="https://user-images.githubusercontent.com/83739271/205435216-309c9b94-9a4a-4dec-bc2b-0fd609bbafde.png">
* 다른 방향으로 투영하는 것보다 분산이 최대로 보존되는 축을 선택하는 것이 정보가 가장 적게 손신된다.

</br>

### [8.3.2] 주성분
* PCA는 훈련 세트에서 분산이 최대인 축을 찾는다.
* 투영할 축을 찾는 $i$번째 축을 이 데이터의 $i$번째 주성분(PC)라고 부른다.
* 주성분을 찾기위해 SVD를 이용하여 찾을 수 있다.

```python
X_centered = X - X.mean(axis = 0)
U, s, Vt = np.linalg.svd(X_centerd)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
```

### [8.3.3] d차원으로 투영하기

* 첫 두 개의 주성분으로 정의된 평면에 훈련 세트를 투영한다. 
```python
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
```

### [8.3.4] 사이킷런 이용하기
* 사이컷의 PCA 모델은 SVD 분해 방법을 사용하여 구현한다.
* 아래 코드는 PCA 모델을 사용해 데이터셋의 차원을 2로 줄이는 코드이다.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transfrom(X)
```

### [8.3.5] 설명된 분산의 비율
* explained_variance_ratio_ 변수에 저장된 주성분의 설명된 분산의 비율도 유용한 정보로 사용된다.
* 이 비율은 각 주성분의 축을 다라 있는 데이터셋의 분산 비율을 나타낸다.

```python
pca.explained_variance_ratio_
>>> array([0.84248607, 0.14631839]) # 출력 예시 
```
* 이 데이터셋 분산의 84.2%가 첫 번째 PC를 따라 놓여있고 14,6%가 두 번째 PC를 따라 놓여 있음을 알 수 있다.

### [8.3.6] 적절한 차원 수 선택하기
* 코드를 통해 차원을 축소하지 않고 PCA를 계산한 뒤 훈련 세트의 분산을 95%로 유지하는 데 필요한 최소한의 차원 수를 계산한다.

```python
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components = 0.95)
X_reduced = pca.fit_transform(X_train)
```

### [8.3.7] 압축을 위한 PCA
* 차원을 축소한 후에는 훈련 세트의 크기가 줄어든다. 
* 원본 데이터와 재구성된 데이터(압축한 후 원복한 것) 사이의 평균 제곱 거리를 재구성 오차 라고 한다.
* 아래 코드를 통해 MNIST 데이터셋을 154차원으로 압축하고 inverse_transform() 메서드를 사용해 784차원으로 복원한다.

```python
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recoverd = pca.inverse_transform(X_reduced)
```

<img width="1139" alt="image" src="https://user-images.githubusercontent.com/83739271/205435921-0af426e9-53bf-4425-acce-554ac82985fd.png">

### [8.3.8] 랜덤 PCA
* svd_solver 매개변수를 'randomized'로 지정하면 사이킷런은 랜덤 PCA라 부르는 확률적 알고리즘을 사용해 처음 d개의 주성분에 대한 근삿값을 빠르게 찾는다.
* d가 n보다 많이 작으면 SVD보다 훨씬 빠르다

```python
rnd_pca = PCA(n_components=154, svd_solver = 'randomized')
X_reduced = rnd_pca.fit_transfrom(X_train)
```


### [8.3.9] 점진적 PCA
* 훈련 세트를 미니배치로 나눈 뒤 IPCA 알고리즘에 한 번에 하나씩 주입한다.
* 이런 방식은 훈련 세트가 클 때 유용하고 온라인으로 (즉, 새로운 데이터가 준비되는 대로 실시간으로) PCA를 적용할 수도 있다.

```python
from sklearn.decomposition import IncrementalPCA

n_batched = 100
inc_pca = IncrementalPCA(n_components = 154)
for X_batch in np.array_split(X_train, n_batched):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```
* IncrementalPCA는 특정 순간에 배열의 일부만 사용하기 때문에 메모리 부족 문제를 해결할 수 있다.

## [8.4] 커널 PCA
* PCA를 적용해 차원 축소를 위한 복잡한 비선형 투형을 수행할 수 있다. 이를 커널 PCA라고 한다. 
* 이 기법은 투영된 후 샘플의 군집을 유지하거나 꼬인 매니폴드에 가까운 데이터셋을 펼칠 때도 유용하다.

```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel = "rbf", gamma = 0.04)
X_reduced = rbf_pca.fit_transform(X)
```
<img width="1109" alt="image" src="https://user-images.githubusercontent.com/83739271/205436272-7c1bf220-7608-4f78-89f2-5a2996f5bf23.png">

### [8.4.1] 커널 선택과 하이퍼파라미터 튜닝
* kPCA는 비지도 학습이기 때문에 명확한 성능 측정 기준이 없다.
* 다음 코드는 두 단계의 파이프라인을 만드는데, 먼저 kPCA를 사용해 차원을 2차원으로 축소하고 분류를 위해 로지스틱 회귀를 적용한다. 그런 다음 파이프라인 마지막 단계에서 가장 높은 분류 정확도를 얻기 위해 GridSearchCV를 사용해 kPCA의 가장 좋은 커널과 gamma 파라미터를 찾는다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("kpca", KernelPCA(n_components = 2)),
    ("log_reg", LogisticRegression())
])

param_grid = [{
    "kpca__gamma" : np.linspace(0.03, 0.05, 10),
    "kpca__kernel" : ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

>>> print(grid_search.best_params_)
{'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}
```

* 차원 축소 후 다시 원본으로 역전시키면 원본 공간으로 돌아가는가? 아니다. 특성 공간(무한 차원)에 놓이게 된다. 원본 공간의 포인트를 찾을 수 있는 방법이 있는가? 재구성 원상을 통해 재구성된 포인트에 가깝게 매핑된 원본 공간의 포인트를 찾을 수 있다. 
* 지도학습 회귀모델을 통해 train : 투영된 샘플, target : 원본 샘플
* fit_inverse_transform=True를 통해 자동으로 수행

```python
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
```

* 그런 다음 재구성 원상 오차를 계산할 수 있다.

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(X, X_preimage)
>>> 32.78630879
```
## [8.6] 다른 차원 축소 기법
* 랜덤 투영
* 다차원 스케일링
* Isomap
* t-SNE
