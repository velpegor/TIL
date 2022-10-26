# <핸즈온 머신러닝 2판 - Chapter.4>

## [4.1] 선형 회귀
* 선형 회귀 모델의 예측
    * $ y  = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... +\theta_n x_n$ 

    * $ y = h_\theta(x) = \theta \cdot x $

    * $ y $는 예측값, $ n $ 은 특성의 수, $x_i$는 $i$번째 특성값, $\theta_j$는 $j$번째 모델 파라미터

* 성능 평가 지표
    * 평균 제곱근 오차(RMSE) : $ \sqrt{MSE(\theta)} $
        * RMSE를 최소화하는 $ \theta $를 찾아야 한다

* 정규 방정식
    * $ \theta = (X^TX)^{-1}X^Ty $
    * $ \theta $ 는 비용 함수를 최소화하는 값이다.
    * $ y $는 $y^{1}$ 부터 $y^{m}$ 까지 포함하는 타겟 벡터

```python
import numpy as np

X = 2 * np.random.rand(100, 1) #100x1 배열을 가지는 표준 정규분포 난수를 생성
y = 4 + 3 * X + np.random.randn(100, 1) #평균0, 표준편차 1의 가우시안 표준정규분포를 가지는 100x1 배열 생성

X_b = np.c_[np.ones((100, 1)), X] # 모든 샘플에 x0 = 1을 추가
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #정규방정식
print(theta_best)
```
* 위 코드를 통해 예측된 $ \theta_0, \theta_1 $을 알 수 있다. ($ \theta_0 = 4, \theta_1 = 3 $ 을 기대한다)

```python
X_new = np.array([[0], [2]]) # x0 = 0 , x1 = 2
X_new_b = np.c_[np.ones((2,1)), X_new] #모든 샘플에 x0 = 1을 추가
y_predict = X_new_b.dot(theta_best) # 모델의 예측 
```
* 위 코드를 통해 직접 수식으로 선형 회귀를 나타내볼 수 있다. 

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_ #선형 회귀 모델의 절편과 기울기
lin_reg.predict(X_new)
``` 
* sklearn의 LinearRegression 함수를 통해 간단하게 구현이 가능하다
    * 계산 복잡도 $ O(n^2) $

## [4.2] 경사 하강법
<br>

![경사하강법](https://ifh.cc/g/FlDdho.png)

* 여러 문제에서 최적의 해법을 찾을 수 있는 일반적인 최적화 알고리즘

* 비용 함수를 최소화 하기 위해 반복해서 파라미터를 조정

<br>


![지역 최솟값](https://ifh.cc/g/RaALvB.png)
* 문제점
    * 알고리즘이 무작위 초기화 때문에 왼쪽에서 시작한다면 전역 최솟값보다 덜 좋은 지역 최솟값에 수렴할 가능성이 있다.(볼록함수인 경우는 전역 최솟값으로 수렴)

<br>

![특성 스케일 차이](https://ifh.cc/g/zg4vNb.png)
* 경사 하강법을 사용할 때는 반드시 모든 특성이 같은 스케일을 갖도록 만들어야 한다.(ex. StadardScaler) 그렇지 않으면 수렴하는데 훨씬 오래걸린다.

### 배치 경사 하강법
* 편도함수 : $ \theta_j $ 가 조금 변경될 때 비용 함수가 얼마나 바뀌는지 계산 

* 비용함수의 편도함수 : $ \frac{\partial}{\partial \theta_j} MSE(\theta) = \frac{2}{m} \sum_{i=1}^{m}(\theta^Tx^{i} - y^i)x_j^i $

* ![비용함수의 그레디언트 벡터](https://ifh.cc/g/YPjhbV.png)

* 위 공식은 매 스텝에서 훈련 데이터 전체를 사용하여 학습한다. 그런 이유로 큰 훈련 세트에서는 느리다는 단점이 있다.

* 경사 하강법의 스텝 : $ \theta^{next step} = \theta - \eta \nabla_{\theta}MSE(\theta) $ 

```python
eta = 0.1 #학습률
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) #무작위 초기화

for iteraion in range(n_interations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
```

* 적절한 학습률이 중요하다

### 확률적 경사 하강법
* 매 스텝에서 한 개의 샘플을 무작위로 선택하고 그 하나의 샘플에 대한 그레디언트를 계산한다

* 반복마다 하나의 샘플만 처리하면 되서 알고리즘 속도가 빠르다

* 반면, 확률적(무작위)이므로 배치 경사 하강법에 비해 불안정하다

* 무작위성이 지역 최솟값을 탈출할 수 있도록 도와준다는 장점이 있다.
    * 알고리즘을 전역 최솟값에 다다르지 못한다. 따라서 초반에 학습률을 크게하여 지역 최솟값에서 벗어나게 하고 점차 줄여가며 전역 최솟값에 도달하게 한다

```python
n_epochs = 50
t0, t1 = 5, 50

#학습률을 점차 줄여주는 함수
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m) # 매 샘플에 대한 그레디언트 계산 후 파라미터 업데이트
    random_index = np.random.randint(m)
    xi = X_b[random_index:random_index+1]
    yi = y[random_index:random_index+1]
        
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)  # 하나의 샘플에 대한 그레이디언트 계산
    eta = learning_schedule(epoch * m + i)        # 학습 스케쥴을 이용한 학습률 조정
    theta = theta - eta * gradients
```

* 배치 경사 하강법은 전체 세트에 대해 1000번 반복하는 동안 위 코드(확률적 경사 하강법)은 훈련 세트에서 50번만 반복하고도 성능이 좋다

```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 1000, tol = 1e-3, penalty = None, eta0 = 0.1)
sgd_reg.fit(X, y.ravel())
```

### 미니배치 경사 하강법
* 미니배치라 불리는 임의의 작은 샘플 세트에 대해 Gradient를 계산한다.

* SGD보다 덜 불규칙적이다, 하지만 SGD보다 지역 최솟값에서 벗어나는 것은 어려울 수 있다


## [4.3] 다항 회귀
* 가지고 있는 데이터가 단순한 직선보다 복잡한 형태일 때 사용한다
    * 각 특성의 거듭제곱을 새로운 특성으로 추가하고, 이 확장된 특성을 포함한 데이터셋에서 선형 모델을 훈련시키는 것 

```python
#비선형 데이터
m = 100
X = 6 * np.random.rand(m, 1) -3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
```
* 위 코드에 선형 모델을 적용한다면 적합하지 않을 것이다 따라서 sklearn의 PolynomialFeatures를 사용해 훈련 데이터를 변환한다.

```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)
```
* X 데이터셋의 제곱된 값이 포함된 값을 X_poly에 적용한다.

```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```
* LinearRegression()에 적용 

## [4.4] 학습 곡선
* 학습 곡선을 통해 모델이 과대적합인지 과소적합인지 판단한다.

* 과소적합의 경우 훈련 샘플을 더 추가해도 의미가 없다. 더 복잡한 모델을 사용하거나 더 나은 특성을 사용해야한다

* 과대적합의 경우 검증 오차가 훈련 오차에 근접할 때까지 더 많은 훈련 데이터를 추가하는 것이다

* 편향 / 분산 트레이드오프
    * 편향 : 편향이 큰 모델은 훈련 데이터에 과소적합되기 쉽다. 예를 들어 데이터가 실제로는 2차인데 선형으로 가정하는 경우

    * 분산 : 분산은 훈련 데이터에 있는 작은 변동에 모델이 과도하게 민감하기 때문에 나타난다. 고차 다항 회귀모델 같이 자유도가 높은 모델의 경우 높은 분산을 가지기 때문에 과대적합이 되는 경향이 있다.

    * 모델의 복잡도가 커지면 통상적으로 분산이 늘어나고 편향은 줄어든다. 반대로 모델의 복잡도가 줄어들면 편향이 커지고 분산이 작아진다

## [4.5] 규제가 있는 선형 모델

* 과대적합을 감소시키는 좋은 방법은 모델을 규제하는 것이다. 즉, 모델을 제한하는 것

* 선형 회귀 모델에서는 보통 모델의 가중치를 제한함으로써 규제를 가한다. 

### 릿지 회귀

* 릿지 회귀는 규제가 추가된 선형 회귀 버전이다.

* 규제항 $ \alpha \Sigma_{i=1}^{n} \theta_i^2 $ 이 비용 함수에 추가된다.
    * $ \alpha $는 모델을 얼마나 많이 규제할지 조정한다. 

* 편향 $ \theta_0 $ 는 규제되지 않는다 따라서 $ i $는 1부터 시작한다

<br>

![릿지 회귀](https://ifh.cc/g/4Hzbgk.png)

```python
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha = 1, solver = "cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

* 아래 코드는 SGD를 사용한 코드이다
```python
sgd_reg = SGDRegresor(penalty = "l2") # penalty는 사용할 규제를 나타낸다
sgd_reg.fit(X,y.ravel())
sgd_reg.predict([[1.5]])
```

### 라쏘 회귀
* 라쏘 회귀는 선형 회귀의 또 다른 규제 버전이다
 
* 릿지 회귀처럼 비용 함수에 규제항을 더하지만 가중치 벡터 $ l_1 $ 노름을 사용한다

* $ J(\theta) = MSE(\theta) + \alpha\Sigma_{i=1}^n \left\vert \theta_i \right\vert$

<br>

![라쏘 회귀](https://ifh.cc/g/Y1y4Cp.png)

* 라쏘를 사용할 때 경사 하강법이 최적점 근처에서 진동하는 것을 막으려면 훈련하는 동안 점진적으로 학습률을 감소시켜야 한다.

* 라쏘 vs 릿지 차이점 
    * 라쏘는 가중치들이 0이지 되지만, 릿지의 가중치들은 0에 가까워질 뿐 0이 되지는 않는다. 
    * 특성이 많은데 그 중 일부분만 중요하다면 라쏘가, 특성의 중요도가 전체적으로 비슷하다면 릿지가 좀 더 괜찮은 모델이라 판단한다.

```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha = 0.1)
lasso_reg.fit(X, y)


SGDRegressor(penalty = "l1") #라쏘 대신 사용 가능하다
```

## 엘라스틱넷
* 엘라스틱넷은 릿지 회귀와 라쏘 회귀를 절충한 모델이다

* $ J(\theta) = MSE(\theta) + r\alpha\Sigma_{i=1}^n\left\vert \theta_i \right\vert + \frac{1-r}{2}\alpha\Sigma_{i=1}^n\theta_i^2$

* 일반적으로 릿지가 기본이 되지만 쓰이는 특성이 몇 개 뿐이라고 의심되면 라쏘나 엘라스틱넷이 낫다
    * 그 이유는 불필요한 특성의 가중치를 0으로 만들어 주기 때문이다

* 특성 수가 훈련 샘플 수보다 많거다 특성 몇 개가 강하게 연관되어 있을 때는 보통 라쏘가 문제를 일으키므로 라쏘보다는 엘라스틱넷을 선호한다

```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
elastic_net.fit(X, y)
```

## 조기 종료
* 검증 에러가 최솟값에 도달하면 바로 훈련을 중지시키는 것이다

```python
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

n_epochs = 500

train_errors, val_errors = [], []                    # 훈련/검증 모델 성능 기록 장치

for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled, y_train)        # warm_start=True 이기에 학습결과를 이어감.
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))  # 훈련/검증모델 성능 기록
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)                  # 최고 성능의 모델 기억해두기
best_val_rmse = np.sqrt(val_errors[best_epoch])
```

## 로지스틱 회귀
* 샘플이 특정 클래스에 속할 확률을 추정하는데 사용 된다.
    * 이메일이 스팸일 확률은 몇인가?

* 확률이 50%가 넘으면 해당 클래스에 속한다고 예측한다. 이를 이진 분류기라고 한다

* 선형 회귀처럼 결과를 출력하지 않고 결괏값의 Logistic을 출력한다
    * 로지스틱은 0과 1 사이의 값을 출력하는 시그모이드 함수이다

    * $\sigma(t) = \frac{1}{1+exp(-t)}$

* 훈련 방법 : 양성 샘플(y=1)에 대해서는 높은 확률을 추정하고 음성 샘플(y=0)에 대해서는 낮은 확률을 추정하는 모델의 파라미터 벡터를 찾는 것이다.

* 하나의 훈련 샘플에 대한 비용 함수
    * y = 1일때 ) $-log(p)$
    
    * y = 0일때 ) $-log(1-p)$

    * 비용함수는 t가 0에 가까워지면 $-log(p)$가 매우 커지므로 타당하다

* 비용 함수 : $J(\theta) = -\frac{1}{m}\Sigma_{i=1}^m[y^{i}log(p^{i})+(1-y^{i})log(1-p^{i})]$

* 이 비용 함수는 볼록 함수(Convex) 이므로 경사 하강법이 전역 최솟값을 찾는 것을 보장한다.

## 소프트맥스 회귀
* 여러 개의 이진 분류기를 훈련시켜 연결하지 않고 직접 다중 클래스를 지원하도록 일반화 될 수 있다. 이를 소프트맥스 회귀(다항 로지스틱 회귀)라고 한다.

* 샘플 x가 주어지면 먼저 소프트맥스 회귀 모델이 각 클래스 k에 대한 점수를 계산하고, 그 점수에 소프트맥스 함수를 적용하여 각 클래스의 확률을 추정한다.
    * 클래스 k에 대한 소프트맥스 점수 : $S_k(x) = (\theta^{k})^T X$

* 소프트맥스 함수 : $P_k = \sigma(s(x))_k = \frac{exp(s_k(x))}{\Sigma_{j=1}^Kexp(s_j(x))}$ 
    * K는 클래스 수, $s(x)$는 샘플 x에 대한 각 클래스의 점수를 담은 벡터, 
    * $\sigma(s(x))_k$는 샘플 x에 대한 각 클래스의 점수가 주어졌을 때 이 샘플이 클래스 k에 속할 추정 확률

* 소프트맥스 회귀 분류기의 예측 : $y = argmax(\sigma(s(x))_k)$ / $argmax$는함수를 최대화하는 변수의 값을 반환한다.

* 소프트맥스는 한 번에 하나의 클래스만 예측한다. 다중 클래스 o 다중 출력 x

* 훈련 방법 : 모델이 타겟 클래스에 대해서는 높은 확률을 추정하도록 만드는 것이 목적이다
    * 크로스 엔트로피 비용 함수를 최소화 하는 것이 타겟 클래스에 대해 낮은 확률을 예측하는 모델을 억제하므로 목적에 부합하다

    * 크로스 엔트로피 : $J(\theta) = -\frac{1}{m}\Sigma_{i=1}^m\Sigma_{k=1}^K(y_k^{i}log(p_k^{i}))$
        * $y_k^{i}$는 i번째 샘플이 클래스 $k$에 속할 타겟 확률

```python
X = iris["data"][:, (2,3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", C = 10)
softmax_reg.fit(X, y)
```
