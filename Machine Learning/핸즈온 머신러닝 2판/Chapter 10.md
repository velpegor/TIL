# <핸즈온 머신러닝 2판 - Chapter.10>

## [10.1] 생물학적 뉴런에서 인공 뉴런까지

### [10.1.2] 뉴런을 사용한 논리 연산
* 매우 단순한 신경망 모델을 제안하였는데, 그것이 인공 뉴런
* 논리 명제 계산 가능

![image](https://user-images.githubusercontent.com/83739271/206389763-f8bd3f3d-68b3-4288-b3aa-d3d564d2acb1.png)

* 첫 번째 네트워크
    * 항등 함수
    * 뉴런 A 활성화, 뉴런 C 활성화
    * 뉴런 A 비활성화, 뉴런 C 비활성화
* 두 번재 네트워크
    * 논리곱 연산
    * A, B 모두 활성화될 때 활성화
* 세 번째 네트워크
    * A, B 중 하나가 활성화되면 C도 활성화
* 네 번째 네트워크 
    * 뉴런 A가 활성화되고 뉴런 B가 비활성화될 때 뉴런 C가 활성화됩니다. 만약 뉴런 A가 항상 활성화되어 있다면 이 네트워크는 논리 부정 연산이 됩니다. 즉, 뉴런 B가 비활성화될 때 뉴런 C가 활성화되고, 또는 정반대로 뉴런 B가 활성화될 때 뉴런 C가 비활성화됩니다.

### [10.1.3] 퍼셉트론
* TLU와 LTU로 구성되어 있다.
* TLU는 입력의 가중치 합을 계산한 뒤 계산 합에 계단함수를 적용하여 결과를 출력

* 퍼셉트론에 널리 사용되는 계단 함수는 헤비사이드 계단 함수이다.

![image](https://user-images.githubusercontent.com/83739271/206391702-eb5eae4f-666d-44b7-b134-138809e90aa2.png)

* TLU는 간단한 선형 이진 분류 문제에 사용 가능하다.
* 퍼셉트론은 층이 하나뿐인 TLU로 구성된다. 각 TLU는 모든 입력에 연결되어 있다. 
* 한 층에 있는 모든 뉴런이 이진 층의 모든 뉴런과 연결되어 있을 때 이를 완전 연결 층, 밀집 층 이라고 한다.

![image](https://user-images.githubusercontent.com/83739271/206392065-64d8a790-f42e-459a-aa80-dccc24f03ec8.png)

![image](https://user-images.githubusercontent.com/83739271/206392602-b1f2625d-0e45-4524-a955-c095024bb9ac.png)

</br>

* 퍼셉트론은 네트워크가 예측할 때 만드는 오차를 반영하도록 조금 변형된 규칙을 사용하여 훈련한다.
* 퍼셉트론은 한 번에 하나의 샘플이 주입되면 각 샘플에 대해 예측이 만들어진다.

![image](https://user-images.githubusercontent.com/83739271/206393035-cff40986-dff9-44a3-bd76-0d60362ed6ea.png)

* wi, j는 i번째 입력 뉴런과 j번째 출력 뉴런 사이를 연결하는 가중치입니다.
* xi는 현재 훈련 샘플의 i번째 뉴런의 입력값입니다.
* yi는 현재 훈련 샘플의 j번째 출력 뉴런의 출력값입니다.
* yj는 현재 훈련 샘플의 j번째 출력 뉴런의 타깃값입니다. 
* η는 학습률입니다

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2,3)] #꽃잎의 길이와 너비
y = (iris.target == 0).astype(np.int) #부채붓꽃인가?

per_clf = Perceptron()
per_clf.fit(X, y)

y_pred = per_clf.predict([2, 0.5])
```

* 퍼셉트론은 XOR 문제를 해결할 수 없는데, 이를 해결할 것이 다층 퍼셉트론(MLP)이다.
![image](https://user-images.githubusercontent.com/83739271/206394082-e531d81a-8b63-4f29-948d-f5a0649ae76d.png)

### [10.1.4] 다층 퍼셉트론과 역전파
* 다층 퍼셉트론은 입력층 하나와 은닉충이라고 불리는 하나 이상의 TLU층과 마지막 출력층으로 구성되어 있다.
* 입력층과 가까운 층을 보통 하위 층이라 부르고, 출력에 가까운 층을 상위 층이라고 부른다.
![image](https://user-images.githubusercontent.com/83739271/206395232-29b6c791-fe52-4552-a315-e48c929ebdc3.png)

</br>

* 은닉층을 여러 개 쌓아 올린 인공 신경망을 심층 신경망(DNN)이라고 한다.
* 역전파 : 효율적인 기법으로 그레이언트를 자동으로 계산하는 경사 하강법이다.
* 입력층 -> 첫 번째 은닉층 -> 층에 있는 모든 뉴런의 출력을 계산 -> 결과를 다음 층으로 전달 -> 계산후 다음 층으로 전달 -> 출력층까지 전달 후 계산 / 이것이 정방향 계산
* 연쇄 법칙 : 이전 층의 연결 가중치가 이 오차의 기여 정도 얼마나 기여했는지 측정. 이렇게 입력층에 도달할 때까지 역방향으로 계속된다.
* 오차 그레디언트를 거꾸로 전파함으로써 효율적으로 네트워크에 있는 모든 연결 가중치에 대한 오차 그레디언트를 측정 -> 방금 계산한 오차 그레디언트를 사용해 네트워크에 있는 모든 연결 가중치를 수정한다. 
* 정방향 계산 오차 측정 -> 역방향 계산으로 오차에 기여한 정도를 측정 -> 오차가 감소하도록 가중치를 조정(경사 하강법)

</br>

* 역전파 알고리즘에 쓰는 활성화 함수는 로지스틱(시그모이드) 함수 뿐만아니라 하이퍼볼릭 탄젠트 함수, ReLU 함수가 존재한다.
* 활성화 함수가 필요한 이유 : 선형 변환을 여러 개 연결해도 얻을 수 있는건 선형 변환 뿐이다. 따라서 층 사이에 비선형성을 추가하지 않으면 아무리 층을 쌓아도 하나의 층과 동일해지기 때문

### [10.1.5] 회귀를 위한 다층 퍼셉트론
* 동시에 여러 값을 예측하는 경우 출력 차원마다 출력 뉴런이 하나씩 필요하다
* 출력이 항상 양수여야 한다면 출력층에 ReLU 활성화 함수를 사용할 수 있다. 또는 softplus 활성화 함수를 이용가능하다.
* 훈련에 사용하는 손실 함수는 전형적으로 평균 제곱 오차이다. 하지만 훈련 세트에 이상치가 많다면 사용할 수 없습니다. 따라서 이 Hubor 손실을 사용 가능하다. (이상치에 덜 민감함)

### [10.1.6] 분류를 위한 다층 퍼셉트론
* 다층 퍼셉트론은 다중 레이블 이진 분류 문제를 쉽게 처리할 수 있다.
* 예를 들어 이메일이 스팸 인지 아닌지 예측하고 동시에 긴급한 이메일인지 아닌지 예측하는 이메일 분류 시스템에 적용 가능하다
* 첫 번째 뉴런은 이메일이 스팸일 확률을 출력하고, 두 번째 뉴런은 긴급한 메일일 확률을 출력한다.
* 여러 클래스 중 한 클래스에만 속할 수 있다면 클래스의 수만큼 하나의 출력 뉴런이 필요하다. 

![image](https://user-images.githubusercontent.com/83739271/206400893-7d7b0364-393d-4aa5-89e5-1a47ba2a8784.png)
* 소프트맥스 함수는 모든 예측 확률을 0과 1사이로 만들고 더했을 때 1이 되도록 만든다.
* 확률 분포를 예측해야 하므로 손실 함수에는 일반적으로 크로스 엔트포리 손실을 선택하는 것이 좋다.

![image](https://user-images.githubusercontent.com/83739271/206401042-9bdc6d70-f906-4e07-98c1-1fd334bbd2be.png)
* 다중 레이블 분류는 여러 개가 정답이 될 수 있다. 따라서 로지스틱 함수를 사용

## [10.2] 케라스로 다층 퍼셉트론 구현하기
### [10.2.2] Sequential API를 사용하여 이미지 분류기 만들기

```python
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full) , (X_test, y_test) = fashion_mnist.load_data()

# validation과 train set 분류
# X값을 255로 나누어 0과 1사이로 스케일링
X_valid, X_train = X_train_full[:5000] / 255.0
, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# 클래스 이름을 정의
class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Sequential API를 사용하여 모델 만들기
# 두 개의 은닉층으로 이루어진 분류용 다층 퍼셉트론
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu")
    keras.layers.Dense(10, activation="softmax")
])
# 첫 번째 라인은 Sequentail 모델을 만든다.
# Flatten 층은 입력 이미지를 1D 배열로 변환한다. 
# 그 다음 뉴런 300개를 가진 Dense 은닉층을 추가한다. ReLu 활성화 함수 사용
# 그 다음 뉴런 100개를 가진 Dense 은닉층을 추가한다. ReLu 활성화 함수 사용
# 마지막으로 뉴런 10개를 가진 Dense 출력층을 추가한다. 배타적인 클래스이고 멀티클래스이므로 소프트맥스 활성화 함수를 사용한다.

# 인덱스를 통해 특정 레이어를 불러올 수 있다.
hidden1 = model.layers[1]
hidden1.name
>>> 'dense'
model.get_layer('dense') is hidden 1
>>> True

# 층의 모든 파라미터는 get_weights()와 set_weights() 메서드를 사용해 접근 가능하다.
weights, biases = hidden1.get_weights()

# 모델 컴파일
# 멀티클래스 분류를 할 때 타겟이 클래스 인덱스인 경우 sparse_categorical_crossentropy 사용
# 멀티클래스 분류를 할 때 타겟이 각 클래스일 확률을 나타낸 벡터인 경우 categorical_crossentropy 사용
# 이진 분류나 다중 레이블 이진 분류를 수행한다면 출력층에 "softmax" 대신 "sigmoid" 함수를 사용하고 "binary_crossentropy" 사용
# 시그모이드는 객체에 대해 개별적인 분류를 수행, 이진분류의 경우 사용
# 소프트맥스는 모든 클래스에 대해 확률값의 합이 1로 고정, 다분류 출력이 가능
model.compile(loss = "sparse_categorical_crossentropy",
optimizer = "sgd",
metrics=["accuracy"]
)

# 모델 훈련과 평가
history = model.fit(X_train, y_train, epochs=30, validation_data = (X_valid, y_valid))

# history object
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # 수직 축의 범위를 [0-1]로 설정
plt.show()
# validation 곡선과 train 곡선이 가깝다. 크게 과대적합 되지 않았다는 증거.

# 훈련된 모델을 평가
model.evaluate(X_test, y_test)

# 모델을 사용해 예측 만들기
# predict()를 사용해 새로운 샘플에 대해 예측을 만들 수 있다.
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

# predict_classes로 무슨 클래스로 예측되는지 구할 수 있다.
y_pred = model.predict_classes(X_new)
y_pred = np.argmax(y_proba, axis=1)
y_pred
>>> array([9, 2, 1])
```
![image](https://user-images.githubusercontent.com/83739271/206410603-f488a132-86fb-45a7-961a-92924f300c97.png)

* 시그모이드 함수는 로지스틱 함수의 한 케이스이다
* input이 하나일 때 사용되는 시그모이드 함수를 input이 여러 개일때도 사용할 수 있도록 일반화 한 것이 소프트맥스 함수이다.
* 소프트맥스 : 3개 이상으로 분류하는 다중 클래스 분류에서 사용되는 활성화 함수 / 확률 분포를 알 수 있다, 각 클래스에 속할 확률을 추정, 출력 값의 총합이 1이 된다.

### [10.2.3] Sequential API를 사용하여 회귀용 다층 퍼셉트론 만들기
* 사이킷런의 fetch_california_housing 데이터를 사용

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StadardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 분류와의 차이점 : 출력층이 활성화 함수가 없는 하나의 뉴런을 가짐, 손실 함수로 평균 제곱 오차를 사용
model = keras.models.Sequential([ 
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]), keras.layers.Dense(1)])
    
    model.compile(loss="mean_squared_error", optimizer="sgd")
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)
    
    X_new = X_test[:3] # 새로운 샘플이라고 생각합니다.
    y_pred = model.predict(X_new)
```

### [10.2.4] 함수형 API를 사용해 복잡한 모델 만들기
* 와이드&딥 신경망은 입력의 일부 또는 전체가 출력층에 바로 연결된다.
* 복잡한 패턴과 간단한 규칙을 모두 학습할 수 있다.

![image](https://user-images.githubusercontent.com/83739271/206437175-cf620b6a-c015-4c11-bc82-53c8de8b411f.png)

```python
#Concatenate 층을 만들고 또 다시 함수처럼 호출하여 두번째 은닉층의 출력과 입력을 연결한다. 입력이 바로 호출됨
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])
```

* 만약 일부 특성은 짧은 경로로 전달하고, 다른 특성들은 깊은 경로로 전달하고 싶다면? 
* 아래 코드는 5개의 특성을 짧은 경로로 보내고 6개의 특성은 깊은 경로로 보낸다고 가정

```python
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))
```
![image](https://user-images.githubusercontent.com/83739271/206438303-76a8abf6-300a-4943-a547-75edbb072952.png)

* 규제 기법을 사용한 MLP
* 보조 출력을 사용해 하위 네트워크가 나머지 네트워크에 의존하지 않고 그 자체로 유용한 것을 학습하는지 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/83739271/206439380-b41c7f4c-d640-41b4-8d7a-c4eeb7582d84.png)

```python
# 규제 기법을 사용한 다층 회귀 퍼셉트론 
# aux_output 레이어를 추가하여 regulation 효과를 얻을 수 있다.
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

# model compile을 수행할 때 얼마나 regulation은 할지 정할 수 있다. 
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")

# 모델 훈련
# 주 출력과 보조 출력이 같은 것을 예측해야 하므로 동일한 레이블을 사용해야 한다.
history = model.fit( 
    [X_train_A, X_train_B], [y_train, y_train], epochs=20, validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

# evaluate의 return value로 총 loss와 더불어 개별 loss까지 전달한다
total_loss, main_loss, aux_loss = model.evaluate(
    [X_test_A, X_test_B], [y_test, y_test]
)

# Predict
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
```

### [10.2.6] 모델 저장과 복원

```python
model = keras.models.Sequential([...]) # 또는 keras.Model([...])
model.compile([...])
model.fit([...])
model.save("my_keras_model.h5")

# 모델 로드
model = keras.models.load_model("my_keras_model.h5")
```

### [10.2.7] 콜백 사용하기
* 훈련의 시작이나 끝에 호출할 객체 리스트를 지정할 수 있다.
* 에포크의 시작이나 끝, 각 배치 처리 전후에 호출 할 수도 있다.

```python
[...] # 모델을 만들고 컴파일하기
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint_cb])

# 조기 종료를 구현하는 방법
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5") # 최상의 모델로 복원

# 조기 종료를 구현하는 또 다른 방법 : EarlyStopping 콜백
# 선택적으로 최상의 모델을 복원 가능
# 시간과 컴퓨팅 자원을 낭비하지 않기 위해 진전이 없는 경우 일찍 멈추는 콜백을 사용 가능
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb])
# patience : 데이터가 더 나빠지는 것을 얼마나 반복해야 early stop하는지를 결정
# restore_best_weights : 훈련 종료 시 가장 성능이 좋았던 모델의 weight를 불러온다
```
* EarlyStopping의 경우 모델이 향상되지 않으면 훈련이 자동으로 중지되기 때문에 에포크의 숫자를 크게 지정해도 된다. 

## [10.3] 신경망 하이퍼파라미너 튜닝하기
* 조정할 하이퍼파라미터가 많기 때문에 어떤 조합이 최적인지 알아야한다.
* GridSearchCV, RandomizedSearchCV를 사용해 하이퍼파라미터 공간을 탐색할 수 있다.
* 아래 코드는 일련의 하이퍼파라미터로 케라스 모델을 만들고 컴파일 하는 함수이다.

```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential() 
    model.add(keras.layers.InputLayer(input_shape=input_shape)) 
    for layer in range(n_hidden): 
        model.add(keras.layers.Dense(n_neurons, activation="relu")) 
    model.add(keras.layers.Dense(1)) 
    optimizer = keras.optimizers.SGD(lr=learning_rate) 
    model.compile(loss="mse", optimizer=optimizer) 
    return model

# 위 함수를 이용해 KerasRegressor 클래스의 객체를 만든다.
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
keras_reg.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)
```

* 아래 코드는 RandomizedSearchCV를 통해 하이퍼파라미터 탐색을 수행

```python
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden" : [0, 1, 2, 3]
    "n_neurons" : np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2)
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
callbacks=[keras.callbacks.EarlyStopping(patience=10)])

rnd_search_cv.best_params_
rnd_search_cv.best_score_
model = rnd_search_cv.best_estimator_.model
```
* RandomizedSearchCV는 k-fold 교차 검증을 사용하기 때문에 X_valid와 y_valid를 사용하지 않는다
* 하이퍼 파라미터 최적화에 사용할 수 있는 몇 개의 파이썬 라이브러리
    * Hyperopt : 모든 종류의 복잡한 탐색 공간에 대해 최적화를 수행할 수 있는 라이브러리
    * Hyperas, kopt, Talos : 케라스 모델을 위한 하이퍼파라미터 최적화 라이브러리
    * 케라스 튜너 : 사용하기 쉬운 케라스 하이퍼파라미터 최적화 라이브러리
    * Scikit-Optimize : 범용 최적화 라이브러리
    * Spearmint : 베이즈 최적화 라이브러리
    * Hyperband : 빠른 하이퍼파라미터 튜닝 라이브러리
    * Sklearn-Deap : 진화 알고리즘 기반 하이퍼파라미터화 라이브러리

### [10.3.2] 은닉층의 뉴런 개수
* 입력층과 출력층의 뉴런 개수는 해당 작업에 필요한 입력과 출력의 형태에 따라 결정
    * MNIST는 28*28개의 입력 뉴런과 10개의 출력 뉴런이 필요
* 일반적으로 층의 뉴런 수 보다 층 수를 늘리는 쪽이 이득이 많음

