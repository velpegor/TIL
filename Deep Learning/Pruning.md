## Pruning


![image](https://user-images.githubusercontent.com/83739271/210047407-6b8174e9-8c96-4ff5-8b21-cdc6bb68e016.png)

* Model의 weight들 중 중요도가 낮은 weight의 연결을 제거하여 모델의 파라미터를 줄이는 방법이다.
* 실제 딥러닝 모델의 저장 용량을 줄이거나 모델의 추론 속도를 더 빠르게 하기 위해 사용된다.


### Structured Pruning
* covolution layer 가중치의 채널, 필터, 구조의 그룹 단위로 제거 되는지의 여부를 판단하는 방법이다.
* Channel Pruning이 대표적으로 존재한다. 
  * network에서 상대적으로 필요없는 channel을 뽑아서 없애는 방법이다. 
  * 연산을 안해도 되므로 속도를 개선할 수 있다.
  * 구조를 통째로 날리는 것이다 보니 pruning하는 비율을 높게 하기는 어렵다.


### Unstructured Pruning
* 구조와 상관없이 특정 기준을 세워서 가지치기하듯 weight를 0으로 만들어버리는 것이다.
* 필요없다고 판단되는 weight를 0으로 만드는 것이라 높은 비율로 pruning 할 수 있다는 장점이 있다.
* pruning을 했으나 실제로는 0의 값을 가지므로 기존의 프레임워크를 사용하여 matrix 연산을 할때 계산을 하긴 해야 하므로 실질적인 inference 속도를 개선하지는 못한다.
