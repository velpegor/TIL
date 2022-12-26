## Degradation problem 

![image](https://user-images.githubusercontent.com/83739271/209525301-5a4132ef-0bfc-4e7f-8afb-526b87d4ddfc.png)

* Degradation problem은 네트워크의 Depth가 커질수록 accuracy는 saturated 상태(현상태에서 더 진전이 없어져 버리는 상태)가 되고 degradation이 진행된다.
  * layer가 깊어지면서 train과 test가 학습이 되지 않는 현상을 말합니다. 쉽게 말해 작은 layer의 네트워크보다 높은 error를 갖게 되는 것을 말한다.
* Degradation은 overfitting에 의해서 생겨나는 것이 아니다. 더 많은 Layer를 넣을수록 training error가 더 높아진다.
* Degradation problem은 Resnet을 통해 모델의 Depth가 깊어지더라도 정확도를 상승하게 함으로써 해결하였다.
