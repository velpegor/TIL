## Residual Learning

![image](https://user-images.githubusercontent.com/83739271/209524336-3f5fe4ab-7ac8-4546-aedc-abed77211f89.png)

* Layer에서 H(x)가 아닌 출력과 입력의 차인 H(x)-x 를 얻도록 목표를 수정한 것
* Residual Function인 F(x)=H(x)-x를 최소화하는 방향으로 학습을 해야한다. 
* x는 변경이 불가능한 입력값이므로 F(x)가 0이 되는 것이 최적의 해이다. 따라서 H(x) = x로 mapping하는 것이 학습의 목표이다.
* H(x) = x라는 최적의 목표값이 존재하기 때문에 F(x)의 학습이 더욱 쉬워진다. 
  * Optimize하기 쉬워 진다 = Loss값이 쉽게 떨어진다 = 최적의 딥러닝 모델 파라미터를 찾기 쉽다.

* Residual Leanring의 깊이가 깊어져도, 정확도가 향상됨.
