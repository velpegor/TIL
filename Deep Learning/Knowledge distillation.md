## Knowledge distillation

* "Distilling the Knowledge in a Neural Network" - NIPS 2014
* 미리 잘 학습된 큰 네트워크(Teacher network)의 지식을 실제로 사용하고자 하는 작은 네트워크(Student network)에게 전달하는 것
* 작은 네트워크도 큰 네트워크와 비슷한 성능을 낼 수 있도록, 학습과정에서 큰 네트워크의 지식을 작은 네트워크에게 전달하여 성능을 높이겠다는 목적

![image](https://user-images.githubusercontent.com/83739271/211696062-56062358-2bc3-4f4b-941e-7eff217d1b36.png)

<img width="466" alt="image" src="https://user-images.githubusercontent.com/83739271/211696732-d59443e9-0fec-4df3-a8c7-86a3912192ce.png">

* Loss Function의 왼쪽 항은 Ground truth와 Student의 분류 결과 차이를 CrossEntropyLoss로 계산한 값
* Loss Function의 오른쪽 항은 큰 네트워크와 작은 네트워크의 분류 결과의 차이를 Loss에 포함시킨 값
  * Teacher와 Student의 Output logit을 Softmax로 변환한 값의 차이를 CrossEntropyLoss로 계산한다. 분류 결과가 같다면 작은 값을 취한다.

<img width="222" alt="image" src="https://user-images.githubusercontent.com/83739271/211697116-b82c6126-1869-40ef-a20c-d09e799eec21.png">

* 두 네트워크의 분류 결과를 비교하기 위해 Soft Label을 사용한다.
  * Hard Label을 사용하면 다른 클래스에 대한 유추 값에 대한 정보가 사라지게 된다.

* T는 Temperature paremeter인데 Softmax함수가 입력값이 큰 것은 아주 크게, 작은 것은 아주 작게 만드는 성질을 완화해준다. 즉, 낮은 입력값의 출력은 더 크게 만들어주고, 큰 입력값의 출력은 작게 만들어준다.
* Temperature를 사용하여 Soft label을 사용하는 이점을 최대화 한다.
