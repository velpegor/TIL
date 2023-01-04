## Lottery Ticket Hypothesis

* ICLR 2019 -  The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
* 위 논문에서 다음과 같은 가설을 제시한다 
  * 무작위로 초기화된 밀집된 Deep NeuralNet에서 임의로 가져왔을 때, 따로 트레이닝을 하더라도, 기존의 네트워크와 같은 성능을 같은 학습 횟수 내에 달성할 수 있을 것이다.

![image](https://user-images.githubusercontent.com/83739271/210503634-d680a4b6-351a-4f48-8399-53fb56cbd401.png)

* 위 처럼 부분적인 네트워크를 잘 골라내면 기존의 큰 네트워크와 같은 효과를 낼 수 있을 것이라 생각했다. 이것이 Lottery Ticket Hypothesis이다.
* Winning Ticket : 쓸데없이 티켓을 많이 사지 말고 불필요한 weight들을 제거함으로써 당첨이 될 winning ticket인 subnetwork만으로 네트워크를 구성해야한다는 것이 저자의 설명이다.

![image](https://user-images.githubusercontent.com/83739271/210504258-ae567c93-fc1c-4ac8-ae99-1bc263e8273c.png)

* 4번에서 처음 초기화된 network weights $\theta_{0}$를 다시 넣어서 재학습을 진행하는 방법
* 기존의 pruning 방식이 network를 1번 학습시키고, 일부만큼 pruning을 하고 나머지 weights를 초기화하는 one-shot 접근법이었다면, 본 논문에서는 n 라운드만큼 각 라운드마다 조금씩 pruning하는 iterative pruning방법을 사용한다고한다.
* 논문에서 제안한 초기 weight를 이용하면 더 빨리 학습하고 높은 테스트 정확도를 얻게된다.
