## Weight decay
* 가중치 감소(Weight decay) 기법은 overfitting을 억제해 성능을 높이는 기술이다.
* loss function에 L2 norm과 같은 penalty를 추가하는 정규화 기법이다.
* 일반적으로 bias가 아닌 weight에만 decay를 적용하는 것을 더 선호한다.

* 장점
  * Overfitting을 방지한다.
  * weight를 작게 유지한다. -> gradient exploding을 방지한다.
