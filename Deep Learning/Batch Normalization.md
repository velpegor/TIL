## Batch Normalization

* 각 layer마다 새로운 input 분포에 적응해야해서 input 분포의 변화는 피할 수 없다. 이러한 분포의 변화를 Internal Covariate Shift라 한다.
* Internal Covariate Shift 현상이 발생하면 매우 불안정한 학습이 발생한다 -> 성능이 떨어지는 문제 발생

![image](https://user-images.githubusercontent.com/83739271/209627514-0dcad9ba-1e17-44b3-b845-a10869358613.png)
         
* 각 배치별로 평균과 분산을 이용해 정규화하는 것을 Batch Normalization이라 한다.
* Batch Normalization의 감마[scale], 베타[shift]을 통해 비선형 성질을 유지하며 학습이 가능하다.
* 초기 값에 대한 gradient의 의존성을 줄임으로써 네트워크의 gradient 흐름에 유익한 영향을 미친다 -> 격차의 위험없이 훨씬 더 높은 학습률을 사용할 수 있다.
         
