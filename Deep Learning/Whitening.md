## Whitening

* Layer로 들어가는 입력을 uncorrelated하게 만들어주고, 평균 0, 분산 1로 바꿔준다.
  * 더 빨리 수렴이 가능하다.
  * input 분포가 고정되므로 Internal Covariate Shift를 줄일 수 있다.


* 문제는 whitening을 하기 위해서는 covariance matrix의 계산과 inverse의 계산이 필요하기 때문에 계산량이 많을 뿐더러, 설상가상으로 whitening을 하면 일부 parameter 들의 영향이 무시된다는 것이다.
  * layer의 입력 u에 편향 b를 더한 것을 x라고 하면 (x=u+b) 여기에 평균을 빼주어 normalization을 한다. 평균값을 주는 과정에서 b도 같이 빠지게 되어 결국 출력에서 b의 영향이 없어진다. 또한 단순히 E[x]를 빼는 것이 아니라 표준편차로 나눠주는 등의 scaling 과정까지 한다면 이러한 경향은 더욱 악화될 것이다. 
* 단순하게 Whitening만 한다면, Whitening 과정과 paremeter를 계산하기 위한 최적화 과정과 무관하게 진행되기 때문에 특정 파라미터가 계속 커지는 상태로 진행될 수 있다.
  * 이러한 Whitening의 문제를 해결하기 위해 Batch Norm이 나옴.
