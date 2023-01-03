## Sparsity

* Pruning의 개념을 되짚어 보면 네트워크의 성능이 크게 저하되지 않는 선에서 weight들을 최대한 sparse(희소하게) 하게 만드는 방법으로 정의한다.
* 여기서 말하는 Sparsity(희소성)의 정의는

![image](https://user-images.githubusercontent.com/83739271/210325976-2ced5db4-df35-4c0b-b978-d895dbd8791e.png)

* 위 세개 행렬중 제일 왼쪽의 행렬이 제일 sparse하다. 하나의 요소를 제외하고는 전부 0이기 때문.
* 대부분의 weight가 0이면, sparse(희소)한 것으로 간주할 수 있다.
* sparse하다의 기준은 네트워크에서 얼마나 많은 weight가 정확하게 0인지를 나타내는 척도이다.
* 가장 간단한 방법은 $l_{0}$ norm을 활용하는 것
  * 수식에 따르면 각 요소가 1 또는 0으로 되며, 0이 아닌 값들의 개수가 $l_{0}$ norm이 된다.
  * 이것이 전체 weight 개수 대비 $l_{0}$ norm의 값을 확인하면 sparsity를 알 수 있다.
