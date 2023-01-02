## Kernel Pruning

* Filter Pruning은 기존의 방법처럼 parameter를 개별적으로 제거하는 것이 아니라 filter를 통째로 제거하는 것
* sparsity를 위한 별도의 라이브러리가 필요하지 않고, conv filter를 직접적으로 제거하다보니 FLOPS가 줄어 inference 시간을 크게 줄일 수 있고 목표로 하는 속도 상승을 쉽게 설정할 수 있다.

<img width="520" alt="image" src="https://user-images.githubusercontent.com/83739271/210203439-88aeffe8-1ce5-4c64-8734-b035bfbd3ad7.png">
* Filter p를 변경하게 되면 다음 Layer에 영향을 줌. 
