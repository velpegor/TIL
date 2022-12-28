## Xavier Initialization
* 이전 층의 노드 개수 n이 있을 때, 현재 레이어의 가중치를 표준편차가 $1/ \sqrt n$인 정규분포로 초기화 하는 것을 의미
* Xavier Initialization는 활성화 함수가 Sigmoid일 때는 잘 동작하지만 ReLu 함수 일때는 고르지 못하게 되는 문제가 있다.
