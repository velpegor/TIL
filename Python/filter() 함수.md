# filter() 함수

* 파이썬 내장 함수 filter()는 list, iterable한 객체와 같은 자료들을 필터링하는 역할을 수행
* 특정 조건에 맞는 요소들만 출력한다는 의미

## filter(function, iterable)
* function 인자는 조건을 정하는 함수, iterable은 list와 같은 순회 가능한 자료형

## 예시

* data라는 list에서 0의 값의 인덱스를 얻고싶을 때
```python
data = [4, 28, 43, 21, 8, 26, 0, 23, 48, 29, 22, 1, 27, 27, 25, 14,
0, 1, 38, 46, 31, 28, 42, 35, 44, 26, 37, 17, 8, 0,
1, 39, 48, 2, 19, 14, 41, 31, 40, 11, 30, 48, 23, 0, 10, 25, 47, 32, 19, 40, 8,
0, 19, 45]

rest_list = list(filter(lambda x: data[x] == 0, range(len(data))))
print(rest_list)
>>> [6, 16, 29, 43, 51]
```

* 40 초과의 수만 출력하고 싶을 떄
```python
def FilterFunc(num):
  return num>40
  
num_list = [10,20,30,40,50,60,70,80,90]

filter_list = list(filter(FilterFunc, numlist))
print(fliter_list)
>>> [50,60,70,80,90]
```
