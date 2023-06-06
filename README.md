# face-recognition

## Data Load
폴더를 순회하며 train/test data와 label을 각각 저장합니다.
train_data : 966개, 50x37 사이즈의 train 이미지가 저장되어있는 리스트
train_labels : train 데이터의 각 라벨이 저장되어있는 문자열 리스트

test_data : 322개, 50x37 사이즈의 test 이미지가 저장되어있는 리스트
test_labels : test 데이터의 각 라벨이 저장되어있는 문자열 리스트

## PCA
### 몇 차원으로 축소해야할까?
차원 축소 시에 가장 중요한 것은 설명률이라고 생각한다. 실제로 64, 128, 256차원으로 축소한 결과 설명률은 다음과 같았다. 보통 95%정도 설명할 수 있으면 좋은 차원이라고 한다. 
실제로 계산을 해보니 135차원일때 95%의 설명률을 지니는 것을 확인할 수 있다. 
64, 128, 256차원을 비교해본 결과 128차원이 94.6%의 설명률을 지니는 것을 알 수 있었다. 
따라서, 주어진 데이터를 135차원으로 줄이는 것이 가장 적절하다고 판단하였다. 
<img width="348" alt="스크린샷 2023-06-06 오후 8 23 09" src="https://github.com/songah119/biosecurity/assets/69359991/7e196517-4441-4c62-b8fb-29881e83b648">
<img width="356" alt="스크린샷 2023-06-06 오후 8 23 27" src="https://github.com/songah119/biosecurity/assets/69359991/cf193421-0905-490c-b75f-1798a1133d85">

### eigenvector 시각화
다음은 PCA의 각 컴포넌트들을 50*37의 이미지로 표현한 것이다. 상위 5개의 컴포넌트만 출력했다.
<img width="718" alt="스크린샷 2023-06-06 오후 8 23 43" src="https://github.com/songah119/biosecurity/assets/69359991/69775046-e66c-480b-885c-b7f5fa0d37ba">

## Exploratory Data Analysis(EDA)
### **Imbalanced Data**
아래와 같이 George_W_Bush의 데이터가 전체 데이터의 40퍼센트 이상인 것을 알 수 있다. 
<img width="704" alt="스크린샷 2023-06-06 오후 8 24 15" src="https://github.com/songah119/biosecurity/assets/69359991/b87b9db4-65c3-4ba0-8f6b-d76f96bbe1a8">
데이터가 불균형하다면 분포도가 높은 클래스에 모델이 가중치를 많이 두게되고, 모델은 "분포가 높은 것으로 예측하게 된다면 어느정도 맞힐 수 있겠지?"라고 생각하게 된다. (다 찍어도 정답률 40은 된다..)
따라서 불균형 문제를 해결하지 않으면 모델은 가중치가 높은 클래스를 더 예측하려고 하기 때문에 Accuracy는 높아질 수 있지만, 분포가 작은 값에 대한 Precision은 낮을 수 있고, 분포가 작은 클래스의 recall이 낮아지는 문제가 발생한다.
데이터 불균형을 해결하지 않은 것과, undersampling을 통해 해결한 것 두가지를 제시해보려한다.

## Prediction in imbalanced data
### L2 distance
<img width="752" alt="스크린샷 2023-06-06 오후 8 26 25" src="https://github.com/songah119/biosecurity/assets/69359991/d75128b2-12de-4da4-ba69-3d28581677ac">

### Random Forest
<img width="754" alt="스크린샷 2023-06-06 오후 8 26 54" src="https://github.com/songah119/biosecurity/assets/69359991/330cd75a-ebc8-49d8-9431-be12ec857467">

### FRR, FAR with George_W_Bush
- L2 distance
<img width="401" alt="스크린샷 2023-06-06 오후 8 29 39" src="https://github.com/songah119/biosecurity/assets/69359991/d8ab9487-141e-4120-a388-ae3371850fee">
<img width="734" alt="스크린샷 2023-06-06 오후 8 30 31" src="https://github.com/songah119/biosecurity/assets/69359991/1ff92208-74c0-464f-9df7-abe811f1e699">

- Random Forest
<img width="404" alt="스크린샷 2023-06-06 오후 8 30 11" src="https://github.com/songah119/biosecurity/assets/69359991/d729b6af-8818-4193-9517-244c2eb018f2">
<img width="729" alt="스크린샷 2023-06-06 오후 8 30 52" src="https://github.com/songah119/biosecurity/assets/69359991/d04226f6-2be0-400b-bd8a-1058a23f3547">

## Prediction in balanced data
### undersampling
원본 train 데이터의 개수의 통계값은 다음과 같다. 
**평균:**138, **중앙값:**94
라벨 당 94개 데이터가 넘어가지 않도록 데이터를 가져와서 balance를 맞춰주었다
<img width="551" alt="스크린샷 2023-06-06 오후 8 31 34" src="https://github.com/songah119/biosecurity/assets/69359991/f08e86cb-df76-4257-a375-fc3108b390cb">

### L2 distance
<img width="723" alt="스크린샷 2023-06-06 오후 8 32 22" src="https://github.com/songah119/biosecurity/assets/69359991/610b7d18-088c-4dc9-8907-9d22885bbfdf">

### Random Forest
<img width="730" alt="스크린샷 2023-06-06 오후 8 32 42" src="https://github.com/songah119/biosecurity/assets/69359991/73ba3e1f-0c87-439d-8b70-0974fada9b7b">

## Result
<img width="721" alt="스크린샷 2023-06-06 오후 8 33 06" src="https://github.com/songah119/biosecurity/assets/69359991/be1881f7-950e-4b96-ba68-ff58dc3cb596">
데이터의 수가 크지 않기 때문에 지표들이 좋지는 않다. 더욱이, balance를 맞춰준 것들은 imbalanced와 비교했을 때 데이터가 더 적어지기 때문에 accuracy가 더 떨어진다. 
하지만 Macro Precision, Recall로 비교해보면 많이 떨어지지 않은 것을 볼 수 있다.
또한, 데이터가 상대적으로 적었던 사람들에 대해서는 precision(모델이 True라고 분류한 것 중에서 실제 True인 것의 비율), recall(실제 True인 것 중에서 모델이 True라고 예측한 것의 비율)이 오른 것을 확인할 수 있다. 

## My face data
### preprocessing
내 얼굴 이미지를 로드하고, dlib 의 detector를 사용해서 얼굴 영역을 검출했다.
이때, 눈코입만 추출하길래 적당한 수를 대입하여 얼굴 영역 중 잘리는 영역이 없도록 조절하고, 37*50 사이즈로 줄였다.

### predict with L2 distance
해당 이미지는 50,37이기 때문에 PCA를 통해 (128,)로 줄인 후 확인해본 결과 
나는 아래 이미지(George_W_Bush)와 가장 L2 distance가 가까웠다.

### I’m not George_W_Bush
어떻게 하면 나와 부시가 다르다는 것을 보일 수 있을까?
<img width="231" alt="스크린샷 2023-06-06 오후 8 37 09" src="https://github.com/songah119/biosecurity/assets/69359991/39472c8c-1ab8-4565-85f7-5a5f531dbbc3">
임계값을 설정해서 distance가 너무 큰 경우, 새로운 얼굴이 학습된 데이터와 다르다고 얘기할 수 있을 것 같다.
이때 임계값을 설정할 때는 내 얼굴과 가장 가까운 distance인 2310보다는 작아야할 것이다.

## PCA limitation
차원은 높은데 개수가 적은 데이터로 학습된 모델은 주어진 데이터에 과대적합한 모델이 된다. 이를 차원의 저주(Curse of dimensionality)라고도 하는데, 따라서 차원축소가 필요하다. 
그러나, PCA는 선형 방식으로 정사영하면서 차원을 축소시킨다. 이때, 군집된 데이터들이 뭉게지는 단점이 있다. 이로 인해서 세부적인 특징들이 손실될 수 있고, 얼굴 이미지 간의 차이를 인식하는 데 어려움을 겪을 수 있다.
따라서, 딥러닝을 사용하거나, UMAP과 같은 차원축소 알고리즘을 사용하는 것이 좋을 것 같다.
