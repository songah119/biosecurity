# face-recognition

## Data Load

- code
    - train data load
    
    ```python
    data_dir = './pca_data1/train' # csv 파일이 저장된 디렉토리의 경로
    
    # 트레인 데이터 로드
    train_data = []
    train_labels = []
    
    # 폴더 내의 이미지 데이터 가져오기
    for folder_name in os.listdir(data_dir):
        folder_dir = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_dir):
            for filename in os.listdir(folder_dir):
                file_path = os.path.join(folder_dir, filename)
                if os.path.isfile(file_path):
                    # 이미지를 128x128 크기로 리사이즈하여 로드
                    image = Image.open(file_path)
                    # 이미지 데이터를 1차원 배열로 변환
                    image_array = np.array(image).flatten()
                    train_data.append(image_array)
                    train_labels.append(os.path.basename(folder_dir))  # 파일 이름에서 라벨 추출
    
    # 이미지 데이터를 numpy 배열로 변환
    train_img = np.array(train_data)
    print(train_img.shape)
    ```
    
    - test data load
    
    ```python
    # 테스트 데이터 경로
    test_data_dir = './pca_data1/test'
    
    # 테스트 데이터 로드
    test_data = []
    test_labels = []
    
    # 폴더 내의 이미지 데이터 가져오기
    for folder_name in os.listdir(test_data_dir):
        folder_dir = os.path.join(test_data_dir, folder_name)
        if os.path.isdir(folder_dir):
            for filename in os.listdir(folder_dir):
                file_path = os.path.join(folder_dir, filename)
                if os.path.isfile(file_path):
                    # 이미지를 128x128 크기로 리사이즈하여 로드
                    image = Image.open(file_path)
                    # 이미지 데이터를 1차원 배열로 변환
                    image_array = np.array(image).flatten()
                    test_data.append(image_array)
                    test_labels.append(os.path.basename(folder_dir))  # 파일 이름에서 라벨 추출
    
    # 이미지 데이터를 numpy 배열로 변환
    test_img = np.array(test_data)
    print(test_img.shape)
    ```
    

폴더를 순회하며 train/test data와 label을 각각 저장합니다.
train_data : 966개, 50x37 사이즈의 train 이미지가 저장되어있는 리스트
train_labels : train 데이터의 각 라벨이 저장되어있는 문자열 리스트

test_data : 322개, 50x37 사이즈의 test 이미지가 저장되어있는 리스트
test_labels : test 데이터의 각 라벨이 저장되어있는 문자열 리스트

## PCA

### 몇 차원으로 축소해야할까?

- code
    - 차원 별 설명률 비교
    
    ```python
    # PCA 차원 비교
    
    # 64차원 PCA
    pca64 = PCA(n_components=64)
    img_data_pca64 = pca64.fit_transform(train_img)
    
    # 128차원 PCA
    pca128 = PCA(n_components=128)
    img_data_pca64 = pca128.fit_transform(train_img)
    
    # 256차원 PCA
    pca256 = PCA(n_components=256)
    img_data_pca64 = pca256.fit_transform(train_img)
    
    # 설명률 출력
    print("64차원 PCA 설명률:", sum(pca64.explained_variance_ratio_))
    print("128차원 PCA 설명률:", sum(pca128.explained_variance_ratio_))
    print("256차원 PCA 설명률:", sum(pca256.explained_variance_ratio_))
    ```
    
    - 설명률이 95가 나오기 위한 차원
    
    ```python
    # PCA를 사용하여 차원 축소
    pca = PCA(n_components=0.95) # 95% 이상의 설명률을 보이는 차원을 선택
    train_img_pca = pca.fit_transform(train_img)
    
    # 설명률 계산
    var_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    print('선택한 차원 수:', np.argmax(var_ratio_cumsum >= 0.95) + 1) # 차원 수 출력
    # 선택한 차원 수: 135
    ```
    

차원 축소 시에 가장 중요한 것은 설명률이라고 생각한다. 실제로 64, 128, 256차원으로 축소한 결과 설명률은 다음과 같았다. 보통 95%정도 설명할 수 있으면 좋은 차원이라고 한다. 

실제로 계산을 해보니 135차원일때 95%의 설명률을 지니는 것을 확인할 수 있다. 

64, 128, 256차원을 비교해본 결과 128차원이 94.6%의 설명률을 지니는 것을 알 수 있었다. 

따라서, 주어진 데이터를 128차원으로 줄이는 것이 가장 적절하다고 판단하였다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cf1d8d5c-afbe-4d4b-8cb0-236135cd7577/Untitled.png)

![스크린샷 2023-04-30 오후 7.37.46.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/71092b8c-2531-419a-b666-e7bdc4cb0491/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.37.46.png)

![스크린샷 2023-04-30 오후 7.41.17.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81a8e1c5-c44a-4eff-9813-1b62b4908e96/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.41.17.png)

### 상위 5개의 eigenvector 시각화

- code
    
    ```python
    # 주성분 상위 5개 eigenvector 가져오기
    eigenvectors = pca.components_[:5]
    
    # 주성분 eigenvector 시각화
    fig, axes = plt.subplots(1, 5, figsize=(12, 2))
    
    for i, eigenvector in enumerate(eigenvectors):
        # 주성분 eigenvector를 이미지로 변환
        eigenvector_image = eigenvector.reshape(50, 37)
        
        # 이미지를 0-255 범위로 조정
        eigenvector_image = (eigenvector_image - np.min(eigenvector_image)) / (
            np.max(eigenvector_image) - np.min(eigenvector_image)
        )
        eigenvector_image = (eigenvector_image * 255).astype(np.uint8)
        
        # 이미지를 흑백으로 표시
        axes[i].imshow(eigenvector_image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'PC {i+1}')
    
    plt.tight_layout()
    plt.show()
    ```
    

다음은 PCA의 각 컴포넌트들을 50*37의 이미지로 표현한 것이다. 상위 5개의 컴포넌트만 출력했다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/061ad4eb-3432-4f2b-9446-39e1fbec36e7/Untitled.png)

## Exploratory Data Analysis(EDA)

### **Imbalanced Data**

아래와 같이 George_W_Bush의 데이터가 전체 데이터의 40퍼센트 이상인 것을 알 수 있다. 

![스크린샷 2023-04-30 오후 7.52.59.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2dc1ed43-dbe9-4260-8f4e-a8381f0cb3da/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.52.59.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f02fbd6-ddd1-4ba3-97ab-c036aec1a847/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b04375dc-c852-466a-9cf2-cf001a7032dc/Untitled.png)

데이터가 불균형하다면 분포도가 높은 클래스에 모델이 가중치를 많이 두게되고, 모델은 "분포가 높은 것으로 예측하게 된다면 어느정도 맞힐 수 있겠지?"라고 생각하게 된다. (다 찍어도 정답률 40은 된다..)

따라서 불균형 문제를 해결하지 않으면 모델은 가중치가 높은 클래스를 더 예측하려고 하기 때문에 Accuracy는 높아질 수 있지만, 분포가 작은 값에 대한 Precision은 낮을 수 있고, 분포가 작은 클래스의 recall이 낮아지는 문제가 발생한다.

데이터 불균형을 해결하지 않은 것과, undersampling을 통해 해결한 것 두가지를 해보았다.

## Prediction in imbalanced data

### L2 distance

- code
    
    ```python
    # 예측된 라벨을 저장할 리스트
    predicted_labels = []
    
    # L2 distance를 계산하여 가장 가까운 훈련 데이터의 라벨을 예측값으로 지정합니다.
    for test_image in test_img_pca:
        distances = np.sqrt(np.sum((train_img_pca - test_image)**2, axis=1))
        closest_idx = np.argmin(distances)
        predicted_label = train_labels[closest_idx]
        predicted_labels.append(predicted_label)
    
    # 테스트 데이터와 예측된 훈련 데이터의 라벨을 함께 출력합니다.
    for i, test_label in enumerate(test_labels):
        print("Test label: {} Predicted label: {}".format(test_label, predicted_labels[i]))
    ```
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2d2ec12e-4ef7-4893-a4bf-e5dd66175273/Untitled.png)

![스크린샷 2023-04-30 오후 8.26.48.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/13629693-1f9b-4947-a43c-ad83c6748cf4/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.26.48.png)

George_W_Bush 로 FRR, FAR

![스크린샷 2023-04-30 오후 8.51.44.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27a57931-72ac-4abb-bf4f-2acca39c2159/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.51.44.png)

### Random Forest

- code
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Create a random forest classifier with 100 trees
    rfc = RandomForestClassifier(n_estimators=100)
    
    # Train the model on the training data
    rfc.fit(train_img_pca, train_labels)
    
    # Make predictions on the test data
    predicted_labels = rfc.predict(test_img_pca)
    
    # 테스트 데이터와 예측된 훈련 데이터의 라벨을 함께 출력합니다.
    for i, test_label in enumerate(test_labels):
        print("Test label: {} Predicted label: {}".format(test_label, predicted_labels[i]))
    ```
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f7b9ab12-9c6e-4afd-a241-6b0f8d851191/Untitled.png)

![스크린샷 2023-04-30 오후 8.28.50.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/86979d0d-6dec-469b-9650-eada0c9f4287/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.28.50.png)

George_W_Bush 로 FRR, FAR

![스크린샷 2023-04-30 오후 8.48.27.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61733b6a-6345-4260-a126-7e197f904799/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.48.27.png)

### FRR, FAR with George_W_Bush

- code
    
    ```python
    # George_W_Bush 를 기준으로 라벨을 0 또는 1로 변환
    train_labels_binary = np.asarray([1 if label == "George_W_Bush" else 0 for label in train_labels])
    test_labels_binary = np.asarray([1 if label == "George_W_Bush" else 0 for label in test_labels])
    pred_labels_binary = np.asarray([1 if label == "George_W_Bush" else 0 for label in predicted_labels])
    
    # FRR: George_W_Bush 인데 George_W_Bush 가 아니라고 예측한 비율
    frr = 1 - np.mean(pred_labels_binary[test_labels_binary == 1])
    
    # FAR: George_W_Bush 가 아닌데 George_W_Bush 로 예측한 비율
    far = np.mean(pred_labels_binary[test_labels_binary == 0])
    
    # 결과 출력
    print("False Rejection Rate (FRR):", frr)
    print("False Acceptance Rate (FAR):", far)
    ```
    
- L2 distance

![스크린샷 2023-04-30 오후 8.51.44.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27a57931-72ac-4abb-bf4f-2acca39c2159/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.51.44.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f8b5ee0-cb8c-4104-b7c7-d5623fcdcb6c/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06d4c4ee-3206-417f-bfb0-c697158f3eff/Untitled.png)

- Random Forest

![스크린샷 2023-04-30 오후 8.48.27.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61733b6a-6345-4260-a126-7e197f904799/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.48.27.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6caae0c0-6c1b-4d85-a059-5e2c455cd0c4/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f72e4b6-741f-488d-b695-0c888c96c41a/Untitled.png)

## Prediction in balanced data

### undersampling

원본 train 데이터의 개수의 통계값은 다음과 같다. 

**평균:**138, **중앙값:**94

라벨 당 94개 데이터가 넘어가지 않도록 데이터를 가져오려고한다.

- code
    
    ```python
    data_dir = './pca_data1/train'
    
    train_data_s = []
    train_labels_s = []
    
    for folder_name in os.listdir(data_dir):
        folder_dir = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_dir):
            count = 0  # 이미지 파일 수 카운터 초기화
            for filename in os.listdir(folder_dir):
                if count >= 94:  # 이미지 파일 수가 94개 이상이면 루프 종료
                    break
                file_path = os.path.join(folder_dir, filename)
                if os.path.isfile(file_path):
                    image = Image.open(file_path)
                    image_array = np.array(image).flatten()
                    train_data_s.append(image_array)
                    train_labels_s.append(os.path.basename(folder_dir))
                    count += 1  # 이미지 파일 수 증가
    
    # 이미지 데이터를 numpy 배열로 변환
    train_img_s = np.array(train_data_s)
    print(train_img_s.shape)
    ```
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e362159c-1f6d-44ec-b0ed-b05eb16e3a89/Untitled.png)

### L2 distance

- code
    
    ```python
    # 예측된 라벨을 저장할 리스트
    predicted_labels = []
    
    # L2 distance를 계산하여 가장 가까운 훈련 데이터의 라벨을 예측값으로 지정합니다.
    for test_image in test_img_pca:
        distances = np.sqrt(np.sum((train_img_pca_s - test_image)**2, axis=1))
        closest_idx = np.argmin(distances)
        predicted_label = train_labels_s[closest_idx]
        predicted_labels.append(predicted_label)
    
    # 테스트 데이터와 예측된 훈련 데이터의 라벨을 함께 출력합니다.
    for i, test_label in enumerate(test_labels):
        print("Test label: {} Predicted label: {}".format(test_label, predicted_labels[i]))
    ```
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/20491493-cc57-45ac-bbdc-a20f892b0967/Untitled.png)

![스크린샷 2023-05-01 오전 12.51.47.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1e851b28-6fc6-4102-9a0f-10fd548030a8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-01_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.51.47.png)

### Random Forest

- code
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Create a random forest classifier with 100 trees
    rfc = RandomForestClassifier(n_estimators=100)
    
    # Train the model on the training data
    rfc.fit(train_img_pca_s, train_labels_s)
    
    # Make predictions on the test data
    predicted_labels = rfc.predict(test_img_pca)
    
    # 테스트 데이터와 예측된 훈련 데이터의 라벨을 함께 출력합니다.
    for i, test_label in enumerate(test_labels):
        print("Test label: {} Predicted label: {}".format(test_label, predicted_labels[i]))
    ```
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5d3ca0b9-937a-4380-a914-e2818f2f02af/Untitled.png)

![스크린샷 2023-05-01 오전 12.52.53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/daca4cbc-13ca-4375-975e-f94fecc4cfba/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-01_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.52.53.png)

## Result

- imbalanced L2

![스크린샷 2023-04-30 오후 8.26.48.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/13629693-1f9b-4947-a43c-ad83c6748cf4/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.26.48.png)

- imbalanced RF

![스크린샷 2023-04-30 오후 8.28.50.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/86979d0d-6dec-469b-9650-eada0c9f4287/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.28.50.png)

- balanced L2

![스크린샷 2023-05-01 오전 12.51.47.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1e851b28-6fc6-4102-9a0f-10fd548030a8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-01_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.51.47.png)

- balanced RF

![스크린샷 2023-05-01 오전 12.52.53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/daca4cbc-13ca-4375-975e-f94fecc4cfba/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-01_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.52.53.png)

imbalanced와 비교했을 때 데이터가 아무래도 적어지기 때문에 accuracy는 떨어지는 것 같다. 

하지만 Macro Precision, Recall의 경우 많이 떨어지지 않은 것을 볼 수 있다.

또한, 데이터가 상대적으로 적었던 사람들에 대해서는 precision(모델이 True라고 분류한 것 중에서 실제 True인 것의 비율), recall(실제 True인 것 중에서 모델이 True라고 예측한 것의 비율)이 오른 것을 확인할 수 있다. 

## My face data

### preprocessing

- code
    
    ```python
    # 이미지 파일 로드
    image = cv2.imread('./pca_data1/myface/my_pic.png')
    
    # 자른 얼굴 이미지 보여주기
    cv2.imshow('Face Image', image)
    plt.imshow(image)
    
    # 얼굴 영역 검출기 생성
    detector = dlib.get_frontal_face_detector()
    
    # 이미지에서 얼굴 영역 검출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # 각 얼굴 영역에 대해서 반복적으로 처리
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = image[y-160:y+h+40, x-10:x+w+10]
        image_c = cv2.resize(face_image, (37,50)) # reshape
        image_c = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./pca_data1/myface/my_pic_crop.png', image_c)
        # 자른 얼굴 이미지 보여주기
        plt.imshow(image_c, cmap='gray')
    ```
    

이미지를 로드하고, dlib 의 detector를 사용해서 얼굴 영역을 검출했다.

이때, 눈코입만 추출하길래 적당한 수를 대입하여 얼굴 영역 중 잘리는 영역이 없도록 조절하였다. 

또한, 37*50 사이즈로 줄였다. 화질이 많이 좋지 않아졌다… 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b1b9a38f-06ef-4790-9cc6-0d6c4acd039b/Untitled.png)

**→**

~~음.. 무서워서 좀 줄였다..~~

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fcb7e628-25b5-4c18-9c04-726db81d662a/Untitled.png)

### predict with L2 distance

- code
    
    ```python
    from sklearn.metrics.pairwise import pairwise_distances
    
    # image_c PCA 변환
    image_c_pca = pca.transform(image_c.reshape(1, -1))
    
    # L2 distance 계산
    distances = np.linalg.norm(train_img_pca_s - image_c_pca, axis=1)
    
    # 가장 가까운 이미지의 인덱스 찾기
    closest_idx = np.argmin(distances)
    
    # 가장 가까운 이미지의 라벨 출력
    closest_label = train_labels_s[closest_idx]
    print(f"The closest image label is {closest_label}")
    
    # 가장 가까운 이미지 출력
    closest_image = train_img_s[closest_idx].reshape(50, 37)
    plt.imshow(closest_image, cmap="gray")
    plt.axis("off")
    plt.title(f"The closest image label is {closest_label}")
    plt.show()
    ```
    

해당 이미지는 50,37이기 때문에 PCA를 통해 (128,)로 줄인 후 확인해본 결과 

나는 아래 이미지(George_W_Bush)와 가장 L2 distance가 가까웠다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a06d837f-d215-437e-8884-4fb1bf27a6d9/Untitled.png)

![스크린샷 2023-04-30 오후 11.31.43.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e811c6ec-5c0d-473b-abea-0ba9fa3a2889/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.31.43.png)

### I’m not George_W_Bush

- code
    
    ```python
    # 가장 가까운 이미지와의 L2 distance 출력
    closest_distance = distances[closest_idx]
    print(f"The closest distance is {closest_distance:.2f}")
    ```
    

어떻게 하면 나와 부시가 다르다는 것을 보일 수 있을까?

![스크린샷 2023-04-30 오후 11.45.09.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/79a17169-e195-4500-bc96-7c1aa7ba85ba/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-04-30_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.45.09.png)

임계값을 설정해서 distance가 너무 큰 경우, 새로운 얼굴이 학습된 데이터와 다르다고 얘기할 수 있을 것 같다.

이때 임계값을 설정할 때는 내 얼굴과 가장 가까운 distance인 2310보다는 작아야할 것이다.

## PCA limitation

차원은 높은데 개수가 적은 데이터로 학습된 모델은 주어진 데이터에 과대적합한 모델이 된다. 이를 차원의 저주(Curse of dimensionality)라고도 하는데, 따라서 차원축소가 필요하다. 

그러나, PCA는 선형 방식으로 정사영하면서 차원을 축소시킨다. 이때, 군집된 데이터들이 뭉게지는 단점이 있다. 이로 인해서 세부적인 특징들이 손실될 수 있고, 얼굴 이미지 간의 차이를 인식하는 데 어려움을 겪을 수 있다.

따라서, 딥러닝을 사용하거나, UMAP과 같은 차원축소 알고리즘을 사용하는 것이 좋을 것 같다.
