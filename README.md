# 2021-2 소프트웨어융합캡스톤디자인
# 이미지의 단계적 차원 축소를 통한 강아지 행동인식 모델 연구

## 1. 개요
### 1.1. 과제 선정 배경
사람에 대한 Pose Estimation 분야와 달리 개, 고양이, 말 등 동물 대상의 Pose Estimation은 연구가 더욱 필요한 분야이다. 인간에 비해 동물의 Pose Estimation이 어려운 원인은 다음 두 가지 원인때문 이다. 
1. 온몸이 털로 덮혀 있는 경우가 많기 때문에 정확한 관절을 찾아내기가 어렵고,
2. 데이터 수집이 어렵기 때문이다.

최근 2021년 6월 25일, 한국지능정보사회진흥원 사업의 일환으로 AI Hub에서 '반려동물 구분을 위한 동물 영상'을 공개하였다.[1](https://aihub.or.kr/aidata/34146) 개, 고양이에 대한 총 500시간의 비디오 프레임과 함께 행동유형, 반려동물의 감정, 키포인트가 태깅된 Annotation 데이터를 제공한다. 이를 기반으로 PoseC3D[2](https://arxiv.org/abs/2104.13586)와 같은 최신 모델을 참고하여 강아지 행동유형을 구분해내어 스마트 펫케어 서비스 등에 적용할 수 있다.

구체적으로 Pose 정보를 활용해 행동인식성능을 높일 수 있는 방법에 대해 연구한다. 기존 PoseC3D는 단순히 Pose Extract한 결과를 히트맵으로 변환하여 그 이미지를 학습에 사용하였다. 그러나 RGB 정보가 무시돼 Pose의 성능에 크게 좌우되는 특성을 가지고 있다. 이러한 특성을 고려하여 포즈에 대한 성능을 높이는 방법을 고안한다. 또, 데이터의 품질이 좋지 않은 CCTV 영상으로 평가하여 실제 서비스에 적용할 수 있게 한다.


### 1.2. 과제 주요내용
- 반려견의 3가지 행동(bodyscratch, bodyshake, turn)에 대해서 70%의 정확도 달성
- 여러 모델을 거쳐 이미지의 차원 축소를 진행, 최종적으로는 pose 정보를 추출하여 강아지의 행동을 분류
- Object Detection, Pose Estimation, Action Recognition
![image](https://user-images.githubusercontent.com/48819383/147041909-b9e04cdd-d503-4847-adc1-8fbfd2e6e1e6.png)


## 2. 과제 수행방법
### 2.1. 과제수행을 위한 도구적 방법
- Geforce 3090 GPU를 이용해 고속병렬연산을 수행
- Yolov3 모델을 이용한 Object Detection
- DeepLabCut 툴을 이용한 Pose Estimation
- MMAction2 툴을 이용한 Action Recognition
- Heatmap을 사용한 딥러닝 모델 연구 논문 조사
- 성능평가방법 조사

### 2.2. 과제 수행 절차
![image](https://user-images.githubusercontent.com/48819383/147036207-5745153e-2ac6-4ea3-8235-dfec97572483.png)

### 2.3. 절차별 세부 내용
1. 유저 데이터 정제
    - 비식별화 처리된 영상을 가져와 행동별 라벨링을 진행한다.
    - 영상을 프레임별로 extract하여 특정 행동(bodyscratch, bodyshake, turn)이 드러나는 구간에 해당하는 10개의 프레임을 선별한다.
<img src = "https://c.tenor.com/479psZjG3V4AAAAM/bea-dog.gif" width ="150" height = "150"/> <img src = "https://cdn.theatlantic.com/media/mt/science/doggif.gif" width ="150" height = "150"/> <img src = "https://c.tenor.com/GPbjPPV9GrgAAAAM/oskar-boston-terrier.gif" width ="150" height = "150"/>

        (사진) 각 행동별 예시. bodyscratch(left), bodyshake(center), turn(right)

2. Pose 데이터 전처리 방법 연구
    - 기존의 Pose 정보를 pickle 파일로 변환하는 과정을 거침
    - mmaction에서 제공하는 pickle 파일 구조에 관한 [설명](https://mmaction2.readthedocs.io/en/latest/supported_datasets.html#the-format-of-posec3d-annotations)을 참조하여 생성함.
<img src="https://user-images.githubusercontent.com/48819383/147040172-8038f5ac-7233-40cc-9a30-8a4e4f36f1bb.png" height="300px"/>

<details>
<summary>annotation pickle 파일 생성코드 예시</summary>
<div markdown="1">       

```python

# 훈련을 위해 annotation 파일을 생성하는 로직을 만듦
tot_anno = list()

for k in range(14):

    file = "/home/pepeotalk/Downloads/210915-userdataset-11-resized/test/bite/dnd9463_20210825_154951_2/dnd9463_20210825_154951_2DLC_resnet152_TrainTestJul2shuffle1_1030000.csv"

    csv_file = pd.read_csv(file,
                          skiprows = 2)
    kp_coord_array_one_frame = np.zeros((28, 2),
                            dtype=np.float16)
    kp_score_array_one_frame = np.zeros((10, 1),
                            dtype=np.float16)
    kp_coord_array_one_video = np.zeros((10,28, 2),
                            dtype=np.float16)
    kp_score_array_one_video = np.zeros((10,10, 1),
                            dtype=np.float16)


    for i in range(len(csv_file)):
        for j in range(1,len(csv_file.columns)):
            if j%3 == 1:
                kp_coord_array_one_frame[j//3][0] = csv_file.iloc[i][j]
                kp_coord_array_one_frame[j//3][1] = csv_file.iloc[i][j+1]
            elif j%3 == 0:
                kp_score_array_one_frame[i] = csv_file.iloc[i][j]
        # print(kp_coord_array_one_frame)
        kp_coord_array_one_video[i] = kp_coord_array_one_frame
        kp_score_array_one_video[i] = kp_score_array_one_frame


    anno = dict()
    anno['keypoint'] = np.array([kp_coord_array_one_video], dtype=np.float16)
    anno['keypoint_score'] = np.array([kp_score_array_one_video], dtype=np.float16)
    anno['frame_dir'] = file
    anno['img_shape'] = (1080, 1920)
    anno['original_shape'] = (1080, 1920)
    anno['total_frames'] = 10
    anno['label'] = 0

    tot_anno.append(anno)
    mmcv.dump(anno, f'/home/pepeotalk/Downloads/210915-userdataset-11-resized/pose/part_{str(k)}.pkl')
mmcv.dump(tot_anno, f'/home/pepeotalk/Downloads/210915-userdataset-11-resized/pose/result.json')
```

</div>
</details>

3. 학습 데이터 고려하여 모델 학습
- object Detection 진행 (이미지 처리) : zero padding, crop and resize 방식 두 가지로 나누어 모델 학습결과를 비교함 
<img src="https://user-images.githubusercontent.com/48819383/147041938-1a5bad37-591b-45ae-bdd5-e43215698f13.png" height="350px"/>

- 데이터셋 구축
<img src ="https://user-images.githubusercontent.com/48819383/147039119-0d584a2f-c882-466c-bb00-dcdd797603be.png" height="350px"/>
- 모델 config 파일 생성 ( 사용할 모델 구조, input shape, transform방식 등 모델 init, train, test에 필요한 모든 정보들을 포함한 파일)
- 훈련 시작

```bash

$ ./tools/dist_train.sh [CONFIG FILE PATH] [GPU NUM]

```

5. 성능평가
- 이미지 처리 방식(zero padding, crop and resize)에 따라 성능이 어떻게 변화하는지 실험한다. 
- config 파일에 지정해둔 test pickle 파일로 성능을 평가할 수 있다.

```bash

$ python tools/test.py [CONFIG FILE PATH] \
    [CHECKPOINT FILE PATH] --eval top_k_accuracy mean_class_accuracy \
    --out result.json

```


## 3. 수행결과
### 3.1. 과제 수행 결과
- Aihub 데이터와 사용자 데이터를 합한 데이터셋을 구축함. 아래의 표는 데이터셋의 분포를 나타냄.
<img src="https://user-images.githubusercontent.com/48819383/147040752-2fe65914-d623-4774-baaa-8bd0859763aa.png" height="150px"/>

- 최종 성능
   - top-1-acc: 0.8889
   - mean-class-acc: 0.8700
   - top-1-acc로 비교한 이미지 처리 방식별 성능
   <img src="https://user-images.githubusercontent.com/48819383/147042081-e325267f-0651-456f-9804-29574e1f3fac.png" height="300px"/>


### 3.2. 최종결과물 주요특징 및 설명
<img src="https://user-images.githubusercontent.com/48819383/147040854-4dc44396-3adc-4ec6-bbdd-bcdd6a41fbaf.png" height="150px"/>

본 연구에서는 강아지 행동인식을 위해 총 3단계를 거쳤다.
  1. Object Detection을 통한 ROI 검출
  2. Pose Estimation을 통한 차원 축소
  3. 추출된 Pose 정보를 활용한 Action Recognition

CCTV 환경의 영상이기 때문에 전체 영역에 강아지에 해당하는 영역이 적다는 한계점을 해결하기 위해 Object Detection 단계가 추가되었다. Yolov3를 이용해 강아지 영역만 가져올 수 있게 하였다. 또한 Pose 정보로 inference하기 때문에 차원 축소가 되어 더욱 빠른 처리가 가능하다는 이점이 있다.


## 4. 기대효과 및 활용방안
- CCTV 등에서 강아지의 행동을 세분화하며 분석할 수 있게 하여 원격진료, 반려견 건강 모니터링을 가능하게 한다. 이후에 분석자료를 활용하여 앱 사용자에게 반려견에 대한 객관적인 정보를 제공할 수 있게 한다. 또한 이러한 접근법은 데이터 확보의 어려움이 있을 때 성능을 개선하기 위해 도움이 될 것이다.
- Object Detection, Pose Estimation, Action Recognition의 3단계를 한 번에 관리할 수 있는 코드를 생성하고 불필요한 로직을 제거하여 실제 프로덕션에 적용한다. 분석 정보는 DB서버로 보내 관리할 수 있게 한다.

## 5. 결론 및 제언
본 연구에서는 Object Detection, Pose Estimation, Action Recognition이라는 서로 다른 분야를 적용하여 강아지의 행동인식을 진행했다. 선별된 3개의 행동들(몸을 긁는 행동, 몸을 터는 행동, 빙빙 도는 행동)은 모두 수의학적으로 유의미한 가치가 있는 정보들이다. 따라서 해당 행동들을 구분할 수 있다면 사용자의 강아지 건강관리에 더욱 도움을 줄 수 있다. 또한 후속 연구로는 오디오 인식 모델을 확용하여 강아지의 행동인식을 더욱 구체화하는 방식이 있다.

