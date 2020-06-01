# infoGAN_MRI
Interpretable Feature Learning of MRI Data using Generative Adversarial Networks for Alzheimer’s Disease Diagnosis - 전민경, Oumate Youssef, 김성찬

### 동기
* 분류 결과를 시각화하기 위한 기존 방법
  - 모델이 진단할 때 주로 보는 feature를 알 수 있음
  - 왜 그렇게 판단했는지 해석하는 것은 여전히 어려움
* Generative Adversarial Network(GAN) 사용
  - class에 따라 feature가 어떻게 변하는 지에 대한 꼬이지 않은 representation을 학습
  - 알츠하이머 진단을 위한 분류 모델의 기준에 대한 해석 가능한 feature 학습
  

### 데이터 
* Alzheimer’s Disease Neuroimaging Initiative (ADNI)-1
  - standardized MRI 데이터셋 중 screening 데이터 사용
  - 정상(NC) : 228개
  - 알츠하이머 환자(AD) : 188개
* 공간적 정규화(SPM v12 이용) 후 axial 측면의 37번 째 슬라이스 이용
![스크린샷, 2020-06-01 17-42-55](https://user-images.githubusercontent.com/25657945/83391996-5a158780-a42f-11ea-88c8-0aca0084c818.png)

### 모델 구조 
* MRI-infoGAN model
  - 목표 : disentangled representation 학습
  - 오리지널 InfoGAN과 다르게 MRI-infoGAN모델은 continuous code만 사용
  ![스크린샷, 2020-06-01 17-36-05](https://user-images.githubusercontent.com/25657945/83391879-2175ae00-a42f-11ea-9fff-f99669d022ed.png)
