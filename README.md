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
![스크린샷, 2020-06-01 17-43-49](https://user-images.githubusercontent.com/25657945/83392072-787b8300-a42f-11ea-9526-360a2a859ed0.png)


### 모델 검증
* 검증 순서   
![스크린샷, 2020-06-01 17-45-44](https://user-images.githubusercontent.com/25657945/83392271-c6908680-a42f-11ea-88ce-b2c1d25c064b.png)

     
![스크린샷, 2020-06-01 17-47-06](https://user-images.githubusercontent.com/25657945/83392358-ee7fea00-a42f-11ea-90c3-64e89a9a9817.png)


### 결과
* 생성된 이미지
  - 목표 : continuous latent code가 어떤 조합에서 disentangled representation을 학습하는지 찾는 것
  - MRI의 feature는 독립적으로 disentangled representation을 학습하지 않음
  - c1, c3 은 특정 조합에서 disentangled representation을 보여줌   
![스크린샷, 2020-06-01 17-49-44](https://user-images.githubusercontent.com/25657945/83393508-de690a00-a431-11ea-94bf-06d13e4e53a5.png)
  - c1과 c3의 조합에 따라 생성된 이미지   
  ![스크린샷, 2020-06-01 17-49-58](https://user-images.githubusercontent.com/25657945/83393511-df9a3700-a431-11ea-8ef2-7114f4646a34.png)


### 결론
* 본 논문에서는 MRI 데이터의 disentangled representation을 배우기 위해 InfoGAN을 확장할 것을 제안했다.
• 제안된 모델은 MRI의 분류에 대한 disentangled representation을 학습했다.
• 제안된 모델은 MRI 기반 알츠하이머병 진단 시 모델의 결정을 해석하는 데 도움이 되는 의미 있는 특징들을 학습했다.
• 제안된 모델은 의료 영상을 이용한 다른 질병 진단에 적용될 수 있으며, 여기서 중요한 특징이 무엇인지 해석하고 시각화 한다.
