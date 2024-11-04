## [논문정리]Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015), Radford et al.,

### Problem statement
라벨이 없는 데이터셋으로부터 GAN을 사용하여 의미 있는 특징을 학습하고, 이 학습된 표현을 이미지 생성뿐만 아니라 이미지 분류와 같은 다양한 downstream task에 적용할 수 있는 비지도 학습 방법을 구현  

### Solution approach
기존 GAN과 달리, DCGAN은 Convolution layer, Batchnorm, Activation function을 조정해서 기존 GAN의 문제점을 해결
- 기존 GAN의 문제점
	- 낮은 해상도
	- 훈련 불안정성
- 해결 
	- Fully Connected layer 없이, Convolution layer만을 사용하여 모델의 복잡성을 줄이고 학습 효율을 높임
	- CNN을 generator에서 fake image 생성에 쓰기 위해, Fractional-Strided Convolution을 활용하여 Upsampling 과정을 학습함
	- Generator의 마지막 레이어, Discriminator의 첫 번째 레이어를 제외한 부분에, Batch normalization을 적용해 학습을 안정화함

### Conclusion
- GAN의 훈련 불안정성 문제 완화
- 이미지 생성 품질 개선
### Strong points
- 비지도 학습의 가능성
	라벨 없이도 고차원적 이미지를 표현할 수 있는 능력을 보이며, DCGAN 구조가 비지도 학습에 효과적임을 제시
- 구조의 단순화
	Fully Connected layer를 제거함으로써 모델이 더 단순화되고, 이로 인해 학습 과정이 안정적이며 효율적임
- 모델 설명의 명확성
	모델 구조와 학습 절차를 명확하게 설명

### Weak points
- 데이터셋 의존성
	특정 데이터셋에 의존할 수 있으며, 새로운 도메인의 데이터셋에서는 동일한 성능을 기대하기 어려울 수 있음
- 실험적 한계
	이미지 표현 학습의 성능을 정량적으로 평가하는 실험이 부족합니다.

### Questions
다른 Representation Learning 기법과 DCGAN을 결합해볼 수 있을지

### New ideas / Comments
