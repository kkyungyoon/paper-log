### Result
Generator를 통과해 생성된 이미지를 출력한 결과입니다.
| 0 Epoch        | 1 Epoch       | 10 Epoch        | 20 Epoch        |
|----------------------|--------------------------|-----------------------|-----------------------|
|![0](https://github.com/user-attachments/assets/df118543-0da0-4670-8614-d67314081a02) |![469](https://github.com/user-attachments/assets/97e27abe-4f38-4caf-ad8c-fedccca73ec3) | ![5159](https://github.com/user-attachments/assets/c9486bec-8309-40a0-9a09-fbd7447c8932) |![8911](https://github.com/user-attachments/assets/b11d2289-fd16-4350-bb82-696fba3d89dc) |

<br>

## [논문정리]Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015), Radford et al.,

### Problem statement
- 지도 학습에서 CNN의 성공과 Unsupervised Learning 간의 격차를 해소
- 특정 구조적 제약을 가진 CNN의 한 종류인 **심층 합성곱 생성적 적대 신경망(DCGAN, Deep Convolutional Generative Adversarial Networks)** 을 소개하며, 이를 Unsupervised Learning의 강력한 후보로 제안
- 라벨이 없는 Data Set으로부터 GAN을 사용하여 의미 있는 특징을 학습하고, 이 학습된 표현을 이미지 생성뿐만 아니라 이미지 분류와 같은 다양한 Downstream Task에 적용할 수 있는 Unsupervised Learning 방법을 구현  

<br>

### Solution approach
기존 GAN과 달리, DCGAN은 Convolution Layer, Batchnorm, Activation Function을 조정해서 기존 GAN의 문제점을 해결
- 기존 GAN의 문제점
	- 낮은 해상도
	- 훈련 불안정성
- 해결 
	- Fully Connected Layer 없이, Convolution Layer만을 사용하여 모델의 복잡성을 줄이고 학습 효율을 높임
	- Generator에서 Fake Image 생성에 CNN을 쓰기 위해, Fractional-Strided Convolution을 활용하여 Upsampling 과정을 학습함
	- Generator의 마지막 레이어, Discriminator의 첫 번째 레이어를 제외한 부분에, Batch Normalization을 적용해 학습을 안정화함

<br>

### Conclusion
- GAN을 훈련하기 위한 더 안정적인 아키텍처를 제안하고, Adversarial Networks가 Supervised Learning과 생성 모델링에 적합한 이미지 표현을 학습한다는 증거를 제공함
- GAN의 훈련 불안정성 문제 완화 및 이미지 생성 품질 개선
- 하지만 여전히 모델 불안정성의 일부 형태가 남아 있음 (예, 모델이 더 오래 훈련되면 일부 필터가 단일 진동 모드로 수렴하는 문제가 발생하는 경우가 존재)
	- 단일 진동 모드 : 필터가 입력 데이터의 특정 패턴(예: 단일 주파수나 색상, 텍스처 등)에만 반응하며, 이를 계속 반복적으로 활성화
   	- Mode Collapse : Generator가 다양한 샘플을 생성하는 대신 하나의 고정된 패턴만 생성함. 단일 진동 모드로 수렴은 이러한 모드 붕괴의 초기 징후
   	  
<br>

### Strong points
- 비지도 학습의 가능성
	라벨 없이도 고차원적 이미지를 표현할 수 있는 능력을 보이며, DCGAN 구조가 Unsupervised Learning에 효과적임을 제시
- 구조의 단순화
	Fully Connected Layer를 제거함으로써 모델이 더 단순화되고, 이로 인해 학습 과정이 안정적이며 효율적임
- 모델 설명의 명확성
	모델 구조와 학습 절차를 명확하게 설명

<br>

### Weak points
- 데이터셋 의존성
	특정 Data Set에 의존할 수 있으며, 새로운 도메인의 Data Set에서는 동일한 성능을 기대하기 어려울 수 있음
- 학습 불안정성
	기존 GAN에 비해서는 안정화됐지만, 여전히 불안정

<br>

### Questions


### New ideas / Comments
