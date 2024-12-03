### Result
Decoder를 통과해 생성된 이미지를 출력한 결과입니다.
| (좌) 0 Epoch        | (가운데) 100 Epoch       | (우) 500 Epoch        |
|----------------------|--------------------------|-----------------------|
| ![0 Epoch](https://github.com/user-attachments/assets/d79720ba-6717-4b3c-b74d-6c22e394b771) | ![100 Epoch](https://github.com/user-attachments/assets/d0478949-c447-4993-81de-bf823882976b) | ![500 Epoch](https://github.com/user-attachments/assets/e315ff68-e5c0-4650-9440-751ee326692a) |

Encoder를 통과해서 나온 Latent z를 이용해 t-SNE를 그린 결과입니다.  
<img src="https://github.com/user-attachments/assets/dd61eec6-d021-479b-92d6-1f90e1cede1d" alt="Latent z tSNE" style="width: 50%;"/>

Latent z의 확률분포가 Prior Distribution으로 설정한 표준정규분포에 근사했는지 확인한 결과입니다.  
<img src="https://github.com/user-attachments/assets/bfdfb362-010c-40bd-97b6-6313756956e9" alt="Latent z mean std" style="width: 50%;"/>

<br>

## [논문정리]Adversarial Autoencoders (2015), Makhzani, Alireza et al.,

### Problem statement
- 스케일 확장이 가능한 생성 모델을 구축하여 오디오, 이미지, 비디오와 같은 풍부한 분포를 캡처하는 것은 머신러닝의 중심 과제 중 하나
- Restricted Boltzmann Machines(RBM), Deep Belief Networks(DBNs), Deep Boltzmann Machines(DBMs)같은 심층 생성 모델은 주로 MCMC(Markov Chain Monte Carlo) 기반 알고리즘에 의해 학습되었음
- 하지만, Markov Chain의 샘플이 Mode 사이를 충분히 빠르게 섞지 못하기 때문에, MCMC 방법은 로그 가능도 기울기를 계산하는 데 사용될 때, 학습이 진행됨에 따라 점점 더 부정확해짐
- MCMC 학습과 관련된 어려움을 피하고 직접 Back-propagation으로 학습할 수 있는 생성 모델이 개발됨 : VAE, Importance weighted autoencoders, GAN

<br>

### Solution approach
- Autoencoder의 Encoder를 GAN의 Generator와 접목
- Autoencoder를 두 가지 목적에 따라 학습시킴
  - 전통적인 Reconstruction Error
  - Autoencoder의 Latent Representation의 Aggregated Posterior Distribution를 임의의 Prior Distribution에 맞추는 Adversarial Training Criterion
- Generative Adversarial Networks (GAN)을 활용하여, Autoencoder의 Latent Code Vector의 Aggregated Posterior를 임의의 Prior Distribution에 맞추는 방식으로 Variational Inference을 수행하는 Probabilistic Autoencoder

![image](https://github.com/user-attachments/assets/b28eeb2b-3c3d-4bc4-9b8e-14acf5ef4a31)

<br>

### Conclusion
- Generative Autoencoder로, MNIST와 Toronto Face 데이터셋에서 **경쟁력 있는 test likelihood**를 달성함
- AAE를 Semi-Supervised Learning에 확장하는 방법을 논의했으며, MNIST와 SVHN 데이터셋에서 경쟁력 있는 Semi-Supervised Learning 분류 성능을 보임

<br>

### Strong points
- VAE애서 Latent z를 Regularization하는 KL Divergence 함수보다, Discriminator를 학습하여 사용하면 복잡한 함수로써 기능하기에 Encoder를 더 정교하게 Update가 가능함
- 이로 인해, 데이터 기반으로 자유롭게 목표 분포 학습 가능함

<br>

### Weak points
- GAN 특유의 문제가 남아있음
- Mode Collapse, 학습 불안정성, 추가 계산비용의 문제
  
<br>

### Questions
- Input x가 AAE에서는 한 번에 Encoder에 입력되어 mu, sigma가 계산되는데, 이를 N개의 집단으로 나누어 각 집단을 각각 Encoder를 통과시켜 mu, sigma를 뽑아  Discriminator에 독립적으로 학습을 진행 후 결과를 앙상블하면, Input 집단에 포함되어있는 이상치의 영향을 덜 받을 수 있지 않을까?
