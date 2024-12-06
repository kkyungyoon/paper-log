## [논문정리]Auto-Encoding Variational Bayes (2013), Diederik P Kingma, Max Welling

### Problem statement
- 연속적인 잠재 변수와 계산 불가능한 사후 분포가 있는 경우에도, 대규모 데이터셋에서 효율적으로 확률 모델에서 추론 및 학습을 수행
    - 연속적인 잠재 변수 : 각 이미지마다 해당 이미지의 특성을 나타내는 하나의 잠재 변수 존재한다고 가정하면, 잠재 변수들은 연속적인 값으로 변할 수 있음
    - 계산 불가능한 사후 분포 : 데이터 x가 주어졌을 때, 잠재 변수 z의 분포
- 변분 하한을 최적화하기 위한 경사 계산의 문제 제기(높은 분산을 가지는 몬테카를로 경사 추정방식을 대체할 방법 필요 (-> 추후에 SGVB 제안))
    - 변분 추론에서는, p(x)를 직접 계산하지 않고, 변분 하한을 통해 근사함
    - 변분 하한 (Variational Lower Bound, ELBO) : Marginal Likelihood를 Lower Bound로 근사하는 값, VAE의 손실 함수     
          ![image](https://github.com/user-attachments/assets/54872add-1d95-42bf-a8a0-252e2013fff9)   
          (위 그림에서 L이 변분하한)
          ![image](https://github.com/user-attachments/assets/5142a1ad-5ff6-4989-b9a2-969bc6f6284d)   
          (변분 하한은, Marginal Likelihood를 KL Divergence로 분해하여 유도됨. KL Divergence는 항상 0 이상이므로, L이 하한이 됨)   
          ![image](https://github.com/user-attachments/assets/b17a6b1b-53f8-4fa6-9942-d3d9876a3790)
          (KL Divergence항을 이항하면, Reconstruction Loss, KL Divergence Loss의 항으로 구성된 VAE의 Loss Function 볼 수 있음)
    - 변분 하한을 최대화하면, 근사 사후 분포 q(z|x)가 실제 사후 분포 p(z|x)에 가까워짐
<br>

### Solution approach
1) 변분 하한을 Reparameterization하여, 표준 확률적 경사 하강법으로 최적화할 수 있는 하한 추정치를 도출
   - SGVB(Stochastic Gradient Variational Bayes) 추정치 : 변분 하한을 Reparameterization하여 변분 하한의 미분 가능하고, 편향되지 않은 추정치
   - SGVB는 연속적인 잠재 변수를 가진 거의 모든 모델에서 효율적 근사 사후 추론에 사용 가능 (표준 확률적 경사 상승 기법을 사용해 최적화 가능)
     - 표준 확률적 경사 상승 기법 : 확률적 경사 하강법(SGD)를 변분 추론의 ELBO 최적화에 적용한 것
     - ELBO : 최적화하려는 목표 (ELBO를 최대화하여 q(z|x)가 p(z|x)와 가까워지도록) 
     ![image](https://github.com/user-attachments/assets/13e73c30-11c0-409c-8ff5-99f0aca6b70e)
     - SGD 사용 : 모든 데이터에 대해 ELBO 기댓값 계산하는 것은 비효율적. 소규모 mini batch를 사용해 기댓값을 근사
       
2) i.i.d.를 따르는 데이터 셋에서, 데이터 포인트 별로 연속적인 잠재 변수가 있는 경우, 제안된 하한 추정치를 사용해 '계산 불가능한 사후 분포'를 근사하기 위한 Recognition Model(q(z|x))을 학습시킴으로써 사후 추론을 효율적으로 수행
   - Auto Encoding VB(AEVB) 알고리즘 제안 (각 데이터 포인트마다 반복적인 추론 (ex, MCMC)을 수행하지 않고도 모델 매개 변수 효율적 학습 가능
   - 과정 : 파라미터 초기화 -> 반복(데이터 포인트 선택 -> 𝜀 분포로부터 random sample 선택 -> Gradients of mini batch estimator -> 파라미터 업데이트(SGD, Adagrad 사용))

<br>

### Conclusion
- SGVB 추정치는 표준 확률적 경사 하강법을 이용해 미분 및 최적화 가능하다.
- i.i.d. 데이터셋과 데이터 포인트당 연속적인 잠재 변수를 가지는 경우, SGVB 추정치를 사용하여 근사 추론 모델을 학습하는 효율적인 추론 및 학습 알고리즘인 **Auto-Encoding VB(AEVB)** 를 소개
   	  
<br>

### Strong points


<br>

### Weak points


<br>

### Questions


### New ideas / Comments
