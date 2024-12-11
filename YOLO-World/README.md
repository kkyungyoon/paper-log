## [논문정리]YOLO-World: Real-Time Open-Vocabulary Object Detection (2024), Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan

### Background Knowledge
1️⃣ Object detection 방식(어떻게 bbox와 label을 예측하는지에 따라 구분됨)
1) Region based methods : Faster R-CNN, R-CNN   
    - Two-stage 방식을 채택   
    - 첫 번째 단계에서, Proposal 생성 수행   
    - 두 번째 단계에서, RoI 단위로 분류, 회귀 수행
    
    (✅장점) 정확도 높다. 대형 모델에 적합하고, 고해상도 객체 인식에 주로 사용된다.   
    (❌단점) 속도 느리다.   

2) Pixel based methods : RetinaNet, SSD
    - One-stage 방식을 채택   
    - 고정된 앵커박스나 픽셀 단위로 클래스 분류와 경계박스회귀를 동시에 수행   

    (✅장점) 속도 빠르다. 계산 비용 낮다.   
    (❌단점) 정확도는 Region-based 방식보다 낮다. 실시간 검출이 중요한 어플리케이션(ex, 자율주행)에 자주 사용된다.   
  
3) Query based methods : DETR, Deformable DETR
    - Transformer 모델을 활용한 객체 검출 방식
    - 입력 이미지의 특징을 Transformer의 쿼리로 변환하고, Transformer의 self-attention 매커니즘을 통해 객체의 위치, 클래스를 예측한다.   
  
    (✅장점) 간단하고 직관적인 구조다. 기존의 CNN 기반 방법과 차별화되어 있으며, Tranformer 아키텍처의 장점을 활용한다.   
    (❌단점) 훈련에 많은 시간이 필요하다.

4) YOLO 방법 : YOLOv1 ~ YOLOv8, PP-YOLO
    - 간단한 CNN 아키텍처를 활용하여, 한 번의 Forward pass로 이미지의 모든 객체를 예측한다.   
    - 고정된 크기의 그리드 셀로 이미지를 나누고, 각 셀에 대해 객체의 클래스와 경계박스 위치를 동시에 예측한다.   
  
    (✅장점) 실시간 객체 검출이 가능하여 자율주행, CCTV 등 실시간 어플리케이션에서 사용된다.   
    
    - 속도와 정확도의 균형을 유지하기 위해, 다양한 네트워크 구조와 기법들이 제안되었다.   
    > #### 📌 속도 향상 기법   
    > ∙ PAN(Path Aggregation Network) : 여러 피처 맵을 효율적으로 결합하여 더 나은 표현 학습 수행   
    > ∙ CSPNet(Cross-Stage Partial Network) : 모델의 중복 계산을 줄이고 효율성을 높인다.   
    > ∙ Re-Parameterization 기법 : Train 단계와 Inference 단계를 분리하여, Inference 중에 모델의 효율성을 향상시킨다.

<br>

2️⃣ 기존 OVD(Open Vocabulary Object Detection) vs Previous OVD vs YOLO-World
![image](https://github.com/user-attachments/assets/dc3f27f4-d6ea-4829-887b-43a346e4d0c5)   

1) 기존 OVD는, 기존의 base 클래스로 detector를 학습하고, 알 수 없는(novel) 클래스에 대해 평가한다.    
    - 기존의 객체 검출기들은, 사전에 정의된 고정된 어휘에 기반하여 사전학습된 카테고리로만 검출을 수행
    - (예시) COCO 데이터 셋의 고양이, 개, 자동차로 학습했을 때, 학습에 포함되지 않은 코끼리가 주어졌을 때, zero shot 방식으로 새로운 객체를 검출 할 수 있는지를 평가한다.
    - (❌한계점) 다양한 도메인에 일반화 능력 떨어짐 (ex, COCO 데이터 셋에서 학습된 모델 -> 의료영상 도메인에서 성능 떨어짐)

2) Previous Open Vocabulary Detectors
    - (핵심 아이디어) 이미지 - 텍스트 관계를 학습하여, 대규모 Image-text pairs를 활용해, 더 큰 범위의 어휘로 검출 모델을 훈련하는 것
    - 최근 OVD 접근 방식은, OVD를 이미지-텍스트 매칭 문제로 재정의했다.
    - 사용자의 prompt를 텍스트 인코더로 인코딩하여, Online vocabulary를 생성한 후, 객체를 검출한다.
    - 이미지와 텍스트를 동시에 인코딩하여 예측을 수행한다.
    - (❌한계점) 이미지-텍스트를 동시에 인코딩하므로 계산 비용이 높고, 실용성이 부족하다. 무거운 Backbone을 사용한다.
    - 대표적인 연구들 : OWL-ViTs, GLIP, Grounding DINO

3) YOLO-World
    - 경량 YOLO 프레임워크 기반
    - Prompt-then-detect 패러다임 도입
      > #### 📌 용어  
      > Online Vocabulary : 모델이 Inference 중에 실시간으로 prompt를 인코딩하여 어휘를 생성하는 방식   
      > Offline Vocabulary : Inference 전에 사전 정의된 어휘를 미리 인코딩하고 저장해두고, Inference 중에는 인코딩 없이 재사용하는 방식   

      > #### 📌 순서
      > 사용자가 필요에 따라 Prompt를 생성한다.   
      > Prompt는 Offline vocabulary로 인코딩된다.   
      > 이 Offline vocabulary는 re-parameterization 되어 모델의 가중치로 변환된다.   
      > 이후 Inference 단계에서는 텍스트 인코더 없이 빠르게 예측할 수 있다.   
  
      > #### 📌 특징
      > prompt를 미리 인코딩하여 Offline vocabulary로 만든다.   
      > 한 번 생성된 어휘는 다시 인코딩할 필요가 없으며, detector가 이를 즉시 추론에 활용 가능하다.   
      > (✅장점) 실시간 추론이 가능하고, 엣지 디바이스에도 적합하다, 텍스트 인코더를 Inference 도중에 호출하지 않으므로 기존방식보다 훨씬 빠르다, Prompt를 사용해 사용자 맞춤형 어휘를 쉽게 추가할 수 있다.   

### Problem statement
- YOLO는 사전 정의되고 학습된 객체 카테고리에 의존하기 때문에 개방형 환경에서는 그 활용에 제한이 있다.  
  > #### 📌 사전 정의되고 학습된 객체 카테고리
  > 학습할 때, 특정한 클래스에 대해서만 학습. 즉, 학습하지 않은 새로운 객체는 인식하지 못 한다.  
  > 예를 들어, COCO 데이터 셋에서는 80개의 카테고리로 객체를 정의하고, 이 카테고리만 학습하도록 되어 있다.  
  > 한 번 객체의 카테고리(클래스)가 정의되고 레이블링이 완료되면, 학습된 검출기는 그 특정한 카테고리만 인식할 수 있다.    
  > 즉, 학습하지 않은 새로운 카테고리는 검출할 수 없기 때문에, Open vocabulary 시나리오에서는 활용에 제한이 생긴다.
  
- 최근 연구들은, BERT 같은 언어 인코더로부터 어휘 지식을 distillation하는 방법을 통해, Open vocabulary detection 문제를 해결하기 위해, Vision language model을 활용했다.
  그러나, Knowledge distillation based 방법들은, 학습 데이터의 부족, 어휘의 다양성이 제한적이므로 한계를 보인다.
  
- 몇몇 연구들은, Object detection 학습을 Region level vision language pretraining으로 재구성하고, 대규모 데이터 셋에서 Open vocabulary 객체 검출기를 훈련하는 법을 제안했다.
  그러나 Real world detection에서 막대한 계산 비용, 엣지 디바이스(ex, 모바일 장치)에 배포하는 과정이 복잡함으로 어려움을 겪는다.
  Large detectors의 사전 학습을 통해 성능 입증했으나, Small detectors에 Open recognition 능력을 부여하기 위한 사전학습에 대한 연구는 충분히 이뤄지지 않았다.

<br>

### Solution approach - Method
1️⃣ YOLO World Model Architecture

![image](https://github.com/user-attachments/assets/497b240d-579e-4835-a468-abaaf0fde044)

📂 Text Encoder    

  - 입력 텍스트를 CLIP으로 사전학습된 Transformer 기반 Text Encoder를 사용하여 Text Embedding 추출 : W   
  - 사용자의 prompt를 Text embedding으로 변환      
  - 이 임베딩은 오프라인 어휘로 저장되어 이후 모델 가중치로 재구성   

  > #### 📌 CLIP Text Encoder의 특징  
  > 1) 텍스트 전용 언어 인코더 BERT에 비해, 시각적 객체와 텍스트를 연결하는 능력이 더 뛰어남  
  > 2) 입력 텍스트가 캡션, 참조표현인 경우, n-gram 알고리즘을 사용해 명사구를 추출하여, 명사구를 text encoder에 입력하여 임베딩 생성  
     
📂 Image Encoder   
  - 입력 이미지를 Multi-scale image features로 변환   
  - YOLO 백본을 사용해 각 계층에서 feature map을 추출
  - YOLOv8기반으로 개발됨
    
  > #### 📌 YOLOv8의 구조
  > Darknet Backbone : 이미지 인코더로 사용. 입력이미지에서 다양한 수준의 Feature 추출   
  > PAN(Path Aggregation Network) : Multi scale Feature Pyramid 생성, 다양한 계층의 특징을 통합하여 객체 검출 성능을 강화   
  > Head : Bbox Regression(객체의 위치를 예측), Object Embedding(객체의 텍스트 임베딩과 매칭하기 위한 임베딩 벡터 생성)   
      
📂 Vision-Language PAN(RepVL-PAN : Re-parrameterizable Vision-Language Path Aggregation Network)    
  ![image](https://github.com/user-attachments/assets/378da8b9-c896-4c29-9ab7-a1d2c2284b36)
  > #### 📌 이미지 설명
  > 상단 컬러 토큰 : 텍스트 임베딩  
  > C3, C4, C5 : 멀티 스케일 이미지 feature map  
  > P3, P4, P5 : 최종 출력 feature map (YOLO World head로 전달되어 객체검출, 텍스트 매칭에 사용됨)  

  📂 PAN : Text-guided CSPLayer   
  - 텍스트 정보를 이미지 특징에 주입하는 과정   
  - 기존의 CSPNet은 이미지 특징의 효율적 융합을 목표로 하지만, (기존 YOLOv5, v8등에서 사용되며, 입력 특징을 절반으로 나눠 병렬 경로로 처리한 후 다시 결합하는 것)   
  - T-CSPLayer는 추가적으로 텍스트 임베딩을 통합하여 텍스트 기반의 weight 조정, feature 조정 수행   
  - 이미지 feature map에 텍스트 임베딩을 함께 입력하면 텍스트 정보를 기반으로 특정 이미지 특징을 강조하거나 억제하여, 텍스트 의미와 연관된 이미지 특징을 더 잘 나타내도록 학습   
  - (예시) prompt로 dog가 입력되면, 이미지의 dog 관련 특징을 더 강조하고, 비관련특징을 억제함   

  > #### 📌과정   
  > <img src="https://github.com/user-attachments/assets/bbfa2e25-eefd-4529-8aad-3cfb751bfb02" width="600px">     
  
  > 1) YOLO의 이미지 특징과 텍스트 임베딩 W를 dot product하여 similarity map을 생성  
  > 2) 모든 텍스트 임베딩 W1, ..., Wc에 대해 최대값 선택 = 모든 텍스트 임베딩에 대해 가장 관련성 높은 텍스트 선택 = 각 픽셀 위치에서 가장 관련성 높은 명사 선택  
  > 3) 시그모이드 = 유사도 값을 [0, 1]로 정규화 = 이미지 특징에 곱해질 가중치 역할  
  > (예시) 이미지 특징 = 강아지, 고양이, 사람이 포함된 이미지 특징  
  > 텍스트 프롬프트 = dog, cat, person  
  > 이미지 특징과 dog의 유사도 계산, 이미지 특징과 cat의 유사도 계산, 이미지 특징과 person의 유사도 계산(방향이 유사하다면 내적값이 크게 나오고, orthogonal하거나 반대방향이면 내적값이 작거나 음수)  
  > 각 픽셀에서 가장 높은 유사도를 가지는 텍스트 선택 후 시그모이드로 가중치 적용  

  <br>
  
  📂 PAN : Image Pooling Attention   
  - 텍스트 임베딩을 이미지 특성에 맞게 조정하는 과정
  - 이미지 특징을 global pooling 수행하여 global image representation 생성
  - global image representation은 텍스트 임베딩의 가중치 조정에 사용됨
  - 결과적으로 이미지에 민감한 텍스트 임베딩 생성
  <img src="https://github.com/user-attachments/assets/c6d63c19-6c0e-44b1-aec8-ff780a40939b" width="600px">

  - 기존의 텍스트 임베딩 W + 이미지 패치 X tilda와의 상호작용을 반영한 새로운 정보 = 업데이트된 텍스트 임베딩(이미지의 전역 정보를 반영한 텍스트 표현)   
  - 텍스트 임베딩 W : Query   
  - 이미지 패치 X tilda : Key, Value   
  - 텍스트 임베딩에 대해 R x R개의 이미지 패치가 MHA으로 상호작용   
    
  > #### 📌 이미지 패치 X tilda 생성과정
  > 멀티스케일 특징 X를 R x R 크기의 작은 영역으로 분할   
  > 각 영역에 대해 max pooling 적용   
  > 스케일 S에 대해 모두 이 과정을 수행(S : YOLO의 계층 수))   

    
  📂 PAN : 동작과정   
  1. Input : 이미지 인코더에 의해 생성된 멀티스케일 특징, 텍스트인코더에서 나온 텍스트 임베딩   
  2. T-CSPLayer : 이미지 feature와 텍스트 임베딩을 결합하여 이미지 feature를 조정   
     (효과) 특정 텍스트와 관련된 이미지 feature가 강조 또는 억제됨   
     (Output) 텍스트로 조정된 이미지 특징   
  3. I-Pooling Attention : 이미지 feauture를 global pooling하여 이미지 전역표현생성 -> 텍스트 임베딩 가중치 조정에 사용   
     (효과) 이미지 인식과 관련된 텍스트 임베딩 강화    
     (Output) 이미지에 민감한 텍스트 임베딩  

  📂 PAN : PAN의 역할   
  - Multi-level cross-modality fusion을 수행 : 이미지 특징과 텍스트 특징을 결합하여 시각적인 의미 표현을 강화   
  - Region-Text Matching : 객체의 이미지 임베딩과 텍스트 임베딩을 매칭   
  - 텍스트 특징과 이미지 특징을 연결하여 더 나은 Visual-Semantic Representation 가능하게 한다.   
  - Inference 단계에서는 텍스트 인코더를 제거할 수 있다.   
  - 사전 학습된 텍스트 임베딩을 RepVL-PAN의 가중치로 Re-parameterization하여 효율적인 배포가 가능하다.   

📂 Prediction   
  - Bounding Box Regressor : 객체의 Bbox를 예측   
  - Text Contrastive Head : 텍스트 임베딩과 매칭된 객체를 분류   
    > #### 📌 Text contrastive head   
    > YOLO World에서는 분리된 헤드를 채택   
    > 2개의 3 x 3 convolution layer를 사용해, 경계상자 bk, 객체임베딩 ek를 예측함(k: 객체의 개수)   
    > Text contrastive head는, 객체 임베딩과 텍스트 임베딩간의 유사도 s를 계산 -> 이걸 리턴함(ex, K개의 객체, C개의 텍스트 -> K x C의 유사도행렬)   
    > ![image](https://github.com/user-attachments/assets/45b64754-a59d-40a1-93a0-ca1770f84fbe)   
    > 학습 가능한 계수 alpha(스케일링 계수), beta(shifing 계수)를 추가 = Affine Transformation(선형변환)   
    > -> L2 정규화와 선형변환은 Region-text 학습의 안정성 높임   


<br>

2️⃣ YOLO World의 동작 과정     
- Train : Prompt에서 명사를 추출 -> 텍스트를 인코딩하여 Text Embedding 생성   
  > #### 📌 Training with online vocabulary   
  > YOLO World는 train동안, mosaic sample에 대해 online vocabulary T를 생성    
  > #### 📌 과정    
  >  - 긍정명사 추출 : 모자이크 이미지에 포함된 명사    
  >  - 부정명사 샘플링 : 모자이크 이미지에 실제로 나타나지 않은 객체 명사 무작위 샘플링    
  >  - 최대어휘크기 M : default=80   
  > #### 📌 목적   
  >  - 긍정명사 : 실제로 존재하는 객체와 관련된 텍스트 제공하여 정확한 매칭을 학습하도록 도움   
  >  - 부정명사 : 잘못된 객체-텍스트 매칭을 방지하도록 학습 안정화

- Deployment : Prompt로 오프라인 어휘 생성 -> 이 어휘를 모델의 가중치로 재구성하여 추론에서 활용   

- Inference : 입력 이미지와 오프라인 어휘를 활용해, Bbox와 객체를 빠르게 예측   
  > #### 📌 Inference with offline vocabulary   
  > 효율성을 극대화하기 위해 offline vocabulary을 활용한 Prompt-then-Detect 전략
  >
  > #### 📌 과정
  > - 사용자 prompt 입력   
  > - 텍스트 인코더를 통해 텍스트 임베딩으로 인코딩   
  > - 해당 임베딩은 offline vocabulary로 저장   
  > - 추론 전에 미리 수행되며, 한 번만 인코딩하면 됨   
  > - offline vocabulary를 모델의 weight로 re-parameterization하여 추론 중 추가적인 인코딩 작업 생략   
  >   추론 시에는 텍스트 인코더 호출하지 않고, 미리 계산된 오프라인 어휘 임베딩 사용   
  > - 입력 이미지에 대해 YOLO 검출기를 사용해 객체의 bbox와 임베딩을 예측하고, 객체임베딩과 offline vocabulary 임베딩의 유사도 점수를 계산하여, 가장 유사한 명사를 객체에 할당   

<br>

3️⃣ Region-Text Contrastive Loss   
- YOLO 검출기의 Open vocabulary pretraing을 위해, Region text contrastive learning을 대규모 데이터 셋에서 수행한다.    
- Detection data, Grounding data, Image text data를 Region text pairs로 통합한다.   
- Region text pairs가 풍부하게 사전학습된 YOLO-World는 대규모 어휘 검출에 강한 능력을 보인다.   
![image](https://github.com/user-attachments/assets/7bbc6ef2-8f02-400f-b6c9-3fcb3a7d8383)   
> #### 📌 Loss 설명     
> Lcon(Region-text contrastive loss) : 텍스트와 객체간의 매칭을 학습 (예측된 객체와 텍스트 간의 유사도점수 계산 -> 예측된 유사도 점수와 실제 라벨 비교해서 cross entropy loss 계산)   
> Liou, Ldfl(distributed focal loss) : 정확한 bbox 회귀 학습(Liou : 정확한 bbox 학습, Ldfl : 라벨 불균형 완화)   
> lambda : 데이터 셋 특성에 따라 1, 0으로 손실 함수 조정해서 효율성 극대화   

> #### 📌 데이터 셋 별   
> Detection, Grounding 데이터 셋 : Lcon + Liou + Ldfl(lambda=1)   
> Image-text 데이터 셋 : Lcon(lambda=0)   


4️⃣ Pseudo labeling with Image text data     
- 자동 레이블링 방식을 통해 Region-text pairs를 생성하여 학습   
  
> #### 📌 전통적 객체 검출방법과 차이   
> - 전통적인 객체 검출방법(ex, YOLO 시리즈) : Instance annotations을 {경계상자 Bi, 카테고리레이블 ci}   
> - YOLO World : Instance annotations을 Region text pairs {경계상자 Bi, 영역Bi에 해당하는 텍스트 ti}, ti : 카테고리 이름, 명사구, 객체설명 등   
> - YOLO World Input : 이미지 I, 텍스트 T(명사집합)   
> - YOLO World Output : 예측된 Bbox들, 해당 객체 임베딩들   

> #### 📌 과정   
> - n-gram 알고리즘 사용해서 텍스트에서 명사구 추출   
> - GLIP등 사전학습된 open vocabulary detector 사용해서 pseudo labeling하여, 명사구에 대응하는 bbox를 자동생성   
> - CLIP을 사용해 연관성 낮은 pseudo label과 이미지 제거, NMS로 중복된 경계 상자 제거   

(결과) CC3M 데이터 셋(246,000개 이미지 샘플)에서 821,000개의 Region-Text Pairs 생성   
(효과) 효율적인 데이터 준비, 텍스트 기반 객체 검출 성능과 Zero-Shot 학습 능력을 강화   


### Solution approach - Experiments

### 1️⃣ 실험 목적 : Vision language 기반 zero shot 객체 검출을 위해 설계된 SOTA 수준 보여주는 방법들과 비교 : GLIP, GLIPv2, Grounding DINO, DetCLIP   
![image](https://github.com/user-attachments/assets/da7a588f-7930-45ac-b557-392005e949b5)        
> #### 📌 비교기준   
> - AP(zero shot 성능), inference speed, model parameters, pretraining data    
> #### 📌 결과   
> - 결과 1 : YOLO-World는 LVIS에서 Zero-Shot 성능(AP) 기준으로 35.4 AP를 달성   
> - 결과 2 : YOLO-World-S는 13M 파라미터라는 경량 구성에서도 강력한 Zero-Shot 성능을 보임   
> - 결과 3 : DetCLIP 대비 20배 빠른 추론 속도   
> - 결과 4 : GLIP, GLIPv2, Grounding DINO와 달리 Cap4M (CC3M + SBU)와 같은 추가 데이터 없이도 더 나은 성능을 보임   

### 2️⃣  
### 1) 실험 목적 : 사전 학습 데이터의 다양성 (다양한 데이터셋이 사전 학습 성능 및 Zero-Shot 검출 성능에 미치는 영향을 분석)     
- Objects365 단독 사용 vs. Objects365 + GoldG 사용   
### 2) 실험 목적 : pseudo labeling 효과 (CC3M 데이터셋에서 생성된 pseudo label이 모델 학습에 기여하는 정도 평가)   
![image](https://github.com/user-attachments/assets/414f405d-28bb-41e1-8671-b1868f6721bb)   
> #### 📌 결과 1 : 데이터 셋의 다양성이 모델성능에 긍정적 영향 미침    
> #### 📌 결과 2 : pseudo labeling이 성능에 유의미한 기여함    

### 3️⃣ 텍스트 인코더 역할 : Frozen vs. Fine-tuned    
![image](https://github.com/user-attachments/assets/e17ed39c-f0e9-4941-bff3-784b86e07006)    
> #### 📌 결과   
>  - BERT-base는 Frozen에서 APr 성능이 매우 낮다.   
>  - BERT-base는 Fine tuning하니 성능이 향상 됐다.   
>  - CLIP-base는 Frozen에서 AP, APr 모두 BERT보다 성능이 높다.   
>  - CLIP-base는 Fine tuning하니 오히려 성능 감소(왜냐, O365는 365개 카테고리로 제한된 데이터셋으로 CLIP의 일반화 능력이 오히려 저하됨)   
>  - Zero-Shot Detection에서 CLIP의 사전 학습된 임베딩이 효과적임   

### 4️⃣ GQA데이터 사용시 성능 여부 + Text guided CSPLayer, Image pooling attention 사용이 유의미한지   
![image](https://github.com/user-attachments/assets/ee875e8e-7ef7-437c-9d02-18cb9b4610e4)   
> #### 📌 비교기준     
> - GQA 데이터셋 사용여부, T-I(Text guided CSPLayer), I-T(Image pooling attention)   
> - AP : 전체 검출 성능   
> - APr : Rare 카테고리에서의 AP   
> - APc : Common 카테고리에서의 AP   
> - APf : Frequent 카테고리에서의 AP   

> #### 📌 결과   
>  - GQA데이터 사용 시 성능이 다 높다. GQA 데이터가 풍부한 텍스트 주석을 포함하고 있기 떄문   
>  - Text guided CSPLayer, Image pooling attention 두 모듈 모두 사용했을 때 성능이 가장 뛰어남   
>  - GQA 데이터 + RepVL-PAN의 모든 구성 요소 다 활용했을 때, APr성능이 22.5로 가장 높음   

### 5️⃣ Fine-tuning YOLO-World : YOLO-World와 기존의 YOLO 계열 모델(YOLOv6, YOLOv7, YOLOv8)을 COCO 데이터셋에서 비교한 결과   
![image](https://github.com/user-attachments/assets/d2467377-723d-4f39-9b6f-54b20f2d4cbe)   
> #### 📌 비교기준    
> - x : 사전 학습 없이 scratch에서 학습   
> - O : Objects365   
> - G : GoldG   
> - C : CC3M   
> - FPS : 초당 프레임 수   

> #### 📌 결과    
>  - 사전학습만으로 COCO 데이터셋에서 zero shot 검출을 수행하며, 일부 YOLO 모델의 scratch 학습 성능과 유사한 수준 달성   
>  - fine tuning시, 기존 YOLOv8-L 능가   
>  - COCO 데이터 셋처럼 어휘 크기가 작은 경우, RepVL-PAN의 중요성은 상대적으로 낮음. RepVL-PAN 제거해도 성능 손실이 적고 FPS 향상됨   
>  - 즉, 사전학습, Fine tuning을 통해 성능, 효율성 모두 기존 YOLO 모델 능가하거나 동등함   


### 6️⃣ Fine-tuning YOLO-World : YOLO-World는 모든 크기(S, M, L)에서 YOLOv8 대비 우수한 성능을 보임   
![image](https://github.com/user-attachments/assets/cfd8a4e6-9412-41b8-841e-fce905ea13d6)   
> #### 📌 결과    
> - APr : 20.4 → YOLOv8-L(10.2) 대비 +10.2 APr   
> - YOLO-World는 ViLD, RegionCLIP, Detic 등 최신 모델과 비교해도 성능이 우수   
> - YOLO-World-S (소형 모델) : APr에서도 YOLOv8-S(7.4) 대비 +5.4 APr   
> - YOLO-World-M (중형 모델) : APf에서는 39.0으로 가장 높은 성능   
> - YOLO-World-L (대형 모델) : 모든 지표(AP, APr, APc, APf)에서 최고 성능 기록   
> YOLO-World는 One stage detector임에도 불구하고, 기존 Two stage detector(DetPro, BARON, ViLD 등)보다 높은 성능을 달성   

<br>

### Visulaizations   
![image](https://github.com/user-attachments/assets/a2800e24-245c-4eed-9d27-dc89c77168a6)   
- YOLO-World-L은 사전 학습된 상태에서 LVIS 데이터셋에 대해 Zero shot 방식으로 객체를 탐지   
- 사전학습된 YOLO-World-L을 채택해서 LVIS 데이터셋의 전체 카테고리(1203개)를 사용하여 COCO val2017 데이터셋에서 추론을 수행   
- 모델은 사전 학습된 상태에서 COCO의 객체들을 LVIS의 1203개 카테고리로 매칭   
- YOLO-World-L은 COCO 데이터셋에서 추가 학습 없이 LVIS의 대규모 카테고리를 통해 객체 탐지 수행   

![image](https://github.com/user-attachments/assets/8090f443-5f20-4491-b3f0-3eb595f95a3d)   
- 사용자가 정의한 맞춤 카테고리(Custom Categories)를 기반으로 YOLO-World-L의 객체 탐지 성능을 평가   
- 특정 객체의 세부 부분 감지 : 자동차를 타이어, 창문, 문 등으로 세분화하여 탐지   
- 서브 카테고리 구분 : 개를 리트리버, 불독, 푸들 등 세부적인 견종으로 분류   

![image](https://github.com/user-attachments/assets/34a3721a-8074-461e-b1f7-3d25d3678e2f)    
- 모델은 입력된 서술형 명사구와 이미지의 객체 간 관계를 학습한 결과를 활용하여 정확히 매칭   
- the red car라는 입력 명사구에 대해 이미지에서 빨간 자동차를 찾아 Bounding Box 생성   
  
<br>

### Conclusion   
- YOLO-World는 실시간 Open-Vocabulary detection을 목표로 한 최신 탐지기   
- 기존 YOLO를 Vision-Language 아키텍처로 재구성하고 RepVL-PAN을 통해 효율성과 탐지 성능을 강화   
- 탐지, 정합, 이미지-텍스트 데이터를 활용한 사전 학습을 통해 대규모 어휘 탐지 능력을 확보   
- 소형 모델에서도 효과적인 Vision-Language 학습 가능성을 입증   
  
- 다양한 객체를 Zero-Shot 방식으로 효율적으로 검출할 수 있도록 한다.   
- LVIS 데이터 셋에서 V100 GPU에서 35.4 AP, 52.0 FPS의 성능을 달성했고, 정확도, 속도 모두에서 최신 기법들보다 뛰어난 성능을 보여준다.   
- 더하여, Fine Tuning된 YOLO-World는 Object Detection, Open Vocabulary Instance Segmentation 등 여러 다운스트림 작업에서 뛰어난 성능을 발휘했다.   

<br>

### Strong points


<br>

### Weak points
- Offline vocabulary를 만들 때 메모리를 많이 차지할 거 같다.

<br>

### Questions
- 부정명사를 무작위 샘플링한다고 했는데 어디서 샘플링을 하는 걸까?  
- infoNCE Loss를 사용하지않고 Cross entropy loss를 사용한 이유가 뭘까?  
### New ideas / Comments
