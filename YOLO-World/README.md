## [논문정리]YOLO-World: Real-Time Open-Vocabulary Object Detection (2024), Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan

### Problem statement
- YOLO는 사전 정의되고 학습된 객체 카테고리에 의존하기 때문에 개방형 환경에서는 그 활용에 제한이 있다.
  ```
  - 사전 정의되고 학습된 객체 카테고리 : 학습할 때, 특정한 클래스에 대해서만 학습. 즉, 학습하지 않은 새로운 객체는 인식하지 못 함
  ```
  예를 들어, COCO 데이터 셋에서는 80개의 카테고리로 객체를 정의하고, 이 카테고리만 학습하도록 되어 있다.
  한 번 객체의 카테고리(클래스)가 정의되고 레이블링이 완료되면, 학습된 검출기는 그 특정한 카테고리만 인식할 수 있다. 즉, 학습하지 않은 새로운 카테고리는 검출할 수 없기 때문에, Open-Vocabulary 시나리오에서는 활용에 제한이 생긴다.
  ```
  전통적인 객체 검출 방법은 크게 3가지 그룹으로 분류된다.
  1) Region Based Methods : Faste R-CNN, R-CNN
    Two-Stage 방식을 채택
    첫 번째 단계에서, Proposal 생성 수행
    두 번째 단계에서, RoI 단위로 분류, 회귀 수행
    (단점) 속도 느리다.
    (장점) 정확도 높다. 대형 모델에 적합하고, 고해상도 객체 인식에 주로 사용된다.
  2) Pixel Based Methods : RetinaNet, SSD
    One-Stage 방식을 채택
    고정된 앵커박스나 픽셀 단위로 클래스 분류와 경계박스회귀를 동시에 수행
    (장점) 속도 빠르다. 계산 비용 낮다.
    (단점) 정확도는 Region-Based 방식보다 낮다. 실시간 검출이 중요한 어플리케이션(ex, 자율주행)에 자주 사용된다.
  3) Query Based methods : DETR, Deformable DETR
    Transformer 모델을 활용한 객체 검출 방식
    입력 이미지의 특징을 Transformer의 쿼리로 변환하고, Transformer의 self-attention 매커니즘을 통해 객체의 위치, 클래스를 예측한다.
    (장점) 간단하고 직관적인 구조다. 기존의 CNN 기반 방법과 차별화되어 있으며, Tranformer 아키텍처의 장점을 활용한다.
    (단점) 훈련에 많은 시간이 필요하다.
  4) YOLO 방법 : YOLOv1 ~ YOLOv8, PP-YOLO
    간단한 CNN 아키텍처를 활용하여, 한 번의 Forward Pass로 이미지의 모든 객체를 예측한다.
    고정된 크기의 그리드 셀로 이미지를 나누고, 각 셀에 대해 객체의 클래스와 경계박스 위치를 동시에 예측한다.
    (장점) 실시간 객체 검출이 가능하여 자율주행, CCTV 등 실시간 어플리케이션에서 사용된다.
          속도와 정확도의 균형을 유지하기 위해, 다양한 네트워크 구조와 기법들이 제안되었다.
          속도 향상 기법
          - PAN(Path Aggregation Network) : 여러 피처 맵을 효율적으로 결합하여 더 나은 표현 학습 수행
          - CSPNet(Cross-Stage Partial Network) : 모델의 중복 계산을 줄이고 효율성을 높인다.
          - Re-Parameterization 기법 : Train 단계와 Inference 단계를 분리하여, Inference 중에 모델의 효율성을 향상시킨다.
  ```
- 최근 연구들은, BERT 같은 언어 인코더로부터 어휘 지식을 distillation하는 방법을 통해, Open Vocabulary Detection 문제를 해결하기 위해, Vision Language Model을 활용했다.
  그러나, Knowledge Distillation based 방법들은, 학습 데이터의 부족, 어휘의 다양성이 제한적이므로 한계를 보인다.
- 몇몇 연구들은, Object Detection 학습을 Region-Level Vision Language Pretraining으로 재구성하고, 대규모 데이터 셋에서 Open Vocabulary 객체 검출기를 훈련하는 법을 제안했다.
  그러나 Real World Detection에서 막대한 계산 비용, 엣지 디바이스(ex, 모바일 장치)에 배포하는 과정이 복잡함으로 어려움을 겪는다.
  Lage Detectors의 사전 학습을 통해 성능 입증했으나, Small Detectors에 Open Recognition 능력을 부여하기 위한 사전학습에 대한 연구는 충분히 이뤄지지 않았다.
<br>

### Solution approach
- YOLO World 구조 : YOLO 표준 아키텍처 + 입력 텍스트를 인코딩하기 위해 사전 학습된 CLIP의 텍스트 인코더 사용
- Vision-Language Modeling, 대규모 Data Set에 대한 Pre-training을 통해 YOLO의 기능을 확장한다.
  
- RepVL-PAN(Re-parrameterizable Vision-Language Path Aggregation Network)
  - 텍스트 특징과 이미지 특징을 연결하여 더 나은 Visual-Semantic Representation 가능하게 함
  - Inference 단계에서는 텍스트 인코더를 제거할 수 있다.
  - 사전 학습된 텍스트 임베딩을 RepVL-PAN의 가중치로 Re-parameterization하여 효율적인 배포가 가능하다.

- Region-Text Contrastive Loss
  - YOLO 검출기의 Open-Vocabulary Pre-traing을 위해, Region-Text Contrastive Learning을 대규모 데이터 셋에서 수행한다.
  - Detection Data, Grounding Data, Image-Text Data를 Region-Text Pairs로 통합한다.
  - Region-Text Pairs가 풍부하게 사전학습된 YOLO-World는 대규모 어휘 검출에 강한 능력을 보인다.


  ![image](https://github.com/user-attachments/assets/dc3f27f4-d6ea-4829-887b-43a346e4d0c5)
  1) 기존 OVD(Open Vocabulary Object Detection)은, 기존의 기본(base) 클래스로 검출기를 학습하고, 알 수 없는(novel) 클래스에 대해 평가한다.
  - 기존의 객체 검출기들은, 사전에 정의된 고정된 어휘에 기반하여 사전학습된 카테고리로만 검출을 수행
  - 예시 : COCO 데이터 셋의 고양이, 개, 자동차로 학습했을 때, 학습에 포함되지 않은 코끼리가 주어졌을 때, zero shot 방식으로 새로운 객체를 검출 할 수 있는지를 평가한다.
  - 한계점 : 다양한 도메인에 일반화 능력 떨어짐 (ex, COCO 데이터 셋에서 학습된 모델 -> 의료영상 도메인에서 성능 떨어짐)
  
  2) Previous Open Vocabulary Detectors
    최근 OVD 접근 방식은, Vision Language Pre traing의 성공에 힘입어, OVD를 이미지-텍스트 매칭 문제로 재정의 했다.
  - 이전의 Open Vocabulary 검출기들은, 사용자의 prompt를 텍스트 인코더로 인코딩하여, Online Vocabulary를 생성한 후, 객체를 검출한다.
  - 이미지와 텍스트를 동시에 인코딩하여 예측을 수행한다.
  - 사용자가 입력한 Prompt를 기반으로 Online Vocabulary를 생성한다.
  - (한계점) 이미지-텍스트를 동시에 인코딩하므로 계산 비용이 높고, 실용성이 부족하다.
  - 핵심 아이디어 : 이미지 - 텍스트 관계를 학습하여, 대규모 Image-Text Pairs를 활용해, 더 큰 범위의 어휘로 검출 모델을 훈련하는 것
  - 대표적인 연구들 : OWL-ViTs [35, 36], GLIP [24], Grounding DINO [30], 통합된 데이터 셋을 통한 OVD 접근 방식
  - 대형검출기, 무거운 Backbone을 사용한다.
  
  3) YOLO-World
  - Lightweight Detector 기반으로 설계되었다.
  - prompt-then-detect 패러다임 도입
    ```
    사용자가 필요에 따라 Prompt(텍스트 입력)를 생성한다.
    Prompt는 **오프라인 어휘(Offline Vocabulary)**로 인코딩된다.
    이 오프라인 어휘는 재구성(re-parameterization) 되어 모델의 가중치로 변환된다.
    이후 Inference 단계에서는 텍스트 인코더 없이 빠르게 예측할 수 있다.
    ```
    - prompt를 미리 인코딩하여 Offline Vocabulary로 만든다.
    - 한 번 생성된 어휘는 다시 인코딩할 필요가 없으며, 검출기가 이를 즉시 활용 가능하다.
    - YOLO World와 같은 검출기는 Offline Vocabulary를 즉시 추론에 활용할 수 있다.
    - YOLO World와 같은 검출기를 한 번 훈련하면, 사용자는 사전 인코딩된 prompt 또는 카테고리를 기반으로 Offline Vocabulary를 생성할 수 있다.
  - (장점) 실시간 추론이 가능하고, 엣지 디바이스에도 적합하다, 텍스트 인코더를 Inference 도중에 호출하지 않으므로 기존방식보다 훨씬 빠르다, Prompt를 사용해 사용자 맞춤형 어휘를 쉽게 추가할 수 있다.
    ```
    - Online Vocabulary : 모델이 Inference 중에 실시간으로 prompt를 인코딩하여 어휘를 생성하는 방식
    - Offline Vocabulary : Inference 전에 사전 정의된 어휘를 미리 인코딩하고 저장해두고, Inference 중에는 인코딩 없이 재사용하는 방식
    ```


<br>

### Conclusion
- 다양한 객체를 Zero-Shot 방식으로 효율적으로 검출할 수 있도록 한다.
- LVIS 데이터 셋에서 V100 GPU에서 35.4 AP, 52.0 FPS의 성능을 달성했고, 정확도, 속도 모두에서 최신 기법들보다 뛰어난 성능을 보여준다.
- 더하여, Fine Tuning된 YOLO-World는 Object Detection, Open Vocabulary Instance Segmentation 등 여러 다운스트림 작업에서 뛰어난 성능을 발휘했다.
   	  
<br>

### Strong points


<br>

### Weak points


<br>

### Questions


### New ideas / Comments
