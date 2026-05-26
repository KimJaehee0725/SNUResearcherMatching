# 서울대학교 공과대학 연구자 매칭 시스템

본 프로젝트는 서울대학교 공과대학 내 연구실에 대한 검색 시스템 학습 및 추론 과정을 담고 있습니다. 
전체 파이프라인은 아래와 같습니다. 
1. 학습: query-passage 쌍을 학습하여 임베딩 모델을 학습합니다. 
2. 추론: 학습된 모델을 활용하여 연구자 정보를 검색합니다. 
3. 데모: 추론 결과를 시각화하여 사용자에게 제공합니다. 

## 프로젝트 구조

이 프로젝트는 크게 두 가지 주요 구성 요소로 나뉩니다.

- **`train/`**: 검색 시스템의 핵심인 언어 모델을 학습시키기 위한 코드와 데이터, 설정 파일이 포함되어 있습니다.
- **`demo/`**: 학습된 모델을 활용하여 실제로 연구자 정보를 검색하고 탐색할 수 있는 웹 기반의 데모 애플리케이션입니다.
- **`data/`**: 모델 학습과 데모 실행에 필요한 원본 데이터셋(Corpus)과 전처리된 파일, Faiss 인덱스 등을 저장하는 디렉토리입니다. 최종 결과물은 다음과 같은 구조를 가집니다.
  ```
  data
  ├── demo
  │   └── ver1
  │       ├── full.faiss      # 전체 교수진 대상 Faiss 인덱스
  │       └── subset.faiss    # 신진 교수진 대상 Faiss 인덱스
  ├── new_researchers
  │   ├── project_new_prof_10.txt
  │   ├── ...
  │   └── research_new_prof_10.txt
  ├── project_eval_randsamp_v2_del_department.jsonl
  ├── project_train_randsamp_v2.jsonl
  ├── project_translation_final.csv             # "과제" 정보 정제 데이터
  ├── research_eval_randsamp_v2_del_department.jsonl
  ├── research_train_randsamp_v2.jsonl
  └── research_translation_final.csv            # "연구" 정보 정제 데이터
  ```
- **`trained_model/`**: 사전 학습되었거나 이 프로젝트를 통해 학습된 모델 파일들을 저장하는 디렉토리입니다. 데모에서 사용하는 최종 모델의 구조는 다음과 같습니다.
  ```
  trained_model
  ├── 1_Pooling
  │   └── config.json
  ├── 2_Normalize
  ├── config.json
  ├── model.safetensors       # 모델의 가중치(weight) 파일
  ├── ...
  └── tokenizer.json          # 토크나이저 관련 설정 파일
  ```
- **`train/`**: 모델 학습을 위한 파이썬 스크립트와 설정 파일이 있습니다. (`train/README.md` 참고)
- **`demo/`**: 학습된 모델을 시연하기 위한 Gradio 기반의 데모 애플리케이션이 있습니다. (`demo/README.md` 참고)

각 디렉토리의 세부적인 사용법과 설명은 해당 디렉토리 내의 `README.md` 파일을 참고하십시오.

- **[모델 학습 관련 안내](./train/README.md)**
- **[데모 실행 관련 안내](./demo/README.md)**

## 주요 기술

- **언어 모델**: `Sentence-BERT`와 같은 Transformer 기반의 모델을 사용하여 텍스트의 의미적 유사도를 측정합니다.
- **검색**: `Faiss`와 같은 라이브러리를 사용하여 대규모 데이터셋에서 빠르고 효율적인 유사도 검색을 수행합니다.
- **웹 데모**: `Gradio`를 활용하여 사용자가 쉽게 상호작용할 수 있는 웹 인터페이스를 제공합니다.
- **분산 학습**: `PyTorch DDP` 및 `accelerate`를 지원하여 다중 GPU 환경에서 효율적으로 모델을 학습할 수 있습니다.
- **컨테이너화**: `Dockerfile`을 제공하여 프로젝트 실행 환경을 손쉽게 구축할 수 있습니다.

## 환경 설정

### 요구 사양

본 프로젝트 개발 시 사용한 Docker 컨테이너는 아래 명시된 환경을 기반으로 구성되어 있습니다.

- **Base Image**: `nvcr.io/nvidia/pytorch:22.09-py3`
- **OS**: Ubuntu 20.04
- **Python**: 3.8
- **CUDA**: 11.8.0

서버에서 Docker를 사용하여 이 프로젝트를 실행하려면, NVIDIA 드라이버 버전 520 이상이 설치되어 있어야 합니다.

프로젝트를 실행하기 위해서는 다음 두 가지 방법 중 하나를 선택할 수 있습니다.

1.  **Docker 기반 설정 (권장)**:
   - 이 방법은 환경 설정의 복잡성을 줄이고 배포의 일관성을 보장합니다.
   - 프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 이미지를 빌드합니다:
     ```bash
     docker build -t snu_engineering_search .
     ```
   - 빌드된 이미지를 사용하여 아래와 같이 다양한 작업을 수행할 수 있습니다.
     
     **중요:** 컨테이너에서 GPU를 정상적으로 사용하려면, 호스트 머신에 **NVIDIA 드라이버 버전 520 이상**이 설치되어 있어야 합니다.

### 3. Docker 컨테이너 실행

빌드된 이미지를 사용하여 컨테이너 내부에서 직접 명령어를 실행하고 싶을 경우, 다음 명령어로 대화형 세션을 시작할 수 있습니다.

```bash
docker run --gpus all -it --name {container_name} snu_search:1.0 /bin/bash
```

대화형 세션에서 `train` 또는 `demo` 디렉토리의 스크립트를 직접 실행하여 모델 학습 및 데모 시연을 진행할 수 있습니다.

2. **수동 패키지 설치**:
   - 프로젝트 실행에 필요한 파이썬 패키지들은 `requirements.txt`에 명시되어 있습니다.
   - 다음 명령어를 사용하여 패키지를 설치합니다:
     ```bash
     pip install -r requirements.txt
     ```