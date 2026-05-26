# 서울대학교 공과대학 연구자 매칭 시스템 데모

이 데모는 서울대학교 공과대학의 연구자 및 연구실 정보를 효과적으로 검색하고 탐색할 수 있는 웹 기반 시스템입니다. 사용자는 키워드 검색을 통해 관련성이 높은 교수님을 찾고, 특정 연구 주제와 유사한 연구를 수행하는 다른 교수님들을 탐색할 수 있습니다.

## 주요 기능

- **전체 교수님 검색**: 전체 교수님들의 연구 및 과제 정보를 바탕으로 입력된 키워드와 관련성이 높은 교수님을 검색합니다.
- **신진 교수님 검색**: 별도로 지정된 신진 교수님 목록 내에서 관련 연구를 수행하는 교수님을 검색합니다.
- **유사 연구실 추천**: 검색 결과에서 특정 연구/과제를 선택하면, 해당 내용과 유사한 연구를 수행하는 다른 교수님들을 추천받을 수 있습니다.

## 파일 구성

- `main.py`: Gradio를 사용하여 웹 데모 인터페이스를 구성하고, 검색 로직을 처리하는 메인 스크립트입니다.
- `run.sh`: 데모 실행에 필요한 환경변수와 인자들을 설정하고 `main.py`를 실행하는 쉘 스크립트입니다.
- `utils.py`: 검색, 인덱싱, 데이터 로딩 등 데모에 필요한 보조 함수들을 포함합니다.
- `new_researchers/`: 신진 교수님 목록이 포함된 디렉토리입니다.

## 실행 방법

### Docker Compose로 실행

레포를 clone한 뒤 프로젝트 루트에서 아래 순서로 실행합니다.

```bash
git clone https://github.com/KimJaehee0725/SNUResearcherMatching.git
cd SNUResearcherMatching

./demo/download_artifacts.sh
./demo/run_docker_compose.sh
```

데모는 기본적으로 `http://localhost:7860`에서 실행됩니다. 포트나 GPU를 바꾸려면 아래처럼 환경변수를 지정합니다.

Docker 이미지는 DGX Spark/Grace Blackwell 환경에 맞춰 `nvcr.io/nvidia/pytorch:25.10-py3`를 기반으로 빌드합니다. 서버에서 NGC 인증이 필요하면 먼저 `docker login nvcr.io`를 수행합니다.

```bash
DEMO_PORT=7861 CUDA_VISIBLE_DEVICES=1 DEMO_DEVICE=cuda:0 ./demo/run_docker_compose.sh
```

DGX Spark에서 PyTorch 이미지 호환성 문제가 계속 나면 더 최신 NGC PyTorch 태그로 바꿔 빌드할 수 있습니다.

```bash
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 ./demo/run_docker_compose.sh
```

### 1. 필요 파일 준비

데모를 실행하기 위해서는 사전에 학습된 언어 모델과 검색 대상이 될 데이터(Corpus), 그리고 생성된 인덱스 파일이 필요합니다. `run.sh` 스크립트에 기본 경로가 지정되어 있으며, 필요에 따라 경로를 수정해야 합니다.

- **모델**: `--model_path` 인자에 지정된 경로에 BAAI/bge-m3와 같은 임베딩 모델이 위치해야 합니다.
- **데이터**: `--research_corpus_path` 및 `--project_corpus_path`에 교수님들의 연구 및 과제 정보가 담긴 CSV 파일이 필요합니다.
- **인덱스**: `--index_path`는 검색을 위해 생성된 Faiss 인덱스가 저장될 경로입니다. 인덱스 파일이 없는 경우, 처음 실행 시 자동으로 생성됩니다.

### 2. 데모 실행

아래의 명령어를 사용하여 데모를 실행합니다.

```bash
bash run.sh
```

스크립트가 실행되면 Gradio 기반의 웹 인터페이스가 실행되며, 터미널에 출력되는 URL을 통해 접속할 수 있습니다.

## `main.py` 인자 설명

`run.sh` 또는 `main.py` 직접 실행 시 다음 인자들을 사용하여 설정을 변경할 수 있습니다.

- `--model_path`: 검색에 사용할 사전 학습된 임베딩 모델의 경로.
- `--research_corpus_path`: 연구 정보 데이터 파일(.csv)의 경로.
- `--project_corpus_path`: 과제 정보 데이터 파일(.csv)의 경로.
- `--research_new_prof_path`: 신진 교수님 목록(연구) 파일의 경로.
- `--project_new_prof_path`: 신진 교수님 목록(과제) 파일의 경로.
- `--index_path`: 생성된 Faiss 인덱스를 저장하거나 불러올 디렉토리 경로.
- `--device`: 모델을 실행할 장치 (예: `cuda:0`, `cpu`).

## 추론 파이프라인 및 주요 함수

이 데모의 추론 파이프라인은 `utils.py`에 정의된 함수들을 기반으로 동작하며, 외부 RAG 시스템에 이식할 때 핵심적인 역할을 합니다.

### 핵심 파이프라인 개요

1.  **초기화**: `get_model_and_corpus_with_index` 함수를 호출하여 임베딩 모델, 전체 데이터(Corpus), 그리고 Faiss로 구축된 검색 인덱스를 메모리에 로드합니다. 인덱스 파일이 지정된 경로에 없으면, 전체 데이터를 인코딩하여 새로운 인덱스를 생성하고 저장합니다.
2.  **검색 수행**:
    -   사용자 질의가 입력되면 `search_full_db` 또는 `search_subset_db` 함수가 호출됩니다.
    -   질의는 임베딩 모델을 통해 벡터로 변환됩니다.
    -   Faiss 인덱스에서 해당 벡터와 유사도가 높은 문서들을 검색합니다.
    -   검색된 문서를 교수 단위로 그룹화하고, 관련성이 높은 순으로 정렬하여 반환합니다.
3.  **유사 교수 탐색**:
    -   사용자가 검색 결과에서 특정 문서를 선택하면 `find_similar_professors` 함수가 호출됩니다.
    -   선택된 문서의 텍스트가 새로운 질의가 되어 Faiss 인덱스에서 다시 검색을 수행합니다.
    -   이를 통해 원본 문서와 유사한 연구/과제를 수행하는 다른 교수들을 추천합니다.

### 주요 함수 설명 (`utils.py`)

-   `get_model_and_corpus_with_index(device, model_path, ..., index_path)`
    -   **역할**: 추론에 필요한 모든 구성 요소를 로드하고 준비합니다.
    -   **주요 동작**:
        -   `SentenceTransformer` 모델을 `model_path`에서 로드합니다.
        -   `research_corpus_path`와 `project_corpus_path`에서 CSV 데이터를 읽어 `full_corpus`를 생성합니다.
        -   `index_path`에 Faiss 인덱스가 있으면 로드하고, 없으면 `full_corpus`를 임베딩하여 새로 생성 후 저장합니다.
        -   신진 교수 목록을 이용해 `subset_corpus`와 `subset_index`를 별도로 생성합니다.
    -   **반환값**: `model`, `full_corpus`, `full_index`, `subset_corpus`, `subset_index`

-   `search_full_db(user_q, index, corpus, model, ...)`
    -   **역할**: 전체 교수님을 대상으로 사용자 질의(`user_q`)와 가장 관련성 높은 교수님 및 문서를 검색합니다.
    -   **주요 동작**:
        1.  `model.encode()`를 사용하여 `user_q`를 쿼리 임베딩으로 변환합니다.
        2.  `index.search()`를 통해 Faiss 인덱스에서 유사도가 높은 문서 ID를 검색합니다.
        3.  검색된 문서들을 교수 단위로 필터링하고 그룹화하여 결과를 정리합니다.
    -   **반환값**: 교수 이름을 key로, 관련 문서 리스트를 value로 갖는 딕셔너리.

-   `find_similar_professors(document, index, corpus, model, ...)`
    -   **역할**: 특정 문서(`document`)의 텍스트를 기반으로 유사한 연구/과제를 수행하는 다른 교수님을 찾습니다.
    -   **주요 동작**: `search_full_db`와 유사하나, 사용자 질의 대신 입력된 `document` 텍스트를 쿼리로 사용합니다. 또한, 이미 검색 결과에 노출된 교수를 제외하는 로직(`excluded_profs`)이 포함됩니다.

### 커스터마이징 가이드

-   **모델 및 데이터 경로**: `main.py`의 `argparse` 부분이나 `run.sh` 스크립트에서 모델, 데이터, 인덱스 경로를 자유롭게 변경할 수 있습니다.
-   **템플릿 수정**: `utils.py` 상단의 `PROJECT_TEMPLATE`, `RESEARCH_TEMPLATE` 변수를 수정하여 모델에 입력되는 문서의 형식을 변경할 수 있습니다. 이는 모델 학습 시 사용된 형식과 일치시키는 것이 중요합니다.
-   **검색 로직 변경**: `_filtered_by_professor` 함수는 검색된 문서를 교수 단위로 그룹화하는 로직을 담고 있습니다. 만약 문서 단위의 검색 결과가 필요하다면 이 부분을 수정하여 사용할 수 있습니다. 
