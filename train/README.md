# SentenceTransformer IR Trainer

## 업데이트된 기능 (FINAL)

| **제목**              | **설명**                                                                 | **완료** |
|------------------------|--------------------------------------------------------------------------|----------|
| **동적 샘플링 (Dynamic Sampling)** | 하나의 Document에 대해 여러 개의 Positive Query가 존재할 수 있으므로, 매 Epoch마다 해당 Document에 대한 Positive Query를 랜덤하게 샘플링합니다. | ✅       |
| **데이터 병렬 처리 (Data-Parallelism)** | `accelerate`의 DDP 기능을 활용하여 데이터 병렬 처리를 구현했습니다. | ✅       |
| **문서 포맷팅 (Document Formatting)** | 메타정보를 활용하여 문서를 구성하는 기능을 구현했습니다. | ✅       |

## 프로젝트 구조

```
root/
├── configs/                                # YAML 구성 파일들이 있는 폴더
│   ├── debug.yaml                          # 예시 구성 파일
│   ├── project_default.yaml
│   ├── research_default.yaml
│   ├── total_default.yaml
│   └── {CONFIG_NAME}.yaml                  # 사용자 정의 구성 파일
├── data/
│   ├── files/                              # jsonl 형식의 데이터 파일
│   │   ├── project_train_randsamp_v2.jsonl     # 학습 데이터
│   │   ├── research_train_randsamp_v2.jsonl     # 학습 데이터
│   │   ├── project_eval_randsamp_v2.jsonl     # 평가 데이터
│   │   ├── research_eval_randsamp_v2.jsonl     # 평가 데이터
│   │   └── {DATA_NAME}.jsonl               # 사용자 데이터
│   └── data_controller.py                  # 데이터 로딩, 전처리, 헬퍼 함수들
├── models/
│   └── {MODEL_NAME}/                       # 모델 캐시 디렉토리
├── results/                                # 학습 결과가 저장되는 디렉토리
│   ├── {MODEL_NAME}/            
│   │   ├── {RUN_NAME}/                     
│   │   │   ├── checkpoint-{STEPS}/         # 특정 스텝의 체크포인트 저장
│   │   │   │   └──...
│   │   │   └── config.yaml                 # 실행 시 사용된 구성 파일
│   │   └── ... 
│   └── ...                      
├── trainer/
│   ├── train.py                            # CustomTrainer와 동적 샘플링을 사용하는 학습 코드
├── main.py                                 # 학습 및 평가를 호출하는 메인 엔트리포인트
└── README.md                               
```

## 실행 방법

- `--config`:
  `./configs/` 폴더에 있는 구성 파일 이름을 확장자 없이 입력합니다 (기본값은 `debug`).

### 싱글 GPU에서 실행

환경변수로 GPU를 지정하고 다음 명령어를 실행하세요:

```
CUDA_VISIBLE_DEVICES=0 python main.py --config {YOUR_CONFIG}
```

### 분산 학습 (DDP)으로 실행

여러 개의 GPU를 사용할 경우 `accelerate` CLI를 사용하거나 PyTorch의 런처를 사용할 수 있습니다:
```
accelerate launch --num_processes 4 main.py --config {YOUR_CONFIG}
```
또는 PyTorch의 기본 런처 사용:
```
python -m torch.distributed.launch --nproc_per_node=4 main.py --config {YOUR_CONFIG}
```

### 결과

- 결과는 `./results/{YOUR_CONFIG[model_name]}/{YOUR_CONFIG[run_name]}/`에 저장됩니다.
- 각 실행의 결과와 설정은 해당 경로에서 확인할 수 있습니다.
