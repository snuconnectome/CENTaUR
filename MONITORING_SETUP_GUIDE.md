# GPU 프로파일링 및 W&B 모니터링 설정 가이드

## 📊 GPU 프로파일링 지속 실행

### ✅ 가능 여부

**네, GPU 프로파일러를 계속해서 동시에 가동할 수 있습니다!**

### 프로파일링 오버헤드

- **CPU 오버헤드**: 매우 낮음 (< 1%)
- **GPU 오버헤드**: 미미함 (trace 모드)
- **디스크 사용**: 프로파일 파일 크기 증가
  - Fine-tuning: ~188KB - 8.4MB
  - 장기 실행 시: 수십 MB ~ 수백 MB

### 자동 프로파일링 설정

새로운 스크립트 `start_centaur_finetuning_with_monitoring.sh`가 다음을 제공합니다:

1. **자동 프로파일링**: 모든 학습에 자동으로 nsys 프로파일링 포함
2. **프로파일 파일 관리**: 오래된 파일 자동 정리 (최신 10개만 유지)
3. **타임스탬프 포함**: 각 실행마다 고유한 프로파일 파일 생성

### 사용 방법

```bash
# 기본 사용 (자동 프로파일링 포함)
bash scripts/start_centaur_finetuning_with_monitoring.sh qwen25

# 프로파일 파일은 자동으로 profiles/ 디렉토리에 저장됨
# 형식: {MODEL_NAME}_{TIMESTAMP}.nsys-rep
```

### 프로파일 파일 관리

- **최대 보관 개수**: 10개 (설정 가능)
- **자동 정리**: 오래된 파일 자동 삭제
- **수동 정리**: 필요시 `profiles/` 디렉토리에서 직접 삭제

## 📈 Weights & Biases (W&B) 통합

### ✅ 권장 사항

**네, Weights & Biases를 사용하는 것이 매우 좋습니다!**

### 장점

1. **실시간 모니터링**
   - Loss, learning rate, GPU 사용률 등 실시간 추적
   - 웹 대시보드에서 확인 가능

2. **실험 비교**
   - 여러 실험을 쉽게 비교
   - 하이퍼파라미터 변경 효과 시각화

3. **자동 로깅**
   - Transformers Trainer와 자동 통합
   - 추가 코드 최소화

4. **협업**
   - 팀원과 실험 결과 공유
   - 실험 히스토리 자동 저장

### 설정 방법

#### 1. W&B 설치

```bash
pip install wandb
```

#### 2. W&B 로그인

```bash
wandb login
# API 키 입력 (https://wandb.ai/settings에서 확인)
```

또는 환경변수로 설정:

```bash
export WANDB_API_KEY='your_api_key_here'
```

#### 3. 설정 파일 업데이트

`ko_centaur/configs/training_qwen25_32b_qlora.yaml`:

```yaml
misc:
  use_wandb: true
  wandb_project: "centaur-training"
  report_to: ["wandb"]
```

#### 4. 학습 시작

```bash
bash scripts/start_centaur_finetuning_with_monitoring.sh qwen25
```

### W&B 대시보드

- **웹 대시보드**: https://wandb.ai
- **프로젝트**: `centaur-training` (설정 가능)
- **실시간 메트릭**: Loss, learning rate, GPU 사용률 등

### 로깅되는 정보

- **학습 메트릭**: Loss, learning rate, gradient norm
- **하이퍼파라미터**: Batch size, learning rate, LoRA rank 등
- **시스템 정보**: GPU 사용률, 메모리 사용량
- **모델 정보**: 모델 이름, quantization 설정

## 🔧 통합 모니터링 설정

### 완전 자동화된 모니터링

새로운 스크립트는 다음을 자동으로 수행합니다:

1. ✅ GPU 프로파일링 (nsys)
2. ✅ W&B 실험 추적
3. ✅ 프로파일 파일 관리
4. ✅ 로그 파일 관리

### 사용 예시

```bash
# 1. W&B API 키 설정 (한 번만)
export WANDB_API_KEY='your_api_key'

# 2. 학습 시작 (프로파일링 + W&B 자동 포함)
bash scripts/start_centaur_finetuning_with_monitoring.sh qwen25

# 3. 모니터링
# - 로그: tail -f logs/qwen25-32b-qlora_*.log
# - W&B: https://wandb.ai
# - GPU: watch -n 1 nvidia-smi
```

## 📊 모니터링 비교

| 기능 | GPU 프로파일링 (nsys) | W&B | 이메일 알림 |
|------|---------------------|-----|------------|
| 실시간 메트릭 | ❌ | ✅ | ⚠️ (10분 간격) |
| GPU 성능 분석 | ✅ | ⚠️ (기본) | ❌ |
| 실험 비교 | ❌ | ✅ | ❌ |
| 웹 대시보드 | ❌ | ✅ | ❌ |
| 오버헤드 | 낮음 | 매우 낮음 | 낮음 |
| 디스크 사용 | 중간 | 낮음 | 낮음 |

## 💡 권장 설정

### 최적 모니터링 구성

1. **GPU 프로파일링**: 모든 학습에 활성화 (오버헤드 낮음)
2. **W&B**: 모든 학습에 활성화 (실험 추적 필수)
3. **이메일 알림**: 장기 실행 시 활성화 (10분 간격)

### 설정 예시

```bash
# .bashrc 또는 환경 설정
export WANDB_API_KEY='your_api_key'
export WANDB_PROJECT='centaur-training'

# 학습 시작
bash scripts/start_centaur_finetuning_with_monitoring.sh qwen25
```

## 🔗 참고 자료

- W&B 문서: https://docs.wandb.ai
- nsys 문서: https://docs.nvidia.com/nsight-systems/
- Transformers W&B 통합: https://huggingface.co/docs/transformers/integrations#wandb

