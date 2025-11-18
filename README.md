# Fashion Recommendation System 🛍️

> **코디가 얼마나 잘 어울리는지 점수를 내고, 최적의 옷 조합을 추천해주는 AI 시스템**

경량 MLP 모델 + FashionCLIP 임베딩 + 트렌드/선호도 스코어를 결합해 옷장 코디 추천, 피드백, 나눔까지 가능한 패션 AI 플랫폼입니다.

---

## 🎯 프로젝트 개요

### 핵심 아이디어
- **FashionCLIP 임베딩**: 모든 옷 이미지를 동일한 512차원 벡터 공간으로 변환
- **Compatibility MLP**: 아이템 조합의 호환성을 0~1 점수로 산출 (Polyvore 데이터로 학습)
- **다중 스코어 합성**: 호환성 + 트렌드 + 개인 취향 + TPO/날씨 + LLM 설명

### 최종 목표
- "오늘 뭐 입지?" → AI 스타일리스트가 추천 + 이유 설명
- 유저 옷장 디지털화 및 코디 기록 관리
- 지속가능한 패션: 옷 나눔/거래 연결

---

## 📊 데이터 소스

| 데이터셋 | 용도 | 특징 |
|---------|------|------|
| **Polyvore Outfit Dataset** | Compatibility MLP 학습 | 실제 유저 코디 세트 (positive) + 랜덤 조합 (negative) |
| **AI-Hub 패션 데이터** | 속성 태깅 + 트렌드 스코링 | 카테고리/색상/스타일/계절/메타정보 |
| **AI-Hub 트렌드 데이터** | 연도별 유행도 반영 | 2020~2025 패션 선호도 추이 |
| **유저 옷장 (웹캠 + 웹 이미지)** | 실제 추천 대상 | 사용자가 직접 등록한 옷 + 메타정보 |
| **컨텍스트 (날씨, TPO, 프로필)** | 필터링 + 가중치 | 기온/강수량/일정/유저 취향 |

---

## 🧠 모델 & 알고리즘

### 1. FashionCLIP (Base Model)
- **역할**: 이미지 → 512차원 임베딩
- **범위**: Polyvore 아이템 + AI-Hub 이미지 + 유저 옷 모두 동일 공간
- **학습**: 사전학습 완료 (추가 학습 X, 프롬프트 엔지니어링 위주)

### 2. Compatibility MLP (핵심)
```
Input: [emb_top ⊕ emb_bottom ⊕ |emb_top - emb_bottom|] → 1024~1536차원
Output: compat_score ∈ [0, 1]
Loss: Binary Cross Entropy
```
- Polyvore의 positive pairs (같은 outfit) vs negative pairs (다른 outfit 섞은 것)로 학습
- 경량 구조: Dense Layer + ReLU + Dropout

### 3. 추가 스코어 모듈
- **Trend Score**: 연도별 스타일 선호도 (AI-Hub 데이터)
- **TPO/Weather Score**: 날씨·일정·계절 필터링
- **Preference Score**: 유저 행동 로그 (좋아요/싫어요/착용 여부) 기반 가중치
- **최종 점수**: `final_score = w₁·compat + w₂·trend + w₃·pref + w₄·tpo`

### 4. LLM 레이어 (설명 생성)
- **모델**: GPT 시리즈
- **역할**: 추천 결과를 자연어로 설명
  - "왜 이 코디를 추천했는가"
  - "어떤 부분이 좋은가"
  - "스타일 팁"
- **학습**: 프롬프트 엔지니어링 (추가 학습 X)

---

## 🔄 학습 파이프라인 (Training)

### Polyvore 기반 Compatibility MLP 학습

```
1️⃣ 데이터 정리
   Polyvore JSON/CSV → outfit 단위 아이템 리스트 추출

2️⃣ 임베딩 생성
   모든 아이템 이미지 → FashionCLIP → 512차원 벡터 저장

3️⃣ 페어/트리플 생성
   Positive: 같은 outfit 내 조합 (label=1)
   Negative: 다른 outfit 섞은 조합 (label=0)

4️⃣ Feature 구성
   x = concat(emb₁, emb₂, |emb₁-emb₂|)

5️⃣ MLP 학습
   Input: X (feature matrix), y (label)
   Loss: BCE
   Output: compat_score

6️⃣ 평가 & 저장
   Train/Val Split → AUC, Accuracy, F1 확인
   모델 가중치 저장 (models/)
```

---

## 🚀 서비스 플로우 (Inference)

### 📱 시나리오 A: "오늘 내가 입으려는 코디 평가"

```
1. 컨텍스트 수집
   ├─ 날씨: 온도, 강수량, 체감온도
   ├─ TPO: 출근 / 데이트 / 발표 / 여행 등
   └─ 유저 옷 선택: 상의 + 하의 + 신발 + (선택사항)

2. 점수 계산
   ├─ Compatibility: MLP(emb₁, emb₂, ...)
   ├─ Trend: AI-Hub 데이터 기반
   ├─ TPO/Weather: 룰 기반 필터링
   └─ Preference: 유저 행동 로그

3. LLM 설명 생성
   "블랙 슬랙스 + 화이트 셔츠는 대비가 깔끔해서 출근룩으로 좋고,
    최근 미니멀 오피스룩 트렌드랑도 잘 맞아요.
    다만 오늘 최고 기온이 30도라 니트 조끼는 빼는 걸 추천해요."

4. 대체 코디 제안 (옵션)
   상의/하의/신발 일부 바꾼 후보 3개 추천
```

### 🔍 시나리오 B: "이 옷에 어울리는 코디 추천"

```
1. Anchor 아이템 선택
   예: "검정 슬랙스에 뭐 입지?"

2. Vector DB에서 후보 찾기
   임베딩 유사도 기반 상의/아우터/신발 추천 후보 추출

3. 조합 생성 + Compatibility MLP 적용
   모든 조합 생성 → compat_score + 추가 점수 계산

4. 상위 N개 추천 + LLM 설명
   "요 슬랙스는 포멀한 느낌이라, 화이트 셔츠/베이지 니트랑
    매치하면 깔끔한 출근룩 됩니다"
```

### 👕 옷 디지털화 파이프라인

```
1. 옷 추가 (사진 촬영 or 웹 이미지)
2. 속성 분석 (카테고리/색상/스타일/계절)
3. FashionCLIP 임베딩 생성
4. Vector DB 저장 (PostgreSQL + pgvector or FAISS)
5. 트렌드 점수 부여 (AI-Hub 데이터 매핑)
```

---

## 📁 프로젝트 구조

```
fashion-model/
├── data/                           # 데이터 (Git 제외)
│   ├── polyvore_outfits/          # Polyvore raw 데이터
│   ├── polyvore_raw/              #
│   └── processed/                 # 전처리된 데이터
├── models/                         # 학습된 모델
│   ├── mlp_image_disjoint.pkl     # 이미지 기반 MLP
│   └── mlp_text_mlp_v1.pkl        # 텍스트 기반 MLP (시도)
├── notebooks/                      # 실험 및 분석
│   ├── train_*.ipynb
│   └── *.ipynb
├── src/                           # 소스 코드
│   ├── extract_image_embeddings.py      # FashionCLIP 임베딩 추출
│   ├── extract_text_embeddings.py       # 텍스트 임베딩
│   ├── make_pairs.py                    # positive/negative 페어 생성
│   ├── make_pairs_disjoint.py           # (확장 버전)
│   ├── parse_polyvore.py                # Polyvore 데이터 파싱
│   ├── train_mlp.py                     # MLP 학습 (메인)
│   ├── train_mlp_image.py               # 이미지 기반 MLP 학습
│   ├── Rank_items.py                    # 코디 점수 계산/랭킹
│   └── ...
├── .gitignore                     # Git 제외 설정
├── requirements.txt               # 의존성 (추후 작성)
└── README.md                      # 이 파일
```

---

## 🛠️ 설정 및 실행

### 필수 라이브러리

```bash
pip install -r requirements.txt
```

**주요 라이브러리**:
- `torch`, `torchvision` - PyTorch
- `transformers` - Hugging Face (FashionCLIP)
- `scikit-learn` - 모델 평가, 스케일러
- `numpy`, `pandas` - 데이터 처리
- `jupyter` - 노트북
- `pickle` - 모델 저장/로드

### 학습 실행

```bash
# 1. FashionCLIP 임베딩 추출 (Polyvore 데이터)
python src/extract_image_embeddings.py

# 2. Positive/Negative 페어 생성
python src/make_pairs.py

# 3. MLP 모델 학습
python src/train_mlp_image.py

# 4. 모델 평가 및 저장
# (train_mlp_image.py 내에서 자동 저장)
```

### 추론 (코디 점수 계산)

```python
import pickle
from src.Rank_items import score_outfit

# 모델 로드
model = pickle.load(open('models/mlp_image_disjoint.pkl', 'rb'))

# 코디 점수 계산
items = [item1_emb, item2_emb, item3_emb]  # FashionCLIP 임베딩
score = score_outfit(items, model)
print(f"Compatibility Score: {score:.2f}")
```

---

## 📈 성능 & 메트릭

### Compatibility MLP 모델 성능 (Polyvore 데이터)

| 메트릭 | 값 |
|--------|-----|
| Train Accuracy | TBD |
| Val Accuracy | TBD |
| AUC (ROC) | TBD |
| F1 Score | TBD |

*현재 `fashion-model` 저장소로 이전 중, 세부 성능 지표는 notebooks에서 확인 가능*

---

## 🗺️ 로드맵

| 버전 | 포커스 | 목표 |
|------|--------|------|
| **V1** | Polyvore + FashionCLIP + MLP | 코디 호환성 모델 데모 |
| **V1.5** | 유저 옷장 업로드 플로우 | 자신의 옷으로 추천 받기 |
| **V2** | AI-Hub 속성 데이터 통합 | 스타일/색상 태깅 정확도 ↑ |
| **V3** | AI-Hub 트렌드 데이터 | 트렌디함 반영 (유행/올드 감점) |
| **V4** | 유저 피드백 + 나눔/거래 | 개인화 강화 + 순환 경제 |
| **V5** | LLM 레이어 강화 | "AI 스타일리스트" 경험 |

---

## 🤝 기여 가이드

현재 프로젝트 진행 상황:
- 모델 학습 ✅
- 추론 로직 정리 중
- 백엔드 (FastAPI/Django) 구성 예정
- 프론트엔드 (React/Vue) 연동 예정

개선 제안 및 이슈는 GitHub Issues에서 등록해주세요!

---

## 📄 라이선스

(라이선스 정보 추가 예정)

---

## 📞 연락처

질문이나 피드백은 GitHub Issues에서 부탁드립니다.

---

**마지막 업데이트**: 2025-11-18
