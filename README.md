# Binance Futures Trading Bot

자동화된 암호화폐 선물 거래 봇입니다. 기술적 분석, 뉴스 분석, 리스크 관리 기능을 포함하고 있습니다.

## 주요 기능

- 실시간 차트 데이터 수집 및 분석 (1분봉)
- RSI + MACD + EMA 기반 매매 신호 생성
- 뉴스, 트위터, 공시 데이터 수집 및 감성 분석
- 바이낸스 API를 통한 실제 매매 실행
- 텔레그램을 통한 실시간 알림

## 설치 방법

1. 저장소 클론:

```bash
git clone [repository-url]
cd autoTrading
```

2. 가상환경 생성 및 활성화:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 의존성 패키지 설치:

```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
   `.env` 파일을 생성하고 다음 변수들을 설정:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
NEWS_API_KEY=your_news_api_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
```

## 실행 방법

```bash
python main.py
```

## 프로젝트 구조

```
autoTrading/
├── config/
│   ├── __init__.py
│   └── settings.py           # 설정 파일
├── src/
│   ├── data/
│   │   ├── binance_client.py # 바이낸스 API 연동
│   │   └── news_collector.py # 뉴스 수집
│   ├── analysis/
│   │   ├── technical.py      # 기술적 분석
│   │   └── sentiment.py      # 감성 분석
│   ├── trading/
│   │   ├── strategy.py       # 매매 전략
│   │   └── executor.py       # 주문 실행
│   └── utils/
│       ├── logger.py         # 로깅
│       └── telegram_bot.py   # 텔레그램 봇
├── main.py                   # 메인 실행 파일
├── requirements.txt          # 의존성 패키지
└── README.md                 # 프로젝트 문서
```

## 주의사항

- 이 봇은 실제 자금을 사용하여 거래를 실행합니다. 충분한 테스트 후 사용하세요.
- API 키는 절대 공개하지 마세요.
- 거래 전략과 리스크 관리 설정을 신중하게 검토하세요.

## 라이선스

MIT License

# AutoTradingBot

바이낸스 선물 거래를 위한 AI 기반 자동 트레이딩 봇

## 🚀 주요 기능

- **기술적 분석**: RSI, MACD, EMA, Stochastic, Bollinger Bands, ADX 등
- **뉴스 감정 분석**: 실시간 뉴스 기반 시장 감정 분석
- **동적 레버리지**: 시장 변동성과 고점/저점에 따른 자동 레버리지 조정
- **다양한 시장 전략**: 폭락/폭등/횡보/강한추세별 최적화된 매매 전략
- **리스크 관리**: 동적 손절/익절, 일일 손실 제한
- **🤖 AI 자동 복구 시스템**: OpenAI API를 활용한 에러 자동 분석 및 수정

## 🛡️ AI 자동 복구 시스템

### 기능 소개

프로그램 실행 중 에러가 발생하면 AI가 자동으로 분석하고 해결책을 제시하여 봇을 자동으로 복구합니다.

### 복구 시스템 작동 방식

1. **에러 감지**: 예외 발생시 자동으로 에러 정보 수집
2. **AI 분석**: OpenAI GPT-4를 사용해 에러 원인 분석 및 해결책 제시
3. **리스크 평가**: 자동 수정의 안전성 평가 (LOW/MEDIUM/HIGH)
4. **자동 수정**: 안전한 경우 자동으로 설정 조정 또는 재시작
5. **관리자 알림**: 위험한 에러나 수동 개입 필요시 즉시 알림

### 지원하는 복구 유형

#### 자동 복구 가능 (LOW/MEDIUM 리스크)

- **네트워크 연결 오류**: 자동 재시작 및 재연결
- **API 타임아웃**: 타임아웃 설정 자동 조정
- **메모리 부족**: 메모리 정리 후 재시작
- **데이터 무결성 오류**: 데이터 리셋 후 재시작
- **API 레이트 제한**: 요청 간격 자동 조정

#### 수동 개입 필요 (HIGH 리스크)

- **API 키 인증 오류**: 수동으로 API 키 확인 필요
- **코드 로직 오류**: 개발자 개입 필요
- **알 수 없는 오류**: 상세 분석 필요

### 환경변수 설정

```bash
# 자동 복구 시스템 활성화
AUTO_RECOVERY_ENABLED=True

# AI 자동 수정 활성화 (신중하게 사용)
AUTO_FIX_ENABLED=False

# OpenAI API 키 (필수)
OPENAI_API_KEY=your_openai_api_key_here

# 알림 웹훅 URL (선택사항)
RECOVERY_WEBHOOK_URL=your_discord_or_slack_webhook_url
```

### 사용 예시

```python
from src.utils.auto_recovery import with_auto_recovery

# 함수에 자동 복구 기능 추가
@with_auto_recovery
async def risky_trading_function():
    # 거래 로직
    result = await execute_trade()
    return result

# 메인 루프에서 자동 복구 시스템 사용
bot = TradingBot()
await bot.run_with_auto_recovery()  # 자동 복구 활성화
```

### 안전성 고려사항

1. **단계적 활성화**: 처음에는 `AUTO_FIX_ENABLED=False`로 시작
2. **모니터링**: 복구 시스템 로그를 주기적으로 확인
3. **백업**: 설정 파일이 자동으로 백업됨
4. **제한**: 최대 3회 복구 시도 후 수동 개입 요청
5. **알림**: 모든 복구 시도에 대해 관리자에게 알림

### 복구 히스토리 확인

```python
# 복구 시스템 상태 확인
status = auto_recovery.get_recovery_status()
print(f"복구 시도 횟수: {status['recovery_attempts']}")
print(f"마지막 복구 시간: {status['last_recovery']}")
```

## 📋 기본 설정

### 필수 환경변수

```bash
# 바이낸스 API
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret

# 텔레그램 알림
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 자동 복구 (권장)
OPENAI_API_KEY=your_openai_key
```

### 선택적 환경변수

```bash
# 거래 설정
TRADING_SYMBOLS=BTCUSDT
MAX_LEVERAGE=50
POSITION_SIZE=100
AUTO_FIX_ENABLED=False

# 리스크 관리
STOP_LOSS_PERCENTAGE=1.0
TAKE_PROFIT_PERCENTAGE=2.0
MAX_DAILY_LOSS=5.0
```

## 🚀 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키 등을 설정

# 봇 실행
python main.py
```

## 📊 모니터링

### 로그 확인

- 봇 상태: `INFO` 레벨 로그
- 자동 복구: `WARNING` 레벨 로그
- 에러 상황: `ERROR` 레벨 로그

### 텔레그램 알림

- 거래 체결 알림
- 포지션 변경 알림
- 시스템 에러 알림
- **자동 복구 상태 알림**

## 🔧 고급 설정

### 자동 복구 시스템 세부 설정

```python
# 복구 시도 횟수 조정
auto_recovery.max_recovery_attempts = 5

# 복구 조건 커스터마이징
auto_recovery.risk_threshold = 0.7  # 신뢰도 임계값

# 알림 방식 설정
auto_recovery.notification_webhook = "your_webhook_url"
```

### 커스텀 복구 로직 추가

```python
# 특정 에러에 대한 커스텀 복구 로직
@auto_recovery.register_custom_handler('SpecificError')
async def handle_specific_error(error_info):
    # 커스텀 복구 로직
    return True  # 성공시 True 반환
```

## ⚠️ 주의사항

1. **테스트 환경에서 먼저 검증**: `TEST_MODE=True`로 설정하여 테스트
2. **자동 수정 신중 사용**: `AUTO_FIX_ENABLED`는 충분한 검증 후 활성화
3. **API 키 보안**: 환경변수로 관리하고 .env 파일을 버전 관리에서 제외
4. **정기적인 모니터링**: 자동 복구 시스템도 완벽하지 않으므로 주기적 확인 필요
5. **백업 유지**: 중요한 설정이나 포지션 정보는 별도 백업 권장

## 🔍 문제 해결

### 자동 복구 시스템 관련

```bash
# 복구 시스템 로그 확인
grep "AutoRecovery" logs/trading_bot.log

# 복구 히스토리 확인
python -c "from src.utils.auto_recovery import auto_recovery; print(auto_recovery.get_recovery_status())"

# OpenAI API 연결 테스트
python -c "import openai; openai.api_key='your_key'; print('API OK')"
```

### 일반적인 문제

1. **API 연결 오류**: API 키와 네트워크 연결 확인
2. **인증 실패**: 바이낸스 API 권한 설정 확인
3. **메모리 부족**: 시스템 리소스 확인 및 정리
4. **데이터 오류**: 데이터 소스 및 네트워크 상태 확인

## 📞 지원

- 자동 복구 실패시: 로그 파일과 함께 이슈 제출
- 기능 요청: GitHub Issues 활용
- 긴급 상황: 텔레그램 알림 확인

---

**⚡ AI 자동 복구 시스템으로 24/7 안정적인 거래를 경험하세요!**
