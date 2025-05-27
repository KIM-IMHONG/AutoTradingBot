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
