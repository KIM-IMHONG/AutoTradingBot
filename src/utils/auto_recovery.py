import asyncio
import logging
import traceback
import json
import os
import time
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Suppress urllib3 warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# OpenAI import를 선택사항으로 만들기
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)

class AutoRecoverySystem:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.recovery_history = []
        self.error_patterns = {}
        self.auto_fix_enabled = os.getenv('AUTO_FIX_ENABLED', 'False').lower() == 'true'
        self.notification_webhook = os.getenv('RECOVERY_WEBHOOK_URL')
        self.openai_enabled = OPENAI_AVAILABLE and bool(self.openai_api_key)
        
        if self.openai_enabled:
            openai.api_key = self.openai_api_key
            logger.info("🤖 Auto-recovery system initialized with AI analysis")
        else:
            if not OPENAI_AVAILABLE:
                logger.warning("⚠️ OpenAI package not installed. Auto-recovery will use fallback analysis only.")
            elif not self.openai_api_key:
                logger.warning("⚠️ OpenAI API key not set. Auto-recovery will use fallback analysis only.")
            else:
                logger.info("📋 Auto-recovery system initialized with fallback analysis")
        
    async def analyze_error_with_ai(self, error_info: Dict) -> Dict:
        """AI API를 사용해서 에러 분석 및 해결책 제안"""
        try:
            if not self.openai_enabled:
                logger.info("Using fallback error analysis (OpenAI not available)")
                return self.fallback_error_analysis(error_info)
            
            # 에러 정보를 AI가 분석할 수 있는 형태로 정리
            error_context = f"""
            Trading Bot Error Analysis Request:
            
            Error Type: {error_info.get('error_type', 'Unknown')}
            Error Message: {error_info.get('error_message', '')}
            Traceback: {error_info.get('traceback', '')}
            Timestamp: {error_info.get('timestamp', '')}
            Function: {error_info.get('function', '')}
            
            Recent Logs:
            {error_info.get('recent_logs', '')}
            
            System Info:
            - Python Version: {sys.version}
            - Recovery Attempts: {self.recovery_attempts}
            
            Please analyze this error and provide:
            1. Root cause analysis
            2. Suggested fix (if it's a simple configuration issue)
            3. Risk assessment (LOW/MEDIUM/HIGH)
            4. Recommended action (RESTART/FIX_CONFIG/MANUAL_INTERVENTION)
            
            Return response in JSON format:
            {
                "analysis": "detailed analysis",
                "root_cause": "identified cause",
                "suggested_fix": "specific fix instructions",
                "risk_level": "LOW/MEDIUM/HIGH",
                "action": "RESTART/FIX_CONFIG/MANUAL_INTERVENTION",
                "confidence": 0.0-1.0
            }
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer specializing in trading bot error analysis and recovery. Provide accurate, safe solutions."},
                    {"role": "user", "content": error_context}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content
            
            # JSON 파싱 시도
            try:
                analysis_result = json.loads(ai_response)
            except json.JSONDecodeError:
                # JSON이 아닌 경우 기본 구조로 래핑
                analysis_result = {
                    "analysis": ai_response,
                    "root_cause": "Unable to parse structured response",
                    "suggested_fix": "",
                    "risk_level": "MEDIUM",
                    "action": "MANUAL_INTERVENTION",
                    "confidence": 0.5
                }
            
            logger.info(f"🤖 AI Analysis completed: {analysis_result.get('action', 'UNKNOWN')}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return self.fallback_error_analysis(error_info)
    
    def fallback_error_analysis(self, error_info: Dict) -> Dict:
        """AI API 사용 불가시 기본 에러 분석"""
        error_msg = error_info.get('error_message', '').lower()
        error_type = error_info.get('error_type', '')
        
        # 일반적인 에러 패턴 분석
        if 'connection' in error_msg or 'timeout' in error_msg:
            return {
                "analysis": "Network connectivity issue detected",
                "root_cause": "Connection timeout or network error",
                "suggested_fix": "Restart with exponential backoff",
                "risk_level": "LOW",
                "action": "RESTART",
                "confidence": 0.8
            }
        elif 'api key' in error_msg or 'authentication' in error_msg:
            return {
                "analysis": "API authentication issue",
                "root_cause": "Invalid or expired API credentials",
                "suggested_fix": "Check API key configuration",
                "risk_level": "MEDIUM",
                "action": "MANUAL_INTERVENTION",
                "confidence": 0.9
            }
        elif 'memory' in error_msg or 'out of memory' in error_msg:
            return {
                "analysis": "Memory shortage detected",
                "root_cause": "Insufficient system memory",
                "suggested_fix": "Restart with memory cleanup",
                "risk_level": "MEDIUM",
                "action": "RESTART",
                "confidence": 0.7
            }
        elif 'nan' in error_msg or 'null' in error_msg:
            return {
                "analysis": "Data integrity issue",
                "root_cause": "Invalid data values (NaN/Null)",
                "suggested_fix": "Reset data and restart",
                "risk_level": "LOW",
                "action": "RESTART",
                "confidence": 0.8
            }
        else:
            return {
                "analysis": "Unknown error pattern",
                "root_cause": "Unrecognized error type",
                "suggested_fix": "",
                "risk_level": "HIGH",
                "action": "MANUAL_INTERVENTION",
                "confidence": 0.3
            }
    
    async def handle_error(self, exception: Exception, context: Dict = None):
        """에러 발생시 자동 처리"""
        try:
            self.recovery_attempts += 1
            
            if self.recovery_attempts > self.max_recovery_attempts:
                logger.error("Max recovery attempts reached. Manual intervention required.")
                await self.notify_admin("Max recovery attempts exceeded")
                return False
            
            # 에러 정보 수집
            error_info = {
                'error_type': type(exception).__name__,
                'error_message': str(exception),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat(),
                'function': context.get('function', '') if context else '',
                'recent_logs': self.get_recent_logs()
            }
            
            logger.error(f"Error detected (Attempt {self.recovery_attempts}): {error_info['error_message']}")
            
            # AI 분석 요청
            analysis = await self.analyze_error_with_ai(error_info)
            
            # 분석 결과를 히스토리에 저장
            self.recovery_history.append({
                'timestamp': datetime.now().isoformat(),
                'error_info': error_info,
                'analysis': analysis,
                'attempt': self.recovery_attempts
            })
            
            # 리스크 레벨에 따른 처리
            if analysis['risk_level'] == 'HIGH' or analysis['confidence'] < 0.5:
                logger.warning("High risk or low confidence. Requesting manual intervention.")
                await self.notify_admin(f"High risk error detected: {analysis['analysis']}")
                return False
            
            # 자동 수정 시도
            if self.auto_fix_enabled and analysis['action'] in ['RESTART', 'FIX_CONFIG']:
                success = await self.execute_fix(analysis, error_info)
                if success:
                    logger.info("Auto-recovery successful!")
                    await self.notify_admin(f"Auto-recovery successful: {analysis['suggested_fix']}")
                    self.recovery_attempts = 0  # 성공시 카운터 리셋
                    return True
                else:
                    logger.error("Auto-recovery failed")
                    await self.notify_admin("Auto-recovery failed")
                    return False
            else:
                logger.info("Manual intervention required or auto-fix disabled")
                await self.notify_admin(f"Manual intervention needed: {analysis['analysis']}")
                return False
                
        except Exception as recovery_error:
            logger.error(f"Error in recovery system: {recovery_error}")
            await self.notify_admin(f"Recovery system error: {recovery_error}")
            return False
    
    async def execute_fix(self, analysis: Dict, error_info: Dict) -> bool:
        """AI 분석 결과에 따른 자동 수정 실행"""
        try:
            action = analysis['action']
            
            if action == 'RESTART':
                logger.info("Executing restart strategy...")
                
                # 메모리 클리어
                import gc
                gc.collect()
                
                # 짧은 대기
                await asyncio.sleep(5)
                
                # 프로세스 재시작 (이 부분은 외부 스크립트나 systemd 등으로 처리)
                logger.info("Restart request completed. External restart required.")
                return True
                
            elif action == 'FIX_CONFIG':
                logger.info("Attempting configuration fix...")
                
                # 설정 파일 백업
                self.backup_config()
                
                # 간단한 설정 수정 (예: 타임아웃 증가, 재시도 횟수 조정)
                config_fixes = self.generate_config_fixes(analysis, error_info)
                
                for fix in config_fixes:
                    await self.apply_config_fix(fix)
                
                return True
                
            else:
                logger.warning(f"Unsupported auto-fix action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing fix: {e}")
            return False
    
    def generate_config_fixes(self, analysis: Dict, error_info: Dict) -> List[Dict]:
        """분석 결과에 따른 설정 수정 사항 생성"""
        fixes = []
        
        error_msg = error_info.get('error_message', '').lower()
        
        if 'timeout' in error_msg:
            fixes.append({
                'file': 'config/settings.py',
                'setting': 'REQUEST_TIMEOUT',
                'old_value': '30',
                'new_value': '60',
                'reason': 'Increase timeout to handle slow responses'
            })
        
        if 'rate limit' in error_msg:
            fixes.append({
                'file': 'config/settings.py',
                'setting': 'API_RATE_LIMIT_DELAY',
                'old_value': '1',
                'new_value': '2',
                'reason': 'Increase delay between API calls'
            })
        
        return fixes
    
    async def apply_config_fix(self, fix: Dict):
        """설정 파일 수정 적용"""
        try:
            logger.info(f"Applying config fix: {fix['reason']}")
            # 실제 구현에서는 파일을 읽고 수정하는 로직 필요
            # 안전을 위해 여기서는 로그만 남김
            logger.info(f"Would change {fix['setting']} from {fix['old_value']} to {fix['new_value']}")
        except Exception as e:
            logger.error(f"Error applying config fix: {e}")
    
    def backup_config(self):
        """설정 파일 백업"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"backups/config_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # config 폴더 백업
            subprocess.run(['cp', '-r', 'config/', str(backup_dir)], check=True)
            logger.info(f"Config backup created: {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error creating config backup: {e}")
    
    def get_recent_logs(self, lines: int = 50) -> str:
        """최근 로그 내용 가져오기"""
        try:
            # 로그 파일이 있는 경우
            log_files = ['trading_bot.log', 'app.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines_list = f.readlines()
                        return ''.join(lines_list[-lines:])
            return "No log file found"
        except Exception as e:
            return f"Error reading logs: {e}"
    
    async def notify_admin(self, message: str):
        """관리자에게 알림 전송"""
        try:
            timestamp = datetime.now().isoformat()
            notification = {
                'timestamp': timestamp,
                'message': message,
                'recovery_attempts': self.recovery_attempts,
                'system': 'AutoRecovery'
            }
            
            logger.warning(f"ADMIN NOTIFICATION: {message}")
            
            # 웹훅이나 텔레그램 등으로 알림 전송
            if self.notification_webhook:
                # 실제 구현에서는 웹훅 호출
                pass
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def get_recovery_status(self) -> Dict:
        """복구 시스템 상태 반환"""
        return {
            'recovery_attempts': self.recovery_attempts,
            'max_attempts': self.max_recovery_attempts,
            'auto_fix_enabled': self.auto_fix_enabled,
            'history_count': len(self.recovery_history),
            'last_recovery': self.recovery_history[-1]['timestamp'] if self.recovery_history else None
        }

# 전역 인스턴스
auto_recovery = AutoRecoverySystem()

# 데코레이터로 자동 복구 기능 추가
def with_auto_recovery(func):
    """함수에 자동 복구 기능을 추가하는 데코레이터"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            context = {'function': func.__name__}
            recovery_success = await auto_recovery.handle_error(e, context)
            
            if recovery_success:
                # 복구 성공시 함수 재실행
                logger.info(f"Retrying {func.__name__} after successful recovery")
                return await func(*args, **kwargs)
            else:
                # 복구 실패시 예외 재발생
                raise e
    
    return wrapper 