#!/usr/bin/env python3
"""
Calibration Scan Agent 실행
"""

from agents import CalibScanAgent

def main():
    print("\n" + "="*70)
    print("⚡ Calibration Scan Agent")
    print("="*70)
    
    # Agent 생성
    # 에너지는 처음 대화에서 입력받도록 설정 (beam_energy=None)
    agent = CalibScanAgent(
        beam_energy=None,
        target_events=None,
        use_base_model=False  # Fine-tuned 모델 사용 시도 (없으면 base로 자동 전환)
    )
    
    # 실행
    with agent:
        agent.run()

if __name__ == "__main__":
    main()
