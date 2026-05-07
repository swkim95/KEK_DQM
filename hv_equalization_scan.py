#!/usr/bin/env python3
"""
HV Equalization Scan Controller
config를 Python에서 한 번만 받고, 타워를 순서대로 돌며 HVEqualizationTowerAgent를 호출.
"""

from tools.hv_equalization_tool import hv_equalization_start
from agents.hv_equalization_agent import HVEqualizationAgent

TOWER_ORDER = ["T1", "T2", "T3", "T6", "T5", "T4", "T7", "T8", "T9"]


def main():
    print("\n" + "=" * 60)
    print("⚡ HV Equalization Scan")
    print("=" * 60)

    # Config 수집 (한 번만)
    beam_energy = float(input("빔 에너지 (GeV): ").strip())
    target_events = int(input("이벤트 수: ").strip())
    target_adc = float(input("목표 peakADC: ").strip())

    # HV Equalization 세션 초기화 (한 번만)
    result = hv_equalization_start.invoke({
        "target_c": target_adc,
        "target_s": target_adc,
        "tower": TOWER_ORDER[0],
    }) if hasattr(hv_equalization_start, "invoke") else hv_equalization_start(
        target_c=target_adc, target_s=target_adc, tower=TOWER_ORDER[0]
    )
    print(result)

    # 타워 루프
    for i, tower in enumerate(TOWER_ORDER):
        print(f"\n[{i + 1}/{len(TOWER_ORDER)}] {tower} 타워 시작")
        agent = HVEqualizationAgent(
            tower=tower,
            beam_energy=beam_energy,
            target_events=target_events,
            target_adc=target_adc,
            use_base_model=False,
        )
        with agent:
            agent.run()
        print(f"✅ {tower} 완료")

    print("\n🎉 모든 타워 HV Equalization 완료!")


if __name__ == "__main__":
    main()
