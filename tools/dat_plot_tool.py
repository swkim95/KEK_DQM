#!/usr/bin/env python3
"""
DATPlotTool
-----------
.dat 파일을 직접 파싱해 Waveform(Avg) / PeakADC / IntADC를 PyROOT로 플롯.
C++ DQM 프로그램 (draw_Avg.cc, draw_peakADC.cc, draw_intADC.cc)과 동일한 스타일.

바이너리 포맷 (TBread.cc 기준):
  - 이벤트 크기  : 65536 bytes
  - Header       : 64 bytes
  - ADC 데이터   : 32736 × int16  →  (1023 samples) × (32 channels)
  - 레이아웃     : adc[sample * 32 + ch]
  - Pedestal     : mean(waveform[1:101])
  - 신호 극성    : 하향 (baseline ~3500, 신호는 내려감)
  - PeakADC      : max(ped - waveform[start:end])
  - IntADC       : sum(ped - waveform[start:end])
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from .base_tool import BaseTool
    from .config_loader import get_path_config, get_data_directory, load_config
    _HAS_BASE = True
except ImportError:
    _HAS_BASE = False
    # 모듈을 standalone (CLI) 로 실행할 때도 config 사용 가능하도록 fallback
    try:
        import sys
        from pathlib import Path as _PathFallback
        sys.path.insert(0, str(_PathFallback(__file__).resolve().parent.parent))
        from tools.config_loader import get_data_directory  # type: ignore
    except Exception:
        get_data_directory = None  # type: ignore

# ROOT를 모듈 로드 시점에 import하고 즉시 배치 모드 설정.
# SetBatch(True)는 어떤 TCanvas/GUI 객체 생성보다 먼저 호출되어야 함.
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetStatFormat("6.6g")
    _HAS_ROOT = True
except ImportError:
    _HAS_ROOT = False


# ─────────────────────────────────────────────
# 파일 포맷 상수
# ─────────────────────────────────────────────
EVENT_SIZE   = 65536
HEADER_SIZE  = 64
ADC_WORDS    = (EVENT_SIZE - HEADER_SIZE) // 2   # 32736
N_CHANNELS   = 32
N_SAMPLES    = ADC_WORDS // N_CHANNELS            # 1023


# ─────────────────────────────────────────────
# 저수준 파싱
# ─────────────────────────────────────────────
def _read_event(f) -> Optional[np.ndarray]:
    """한 이벤트를 읽어 (32, 1023) int16 배열로 반환. EOF이면 None."""
    header = f.read(HEADER_SIZE)
    if len(header) < HEADER_SIZE:
        return None
    raw = f.read(ADC_WORDS * 2)
    if len(raw) < ADC_WORDS * 2:
        return None
    adc = np.frombuffer(raw, dtype=np.int16).reshape(N_SAMPLES, N_CHANNELS).T
    return adc  # shape: (32, 1023)


def _count_events(dat_file: Path) -> int:
    return dat_file.stat().st_size // EVENT_SIZE


def _read_events(dat_file: Path, max_events: int) -> List[np.ndarray]:
    """max_events 개의 이벤트를 (32, 1023) 배열 리스트로 반환.
    max_events == -1 이면 파일의 모든 이벤트를 읽음."""
    events = []
    limit = max_events if max_events != -1 else float("inf")
    with open(dat_file, "rb") as f:
        while len(events) < limit:
            ev = _read_event(f)
            if ev is None:
                break
            events.append(ev)
    return events


# ─────────────────────────────────────────────
# 분석 함수
# ─────────────────────────────────────────────
def _ped(wf: np.ndarray) -> float:
    """Pedestal: wf[1:101] 평균 (C++ getPed 와 동일)."""
    return float(np.mean(wf[1:101]))


def _peak_adc(wf: np.ndarray, start: int, end: int) -> float:
    """PeakADC = max(ped - wf[start:end])"""
    ped = _ped(wf)
    return float(np.max(ped - wf[start:end].astype(np.float64)))


def _int_adc(wf: np.ndarray, start: int, end: int) -> float:
    """IntADC = sum(ped - wf[start:end])"""
    ped = _ped(wf)
    return float(np.sum(ped - wf[start:end].astype(np.float64)))


# ─────────────────────────────────────────────
# 데이터 파일 경로 탐색
# ─────────────────────────────────────────────
def _resolve_base_dirs(data_dir: Optional[str]) -> List[str]:
    """탐색할 데이터 베이스 디렉터리 목록을 결정.

    우선순위:
      1) 호출자가 명시한 ``data_dir``
      2) ``config_general.yml`` 의 ``BaseDirectory``  (DQM/HV/PositionScan과 통일)
    """
    if data_dir:
        return [data_dir]
    if get_data_directory is not None:
        try:
            return [get_data_directory()]
        except Exception:
            pass
    return []


def _find_dat_files(run_number: int, mid: int, data_dir: Optional[str] = None) -> List[Path]:
    """FILE_0, FILE_1, … 모두 반환 (파일 번호 순 정렬)."""
    for base in _resolve_base_dirs(data_dir):
        if not base:
            continue
        run_dir = (Path(base)
                   / f"Run_{run_number}"
                   / f"Run_{run_number}_Wave"
                   / f"Run_{run_number}_Wave_MID_{mid}")
        if run_dir.exists():
            files = sorted(run_dir.glob(f"Run_{run_number}_Wave_MID_{mid}_FILE_*.dat"))
            if files:
                return files
    return []


# ─────────────────────────────────────────────
# 플롯 함수
# ─────────────────────────────────────────────

# draw_Avg.cc 의 myColorPalette 와 동일하게 가시 스펙트럼에서 고른 색
_LINE_COLORS = [4, 2, 3, 6, 7, 8, 9, 28]  # kBlue, kRed, kGreen, kMagenta, ...


def _plot_waveform(events: List[np.ndarray], channels: List[int],
                   run_number: int, output_dir: Path) -> Path:
    """채널별 평균 파형 (draw_Avg.cc 방식).
    TH1F: 1000 bins, 0–1000, ylabel=ADC, Y range 1000–4096.
    GetAvg: 각 bin에 waveform/nEvent 누적 → 평균선."""
    ROOT.gStyle.SetPalette(ROOT.kVisibleSpectrum)
    ROOT.gStyle.SetOptStat(0)   # draw_Avg.cc: do not need stat box

    n_events = len(events)
    n_bins   = 1000

    c = ROOT.TCanvas("c_wave", "c_wave", 1000, 800)
    c.cd()

    leg = ROOT.TLegend(0.75, 0.2, 0.9, 0.4)

    plots = []
    for idx, ch in enumerate(channels):
        h = ROOT.TH1F(f"h_wave_ch{ch}", f";bin;ADC", n_bins, 0, 1000)
        h.SetTitle(f"Run {run_number} -Avg Waveform")

        # draw_Avg.cc GetAvg: average over all events per bin
        avg = np.mean(
            np.stack([ev[ch][:n_bins].astype(np.float64) for ev in events]),
            axis=0
        )
        for b in range(n_bins):
            h.SetBinContent(b + 1, avg[b])  # TH1F bins are 1-indexed

        color = _LINE_COLORS[idx % len(_LINE_COLORS)]
        h.SetLineWidth(2)
        h.SetLineColor(color)
        h.GetYaxis().SetRangeUser(1000, 4096)

        c.cd()
        if idx == 0:
            h.Draw("Hist")
        else:
            h.Draw("Hist sames")

        leg.AddEntry(h, f"Ch {ch}", "l")
        c.Update()
        plots.append(h)  # keep reference

    leg.Draw("sames")
    c.Update()

    ch_str = "_".join(map(str, channels))
    out = output_dir / f"Run{run_number}_waveform_ch{ch_str}.png"
    c.SaveAs(str(out))
    return out


def _plot_peakADC(values_per_ch: Dict[int, List[float]], channels: List[int],
                  run_number: int, output_dir: Path,
                  start_bin: int, end_bin: int) -> Path:
    """PeakADC 히스토그램 (draw_peakADC.cc 방식).
    TH1F: 4096 bins, 0–4096, xlabel=peakADC, ylabel=nEvents."""
    ROOT.gStyle.SetPalette(ROOT.kVisibleSpectrum)
    ROOT.gStyle.SetOptStat("emr")   # entries, mean, RMS

    c = ROOT.TCanvas("c_peak", "c_peak", 1000, 800)
    c.cd()

    leg = ROOT.TLegend(0.75, 0.2, 0.9, 0.4)

    plots = []
    for idx, ch in enumerate(channels):
        h = ROOT.TH1F(f"h_peak_ch{ch}",
                      f"Run {run_number} -peakADC  [bins {start_bin}–{end_bin}];peakADC;nEvents",
                      4096, 0, 4096)
        for v in values_per_ch.get(ch, []):
            h.Fill(v)

        color = _LINE_COLORS[idx % len(_LINE_COLORS)]
        h.SetLineColor(color)
        h.SetLineWidth(2)

        c.cd()
        if idx == 0:
            h.Draw("hist")
        else:
            h.Draw("hist sames")

        leg.AddEntry(h, f"Ch {ch}", "l")
        c.Update()
        plots.append(h)

    leg.Draw("sames")
    c.Update()

    ch_str = "_".join(map(str, channels))
    out = output_dir / f"Run{run_number}_peakADC_ch{ch_str}.png"
    c.SaveAs(str(out))
    return out


def _plot_intADC(values_per_ch: Dict[int, List[float]], channels: List[int],
                 run_number: int, output_dir: Path,
                 start_bin: int, end_bin: int) -> Path:
    """IntADC 히스토그램 (draw_intADC.cc 방식).
    TH1F: 840 bins, -18000–350000, xlabel=intADC, ylabel=nEvents."""
    ROOT.gStyle.SetPalette(ROOT.kVisibleSpectrum)
    ROOT.gStyle.SetOptStat("emr")   # entries, mean, RMS

    c = ROOT.TCanvas("c_int", "c_int", 1000, 800)
    c.cd()

    leg = ROOT.TLegend(0.75, 0.2, 0.9, 0.4)

    plots = []
    for idx, ch in enumerate(channels):
        h = ROOT.TH1F(f"h_int_ch{ch}",
                      f"Run {run_number} -intADC  [bins {start_bin}–{end_bin}];intADC;nEvents",
                      840, -18000, 350000)
        for v in values_per_ch.get(ch, []):
            h.Fill(v)

        color = _LINE_COLORS[idx % len(_LINE_COLORS)]
        h.SetLineColor(color)
        h.SetLineWidth(2)

        c.cd()
        if idx == 0:
            h.Draw("hist")
        else:
            h.Draw("hist sames")

        leg.AddEntry(h, f"Ch {ch}", "l")
        c.Update()
        plots.append(h)

    leg.Draw("sames")
    c.Update()

    ch_str = "_".join(map(str, channels))
    out = output_dir / f"Run{run_number}_intADC_ch{ch_str}.png"
    c.SaveAs(str(out))
    return out


# ─────────────────────────────────────────────
# 메인 실행 함수
# ─────────────────────────────────────────────
def run_dat_plot(
    run_number: int,
    mid: int = 8,
    channels: Optional[List[int]] = None,
    start_bin: int = 150,
    end_bin: int = 350,
    max_events: int = -1,
    mode: str = "all",
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    run_number  : Run 번호
    mid         : MID 번호 (default 8)
    channels    : 플롯할 채널 인덱스 목록 0-31 (default [0])
    start_bin   : PeakADC/IntADC 계산 시작 bin (default 150)
    end_bin     : PeakADC/IntADC 계산 끝 bin   (default 350)
    max_events  : 읽을 최대 이벤트 수 (-1이면 파일의 모든 이벤트, default -1)
    mode        : "wave" | "peakADC" | "intADC" | "all"
    data_dir    : .dat 파일 베이스 디렉토리 (None이면 자동 탐색)
    output_dir  : PNG 저장 디렉토리 (None이면 DQM/output/dat_plots)
    """
    if channels is None:
        channels = [0]

    # 출력 디렉토리
    if output_dir is None:
        try:
            dqm_dir = get_path_config("DqmDir") if _HAS_BASE else "/Users/yhep/autoTB/DQM"
        except Exception:
            dqm_dir = "/Users/yhep/autoTB/DQM"
        out_dir = Path(dqm_dir) / "output" / "dat_plots"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dat 파일 탐색 (FILE_0, FILE_1, … 모두)
    searched_bases = _resolve_base_dirs(data_dir)
    dat_files = _find_dat_files(run_number, mid, data_dir)
    if not dat_files:
        return {"status": "error",
                "message": f"Run {run_number} MID {mid} .dat 파일을 찾을 수 없습니다.\n"
                           f"탐색 경로(BaseDirectory): {searched_bases or '(미설정)'}"}

    # 각 파일의 이벤트 수 집계
    file_counts = [_count_events(f) for f in dat_files]
    total_events = sum(file_counts)
    n_read = total_events if max_events == -1 else min(max_events, total_events)

    file_summary = ", ".join(f"FILE_{i}:{c}" for i, c in enumerate(file_counts))
    print(f"📂 Run {run_number} MID {mid}: {len(dat_files)}개 파일 ({file_summary})")
    print(f"   총 {total_events}개 중 {n_read}개 읽는 중...")

    # 모든 파일에서 순서대로 읽기
    events: List[np.ndarray] = []
    remaining = n_read
    for dat_file in dat_files:
        if remaining <= 0:
            break
        chunk = _read_events(dat_file, remaining if max_events != -1 else -1)
        events.extend(chunk)
        remaining -= len(chunk)

    if not events:
        return {"status": "error", "message": "이벤트를 읽지 못했습니다."}

    saved_files = []

    # ── Waveform (Avg) ──
    if mode in ("wave", "all"):
        p = _plot_waveform(events, channels, run_number, out_dir)
        saved_files.append(str(p))
        print(f"   ✅ Waveform saved → {p.name}")

    # ── PeakADC / IntADC ──
    if mode in ("peakADC", "intADC", "all"):
        peak_vals: Dict[int, List[float]] = {ch: [] for ch in channels}
        int_vals:  Dict[int, List[float]] = {ch: [] for ch in channels}

        for ev in events:
            for ch in channels:
                wf = ev[ch]
                if mode in ("peakADC", "all"):
                    peak_vals[ch].append(_peak_adc(wf, start_bin, end_bin))
                if mode in ("intADC", "all"):
                    int_vals[ch].append(_int_adc(wf, start_bin, end_bin))

        if mode in ("peakADC", "all"):
            p = _plot_peakADC(peak_vals, channels, run_number, out_dir, start_bin, end_bin)
            saved_files.append(str(p))
            print(f"   ✅ PeakADC saved → {p.name}")

        if mode in ("intADC", "all"):
            p = _plot_intADC(int_vals, channels, run_number, out_dir, start_bin, end_bin)
            saved_files.append(str(p))
            print(f"   ✅ IntADC  saved → {p.name}")

    return {
        "status": "success",
        "run_number": run_number,
        "mid": mid,
        "events_processed": len(events),
        "channels": channels,
        "start_bin": start_bin,
        "end_bin": end_bin,
        "saved_files": saved_files,
        "output_dir": str(out_dir),
    }


# ─────────────────────────────────────────────
# Tool 클래스 (agent 통합용)
# ─────────────────────────────────────────────
if _HAS_BASE:
    class DATPlotTool(BaseTool):
        """DAT 파일 직접 파싱 플롯 툴 (PyROOT, ROOT 필요)."""

        def __init__(self):
            super().__init__(
                name="dat_plot_tool",
                description="Read .dat file directly and plot Waveform/PeakADC/IntADC with PyROOT"
            )

        def execute(self, params: Dict[str, Any]) -> str:
            run_number = params.get("run_number")
            if run_number is None:
                return "❌ run_number 필수"

            result = run_dat_plot(
                run_number=int(run_number),
                mid=int(params.get("mid", 8)),
                channels=params.get("channels", [0, 1, 2, 3]),
                start_bin=int(params.get("start_bin", 150)),
                end_bin=int(params.get("end_bin", 350)),
                max_events=int(params.get("max_events", -1)),
                mode=params.get("mode", "all"),
                data_dir=params.get("data_dir"),
                output_dir=params.get("output_dir"),
            )

            if result["status"] == "error":
                return f"❌ {result['message']}"

            lines = [
                f"✅ DATPlot 완료  Run {result['run_number']}  MID {result['mid']}",
                f"   이벤트: {result['events_processed']}개",
                f"   채널: {result['channels']}",
                f"   구간: bins {result['start_bin']}–{result['end_bin']}",
                f"   저장 위치: {result['output_dir']}",
            ]
            for f in result["saved_files"]:
                lines.append(f"   📊 {Path(f).name}")
            return "\n".join(lines)


# ─────────────────────────────────────────────
# 커맨드라인 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DAT 파일 플롯 (Waveform / PeakADC / IntADC)")
    parser.add_argument("run_number",            type=int,            help="Run 번호")
    parser.add_argument("--mid",                 type=int, default=8, help="MID 번호 (default 8)")
    parser.add_argument("--channels", "-c",      type=int, nargs="+", default=[0],
                        help="채널 인덱스 0-31 (default: 0)")
    parser.add_argument("--start-bin",           type=int, default=150)
    parser.add_argument("--end-bin",             type=int, default=350)
    parser.add_argument("--max-events", "-n",    type=int, default=-1,
                        help="읽을 최대 이벤트 수 (-1이면 전체, default -1)")
    parser.add_argument("--mode", "-m",          choices=["wave","peakADC","intADC","all"],
                        default="all")
    parser.add_argument("--data-dir",            type=str, default=None)
    parser.add_argument("--output-dir",          type=str, default=None)
    args = parser.parse_args()

    result = run_dat_plot(
        run_number=args.run_number,
        mid=args.mid,
        channels=args.channels,
        start_bin=args.start_bin,
        end_bin=args.end_bin,
        max_events=args.max_events,
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    if result["status"] == "error":
        print(f"❌ {result['message']}")
    else:
        print(f"\n✅ 완료! 저장 위치: {result['output_dir']}")
        for f in result["saved_files"]:
            print(f"   → {f}")
