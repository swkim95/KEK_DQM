#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import json
import time

# Suppress warnings
os.environ['PYTORCH_NNPACK_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Configuration
BOTTLENECK_DIM = 16
DROPOUT = 0.15
DEPTH = 4
LAMBDA_DERIV = 0.15
PEAK_TOPK = 0.1
PEAK_HIGH = 4.0

# Data Paths
DATA_DIR = "/Volumes/SSD_8TB"
MAPPING_PATH = "/Users/yhep/DRC/KEK/mapping/mapping_KEK_anomaly.csv"
MODEL_DIR = "/Users/yhep/DRC/KEK/anomaly"

# Thresholds
THRESHOLD_C = 4.85
THRESHOLD_S = 5.42

# ============================================================================
# Data Processing Functions (simplified from detect_run.py)
# ============================================================================

def count_events_in_file(path):
    filesize = os.path.getsize(path)
    event_size = 64 + 32736*2
    return filesize // event_size

def load_dat_event(path, target_ch, event_idx):
    event_size = 64 + 32736*2
    with open(path, "rb") as f:
        f.seek(event_idx * event_size)
        header = f.read(64)
        data = f.read(32736*2)
    adc = np.frombuffer(data, dtype="<i2")
    reshaped = adc.reshape((1023, 32))
    wf = reshaped[:, target_ch][1:1000]
    return wf

def preprocess_waveform(wf, fixed_window=150):
    baseline = wf[:50].mean()
    wf_clean = wf.astype(float) - baseline
    
    # Simple peak finding
    min_idx = np.argmin(wf_clean)
    
    # Extract window around peak
    start_idx = max(0, min_idx - 50)
    end_idx = min(len(wf_clean), start_idx + fixed_window)
    
    signal_region = wf_clean[start_idx:end_idx]
    
    if len(signal_region) != fixed_window:
        if len(signal_region) > fixed_window:
            signal_region = signal_region[:fixed_window]
        else:
            signal_region = np.pad(signal_region, (0, fixed_window - len(signal_region)), 
                                 mode='constant', constant_values=0)
    
    # Simple normalization
    scale = np.percentile(np.abs(signal_region), 95)
    if scale > 0:
        signal_region = signal_region / scale
    
    return signal_region

# ============================================================================
# Model Definition (simplified)
# ============================================================================

class Conv1DAE_PeakPreserving(nn.Module):
    def __init__(self, seq_len=150, bottleneck_dim=16, dropout=0.15, depth=4):
        super().__init__()
        
        base_channels = 32
        max_channels = base_channels * (2 ** (depth - 1))
        
        # Encoder
        encoder_layers = []
        encoder_layers.extend([
            nn.Conv1d(1, base_channels, 9, stride=1, padding=4),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        
        current_channels = base_channels
        for i in range(depth - 1):
            next_channels = current_channels * 2
            if i < depth - 2:
                encoder_layers.extend([
                    nn.Conv1d(current_channels, next_channels, 9, stride=2, padding=4),
                    nn.BatchNorm1d(next_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            else:
                encoder_layers.extend([
                    nn.Conv1d(current_channels, next_channels, 5, stride=2, padding=2),
                    nn.BatchNorm1d(next_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            current_channels = next_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(max_channels, bottleneck_dim, 1),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )
        
        # Decoder
        decoder_layers = [
            nn.Conv1d(bottleneck_dim, max_channels, 1),
            nn.BatchNorm1d(max_channels),
            nn.ReLU(),
        ]
        
        current_channels = max_channels
        for i in range(depth - 1):
            next_channels = current_channels // 2
            decoder_layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(current_channels, next_channels, 9, padding=4),
                nn.BatchNorm1d(next_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_channels = next_channels
        
        decoder_layers.append(nn.Conv1d(current_channels, 2, 9, padding=4))
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        
        if decoded.shape[-1] != x.shape[-1]:
            decoded = F.interpolate(decoded, size=x.shape[-1], mode='linear', align_corners=False)
        
        mu = decoded[:, :1, :]
        log_scale = decoded[:, 1:, :]
        
        return mu, log_scale, bottleneck.mean(dim=2)

# ============================================================================
# Loss Functions
# ============================================================================

def peak_weighted_nll_loss(x, mu, log_scale, topk=0.1, w_high=3.0, w_low=1.0):
    with torch.no_grad():
        mag = x.abs()
        B, C, T = mag.shape
        k = max(1, int(T * topk))
        
        flat_mag = mag.view(B, -1)
        top_values, _ = torch.topk(flat_mag, k, dim=1, largest=True)
        thresh = top_values[:, -1:].view(B, 1, 1)
        
        weight_mask = torch.where(mag >= thresh, 
                                torch.full_like(mag, w_high), 
                                torch.full_like(mag, w_low))
    
    scale = torch.clamp(F.softplus(log_scale), min=1e-3, max=1.0)
    nll = torch.abs(x - mu) / scale + torch.log(scale)
    weighted_nll = weight_mask * nll
    
    offset = 10.0
    return weighted_nll.mean() + offset

def derivative_loss(x, y, loss_type='l1'):
    dx = x[..., 1:] - x[..., :-1]
    dy = y[..., 1:] - y[..., :-1]
    
    if loss_type == 'l1':
        return F.l1_loss(dy, dx)
    else:
        return F.mse_loss(dy, dx)

# ============================================================================
# Waveform Plotting
# ============================================================================

def plot_waveforms(events, cs_type, run_num, output_dir, is_anomaly=True, max_plots=100):
    """
    Waveform들을 PNG로 저장
    
    Args:
        events: list of (wf_raw, wf_processed, score, tower_name, event_idx, ch)
        cs_type: "C" or "S"
        run_num: Run number
        output_dir: 출력 디렉토리
        is_anomaly: True면 anomalous, False면 normal
        max_plots: 최대 그릴 waveform 개수
    
    Returns:
        PNG 파일 경로
    """
    if len(events) == 0:
        return None
    
    # 최대 개수 제한
    n_plots = min(len(events), max_plots)
    events = events[:n_plots]
    
    # Score 순으로 정렬
    if is_anomaly:
        # Anomalous: 높은 score 순서대로
        events.sort(key=lambda x: x[2], reverse=True)
        label = "anomaly"
    else:
        # Normal: 낮은 score 순서대로
        events.sort(key=lambda x: x[2], reverse=False)
        label = "normal"
    
    # 출력 파일 경로
    output_file = os.path.join(output_dir, f'Run_{run_num}_{label}_{cs_type}.png')
    
    # Grid 크기 계산 (최대 10x10)
    n_cols = min(5, n_plots)
    n_rows = min(20, (n_plots + n_cols - 1) // n_cols)
    
    # Figure 생성
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot
    plot_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if plot_idx >= n_plots:
                axes[row, col].axis('off')
                continue
            
            wf_raw, wf_processed, score, tower_name, event_idx, ch = events[plot_idx]
            
            ax = axes[row, col]
            
            # Raw waveform 그리기
            color = 'red' if is_anomaly else 'blue'
            ax.plot(wf_raw, color=color, alpha=0.7, linewidth=1, label='Raw')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            
            title_prefix = '⚠️' if is_anomaly else '✓'
            ax.set_title(f'{title_prefix} {tower_name} Evt{event_idx}\nScore: {score:.2f}', fontsize=9)
            ax.set_xlabel('Time Sample', fontsize=8)
            ax.set_ylabel('ADC', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    event_type = "anomalous" if is_anomaly else "normal"
    print(f"✅ Saved {n_plots} {event_type} waveforms to {output_file}")
    return output_file

# ============================================================================
# Quick Detection
# ============================================================================

def quick_detect(run_num, tower_list, output_path, mapping_path, model_dir):
    """
    빠른 anomaly detection - 첫 파일(0.dat)의 모든 이벤트 체크
    Anomalous한 경우에만 waveform PNG 생성
    """
    import pandas as pd
    
    # 재현 가능성을 위한 난수 시드 고정
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cpu")  # CPU for fast startup
    
    # Load mapping
    if not os.path.exists(mapping_path):
        return {"status": "error", "message": "Mapping file not found"}
    
    mapping_df = pd.read_csv(mapping_path)
    
    # Load models
    models = {}
    for cs_type in ["C", "S"]:
        model_path = os.path.join(model_dir, f"anomaly_{cs_type}.pth")
        if not os.path.exists(model_path):
            return {"status": "error", "message": f"Model not found: {model_path}"}
        
        model = Conv1DAE_PeakPreserving(seq_len=150, bottleneck_dim=BOTTLENECK_DIM, dropout=DROPOUT, depth=DEPTH).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[cs_type] = model
    
    results = {"C": {"total": 0, "anomalies": 0}, "S": {"total": 0, "anomalies": 0}}
    
    # Events 수집용 (anomalous와 normal 둘 다)
    anomalous_events_C = []
    anomalous_events_S = []
    normal_events_C = []
    normal_events_S = []
    
    for cs_type in ["C", "S"]:
        model = models[cs_type]
        threshold = THRESHOLD_C if cs_type == 'C' else THRESHOLD_S
        anomalous_events = []
        normal_events = []
        
        for tower_name in tower_list:
            # Get channel info
            full_tower_name = f"{tower_name}-{cs_type}"
            matching_rows = mapping_df[mapping_df['pmt'].astype(str).str.strip() == full_tower_name.strip()]
            
            if len(matching_rows) == 0:
                continue
            
            row = matching_rows.iloc[0]
            mid = int(row['mid'])
            ch = int(row['ch']) - 1
            
            # Find first file (FILE_0.dat) only
            pattern = os.path.join(DATA_DIR, 
                                  f"Run_{run_num}/Run_{run_num}_Wave/Run_{run_num}_Wave_MID_{mid}/Run_{run_num}_Wave_MID_{mid}_FILE_0.dat")
            target_files = glob.glob(pattern)
            
            if not target_files:
                continue
            
            target_file = target_files[0]
            
            # Get total number of events in 0.dat file
            try:
                n_events = count_events_in_file(target_file)
            except Exception as e:
                print(f"Error counting events in {target_file}: {e}")
                continue
            
            # Process all events in 0.dat
            n_anomalies = 0
            
            with torch.no_grad():
                for event_idx in range(n_events):
                    try:
                        wf_raw = load_dat_event(target_file, ch, event_idx)
                        processed_wf = preprocess_waveform(wf_raw)
                        x = torch.from_numpy(processed_wf).float().unsqueeze(0).unsqueeze(0).to(device)
                        
                        mu, log_scale, _ = model(x)
                        
                        nll = peak_weighted_nll_loss(x, mu, log_scale, topk=PEAK_TOPK, w_high=PEAK_HIGH, w_low=1.0)
                        deriv = derivative_loss(x, mu, loss_type='l1')
                        score = nll + LAMBDA_DERIV * deriv
                        
                        # Baseline-subtracted waveform
                        baseline = wf_raw[:50].mean()
                        wf_clean = wf_raw.astype(float) - baseline
                        event_data = (
                            wf_clean,  # baseline-subtracted raw waveform
                            processed_wf,  # preprocessed waveform
                            score.item(),  # anomaly score
                            tower_name,  # tower name (e.g., "T1")
                            event_idx,  # event index
                            ch  # channel
                        )
                        
                        if score.item() > threshold:
                            n_anomalies += 1
                            # Anomalous event 저장 (최대 100개까지만)
                            if len(anomalous_events) < 100:
                                anomalous_events.append(event_data)
                        else:
                            # Normal event 저장 (최대 100개까지만)
                            if len(normal_events) < 100:
                                normal_events.append(event_data)
                    except Exception as e:
                        # Skip problematic events
                        continue
            
            results[cs_type]["total"] += n_events
            results[cs_type]["anomalies"] += n_anomalies
        
        # Store events by type
        if cs_type == "C":
            anomalous_events_C = anomalous_events
            normal_events_C = normal_events
        else:
            anomalous_events_S = anomalous_events
            normal_events_S = normal_events
    
    # Save result
    total_events = results["C"]["total"] + results["S"]["total"]
    total_anomalies = results["C"]["anomalies"] + results["S"]["anomalies"]
    
    result = {
        "status": "success",
        "run": run_num,
        "towers": tower_list,  # Tower 정보 저장
        "total_events": total_events,
        "anomalies": total_anomalies,
        "anomaly_rate": (total_anomalies / total_events * 100) if total_events > 0 else 0,
        "details": results,
        "is_anomaly": total_anomalies > 0
    }
    
    # PNG 생성 (Anomalous 또는 Normal) - 비율 기준으로 판단
    output_dir = os.path.dirname(output_path)
    png_files = []
    
    # 임계값: 10% 이상이 anomaly면 anomalous run으로 간주
    ANOMALY_THRESHOLD = 0.10
    
    # C-type 처리
    if results["C"]["total"] > 0:
        c_anomaly_rate = results["C"]["anomalies"] / results["C"]["total"]
        
        if c_anomaly_rate >= ANOMALY_THRESHOLD and len(anomalous_events_C) > 0:
            # Anomalous run - anomalous 이벤트 표시
            png_file = plot_waveforms(anomalous_events_C, "C", run_num, output_dir, is_anomaly=True, max_plots=100)
            if png_file:
                png_files.append(os.path.basename(png_file))
                print(f"✅ C-type anomalies: {results['C']['anomalies']}/{results['C']['total']} events ({c_anomaly_rate*100:.1f}%)")
        elif len(normal_events_C) > 0:
            # Normal run - normal 이벤트 표시
            png_file = plot_waveforms(normal_events_C, "C", run_num, output_dir, is_anomaly=False, max_plots=100)
            if png_file:
                png_files.append(os.path.basename(png_file))
                normal_count = results['C']['total'] - results['C']['anomalies']
                print(f"✅ C-type normal: {normal_count}/{results['C']['total']} events ({(1-c_anomaly_rate)*100:.1f}%)")
    
    # S-type 처리
    if results["S"]["total"] > 0:
        s_anomaly_rate = results["S"]["anomalies"] / results["S"]["total"]
        
        if s_anomaly_rate >= ANOMALY_THRESHOLD and len(anomalous_events_S) > 0:
            # Anomalous run - anomalous 이벤트 표시
            png_file = plot_waveforms(anomalous_events_S, "S", run_num, output_dir, is_anomaly=True, max_plots=100)
            if png_file:
                png_files.append(os.path.basename(png_file))
                print(f"✅ S-type anomalies: {results['S']['anomalies']}/{results['S']['total']} events ({s_anomaly_rate*100:.1f}%)")
        elif len(normal_events_S) > 0:
            # Normal run - normal 이벤트 표시
            png_file = plot_waveforms(normal_events_S, "S", run_num, output_dir, is_anomaly=False, max_plots=100)
            if png_file:
                png_files.append(os.path.basename(png_file))
                normal_count = results['S']['total'] - results['S']['anomalies']
                print(f"✅ S-type normal: {normal_count}/{results['S']['total']} events ({(1-s_anomaly_rate)*100:.1f}%)")
    
    # Add PNG files to result
    result["png_files"] = png_files
    
    # Save result to JSON (needed for web API)
    try:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save JSON result: {e}")
    
    return result

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick live anomaly detection")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--towers", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mapping", type=str, default="../mapping/mapping_TB2025_v1.csv")
    parser.add_argument("--models", type=str, default="/u/user/jiy1221/anomaly")
    
    args = parser.parse_args()
    
    tower_list = [t.strip() for t in args.towers.split(',')]
    
    result = quick_detect(
        run_num=args.run,
        tower_list=tower_list,
        output_path=args.output,
        mapping_path=args.mapping,
        model_dir=args.models
    )
    
    if result["status"] == "success":
        print(f"✅ Quick detection completed")
        print(f"   Total events: {result['total_events']}")
        print(f"   Anomalies: {result['anomalies']}")
        print(f"   Rate: {result['anomaly_rate']:.2f}%")
        if result['is_anomaly']:
            print(f"⚠️  ANOMALY DETECTED!")
            if 'png_files' in result and result['png_files']:
                print(f"   PNG files generated: {', '.join(result['png_files'])}")
    else:
        print(f"❌ Detection failed: {result.get('message', 'Unknown error')}")

