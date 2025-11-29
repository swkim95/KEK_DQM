import os
import sys
import argparse
import glob
import struct
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

import ROOT
ROOT.gROOT.SetBatch(True)

# Data paths
DATA_DIR = "/Volumes/yhep/scratch/YUdaq"
CSV_PATH = "/Users/yhep/DRC/KEK/anomaly/trainset.csv"
MAPPING_PATH = "/Users/yhep/DRC/KEK/mapping/mapping_KEK_anomaly.csv"
OUT_DIR = "/Users/yhep/DRC/KEK/peak/hist"
CS_LIST = ['C', 'S']

def count_events_in_file(path):
    """Count total events in a .dat file (same as train.py)"""
    filesize = os.path.getsize(path)
    event_size = 64 + 32736*2
    return filesize // event_size

def load_dat_event(path, target_ch, event_idx):
    """Load a single event from .dat file (same as train.py)"""
    event_size = 64 + 32736*2
    with open(path, "rb") as f:
        f.seek(event_idx * event_size)
        header = f.read(64)
        data = f.read(32736*2)
    adc = np.frombuffer(data, dtype="<i2")
    reshaped = adc.reshape((1023, 32))
    wf = reshaped[:, target_ch][1:1000]
    return wf

def compute_peakADC_100bin(wf):
    """Compute peakADC using first 100 bins for baseline"""
    if len(wf) < 100:
        return None
    
    baseline = np.mean(wf[:100])
    peakADC = np.max(baseline - wf)
    return float(peakADC)

def collect_peakADC_for_run(run_num, cs_type):
    """Collect peakADC values for a specific run and CS type (following train.py structure)"""
    
    if not os.path.exists(CSV_PATH) or not os.path.exists(MAPPING_PATH):
        return []
    
    trainset_df = pd.read_csv(CSV_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    
    peakADC_values = []
    
    # Find rows for this specific run
    run_rows = trainset_df[trainset_df['Run Number'] == run_num]
    
    for _, row in run_rows.iterrows():
        center = str(row['Center']).strip()
        sub_center = (center + f"-{cs_type}").strip()
        
        # Find mapping entries for this sub_center
        mapping_rows = mapping_df[mapping_df['pmt'].astype(str).str.strip() == sub_center.strip()]
        
        for _, mrow in mapping_rows.iterrows():
            if pd.isna(mrow['mid']) or pd.isna(mrow['ch']):
                continue
            mid = int(mrow['mid'])
            ch = int(mrow['ch']) - 1  # Convert to 0-based indexing
            if not (0 <= ch < 32):
                continue
            
            # Use files 0, 1, 2 (same as train.py)
            pattern = os.path.join(DATA_DIR, f"Run_{run_num}/Run_{run_num}_Wave/Run_{run_num}_Wave_MID_{mid}/Run_{run_num}_Wave_MID_{mid}_FILE_*.dat")
            target_files = glob.glob(pattern)
            
            if not target_files:
                continue
            
            # Process all files for this MID
            for target_file in target_files:
                n_events = count_events_in_file(target_file)
                
                if n_events == 0:
                    continue
                
                for event_idx in range(n_events):
                    wf = load_dat_event(target_file, ch, event_idx)
                    if wf is not None:
                        peakADC = compute_peakADC_100bin(wf)
                        if peakADC is not None:
                            peakADC_values.append(peakADC)
    
    return peakADC_values

def smart_valley_cut(values, min_valley_height_ratio=0.1, cut_at_zero=True, bins=200):
    """
    Smart VERTICAL cut to remove noise peaks while preserving real signals
    
    Logic:
    - Deep valley (< threshold) → Noise peak → Cut at valley position (left cut)
    - Shallow valley (≥ threshold) → Double peak (real signals) → Don't cut
    - Cut at first zero count bin from right (right cut)
    
    Purpose: Remove noise, preserve real double peaks
    """
    if len(values) == 0:
        return values
    
    values = np.array(values)
    
    # Create histogram for analysis
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth histogram for peak detection (simple moving average fallback)
    try:
        from scipy.ndimage import gaussian_filter1d
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
    except ImportError:
        # Simple moving average fallback
        window = 5
        hist_smooth = np.convolve(hist.astype(float), np.ones(window)/window, mode='same')
    
    # Find peaks (local maxima with significant height)
    peaks = []
    min_peak_height = 0.05 * np.max(hist_smooth)  # Peaks must be > 5% of max height
    
    for i in range(1, len(hist_smooth) - 1):
        if (hist_smooth[i] > hist_smooth[i-1] and 
            hist_smooth[i] > hist_smooth[i+1] and 
            hist_smooth[i] > min_peak_height):
            peaks.append(i)
    
    if len(peaks) < 2:
        filtered_values = values
    else:
        # Find main peak (rightmost significant peak)
        main_peak_idx = peaks[-1]  # Rightmost peak
        main_peak_height = hist_smooth[main_peak_idx]
        
        # Look for valleys to the left of main peak (start from closest to main peak)
        best_cut_position = None
        
        # Reverse order: check peaks closer to main peak first
        for peak_idx in reversed(peaks[:-1]):  # Start from peak closest to main peak
            # Find valley between this peak and the main peak
            start_idx = peak_idx
            end_idx = main_peak_idx
            
            if start_idx < end_idx:
                valley_region = hist_smooth[start_idx:end_idx]
                valley_idx_relative = np.argmin(valley_region)
                valley_idx_absolute = start_idx + valley_idx_relative
                valley_height = hist_smooth[valley_idx_absolute]
                
                # Calculate valley height ratio
                valley_height_ratio = valley_height / main_peak_height
                
                # DEEP valley indicates noise peak -> CUT
                # SHALLOW valley indicates real double peak -> DON'T CUT
                if valley_height_ratio < min_valley_height_ratio:
                    best_cut_position = bin_centers[valley_idx_absolute]
                    break  # Take the first deep valley found (closest to main peak)
        
        if best_cut_position is None:
            filtered_values = values
        else:
            # Apply VERTICAL cut at the valley position (left cut)
            filtered_values = values[values >= best_cut_position]  # Keep values >= valley position
    
    # Apply zero-count cut if requested (right cut from main peak)
    if cut_at_zero and len(filtered_values) > 0:
        # Create histogram to find zero count bins
        hist_for_zero, bin_edges_for_zero = np.histogram(filtered_values, bins=bins)
        bin_centers_for_zero = (bin_edges_for_zero[:-1] + bin_edges_for_zero[1:]) / 2
        
        # Find main peak position in the histogram
        main_peak_bin_idx = np.argmax(hist_for_zero)
        
        # From main peak, go right and find first zero count bin
        zero_cut_position = None
        for i in range(main_peak_bin_idx + 1, len(hist_for_zero)):
            if hist_for_zero[i] == 0:
                zero_cut_position = bin_edges_for_zero[i]  # Left edge of first zero bin
                break
        
        if zero_cut_position is not None:
            filtered_values = filtered_values[filtered_values < zero_cut_position]
    
    return filtered_values

def hartigans_dip_test(values):
    """Hartigans' Dip Test for unimodality vs multimodality using diptest package"""
    import diptest
    
    if len(values) < 3:
        return np.nan, np.nan
    
    # Convert to numpy array for diptest
    values_array = np.array(values)
    dip_stat, p_value = diptest.diptest(values_array)
    return dip_stat, p_value

def calculate_anomaly_metrics(test_data, reference_data, cs_type):
    """Calculate anomaly detection metrics using scipy functions"""
    if len(test_data) == 0 or len(reference_data) == 0:
        return None
    
    metrics = {}
    
    # 1. Wasserstein Distance (scipy.stats)
    wasserstein_dist = stats.wasserstein_distance(test_data, reference_data)
    metrics['wasserstein_distance'] = wasserstein_dist
    
    # 2. Hartigans' Dip Test
    dip_stat, dip_p_value = hartigans_dip_test(test_data)
    metrics['dip_statistic'] = dip_stat
    metrics['dip_p_value'] = dip_p_value
    
    # 3. Multimodality Test (scipy-based)
    multimodal_score, cvm_p_value = multimodality_test_scipy(test_data)
    metrics['multimodal_score'] = multimodal_score
    metrics['cvm_p_value'] = cvm_p_value
    
    # 4. FWHM Ratio (scipy-enhanced)
    test_fwhm = calculate_fwhm_scipy(test_data)
    ref_fwhm = calculate_fwhm_scipy(reference_data)
    fwhm_ratio = test_fwhm / ref_fwhm if ref_fwhm > 0 and not np.isnan(ref_fwhm) else np.nan
    metrics['fwhm_ratio'] = fwhm_ratio
    
    # 5. Additional scipy.stats tests
    # Cramer-von Mises test between test and reference
    try:
        cvm_test_stat, cvm_test_p = stats.cramervonmises_2samp(test_data, reference_data)
        metrics['cvm_test_statistic'] = cvm_test_stat
        metrics['cvm_test_p_value'] = cvm_test_p
    except:
        metrics['cvm_test_statistic'] = np.nan
        metrics['cvm_test_p_value'] = np.nan
    
    # Kolmogorov-Smirnov test
    try:
        ks_stat, ks_p = stats.ks_2samp(test_data, reference_data)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_p_value'] = ks_p
    except:
        metrics['ks_statistic'] = np.nan
        metrics['ks_p_value'] = np.nan
    
    return metrics

def plot_single_distribution(run_num, peakADC_data, cs_type, cut="valley"):
    """Create a single histogram plot for one CS type"""
    
    if len(peakADC_data) == 0:
        return
    
    # Process data
    if cut == "none":
        filtered_data = peakADC_data
        title = f"Run {run_num} - {cs_type} Type"
    elif cut == "valley":
        filtered_data = smart_valley_cut(peakADC_data, min_valley_height_ratio=0.3, cut_at_zero=True)
        title = f"Run {run_num} - {cs_type} Type"
    else:
        filtered_data = peakADC_data
        title = f"Run {run_num} - {cs_type} Type"
    
    if len(filtered_data) == 0:
        return
    
    # Create canvas and plot
    canvas = ROOT.TCanvas(f"c_{cs_type}", title, 800, 600)
    
    # Use linear scale for both C and S types
    
    # Determine histogram range
    values = np.array(filtered_data)
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    # Create histogram
    hist = ROOT.TH1F(f"hist_{cs_type}", title, 100, 
                      min_val - 0.05 * range_val, max_val + 0.05 * range_val)
    
    # Set colors
    if cs_type == 'C':
        hist.SetLineColor(ROOT.kBlue)
    else:  # S type
        hist.SetLineColor(ROOT.kRed)
    
    hist.GetXaxis().SetTitle("peakADC")
    hist.GetYaxis().SetTitle("Count")
    
    # Fill histogram
    for value in filtered_data:
        hist.Fill(value)
    
    hist.Draw()
    
    # Save plot
    output_file = f"{OUT_DIR}/peakADC_Run_{run_num}_{cs_type}.root"
    canvas.SaveAs(output_file)
    canvas.Close()

def main():
    parser = argparse.ArgumentParser(description="peakADC Distribution Analysis with Self-Anomaly Detection")
    parser.add_argument("--run", type=int, required=True, help="Run number to analyze")
    parser.add_argument("--cut", type=str, choices=["none", "valley"], default="valley",
                       help="Cut method: none=full range, valley=valley detection (default: valley)")
    
    args = parser.parse_args()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Collecting C type data...")
    peakADC_data_C = collect_peakADC_for_run(args.run, 'C')
    print(f"C data collected: {len(peakADC_data_C)} events")
    
    print("Collecting S type data...")
    peakADC_data_S = collect_peakADC_for_run(args.run, 'S')
    print(f"S data collected: {len(peakADC_data_S)} events")
    
    if len(peakADC_data_C) == 0 and len(peakADC_data_S) == 0:
        return
    
    # Create separate plots for C and S
    plot_single_distribution(
        run_num=args.run,
        peakADC_data=peakADC_data_C,
        cs_type='C',
        cut=args.cut
    )
    
    plot_single_distribution(
        run_num=args.run,
        peakADC_data=peakADC_data_S,
        cs_type='S',
        cut=args.cut
    )
    
    # Apply same filtering to test data
    filtered_test_C = peakADC_data_C
    filtered_test_S = peakADC_data_S
    
    if args.cut == "valley":
        if len(peakADC_data_C) > 0:
            filtered_test_C = smart_valley_cut(peakADC_data_C, min_valley_height_ratio=0.3, cut_at_zero=True)
        if len(peakADC_data_S) > 0:
            filtered_test_S = smart_valley_cut(peakADC_data_S, min_valley_height_ratio=0.3, cut_at_zero=True)
    
    # Basic anomaly detection (self-analysis)
    print(f"\n=== Anomaly Analysis (Self-Detection) ===")
    
    # Analyze C type
    if len(filtered_test_C) > 0:
        print(f"\n--- C Type Analysis ---")
        
        # Hartigans' Dip Test (정석 방법)
        dip_stat, dip_p = hartigans_dip_test(filtered_test_C)
        print(f"Dip Statistic: {dip_stat:.6f}")
        print(f"Dip P-value: {dip_p:.4f} {'⚠️ Multimodal!' if dip_p < 0.05 else '✅ Unimodal'}")
    
    # Analyze S type  
    if len(filtered_test_S) > 0:
        print(f"\n--- S Type Analysis ---")
        
        # Hartigans' Dip Test (정석 방법)
        dip_stat, dip_p = hartigans_dip_test(filtered_test_S)
        print(f"Dip Statistic: {dip_stat:.6f}")
        print(f"Dip P-value: {dip_p:.4f} {'⚠️ Multimodal!' if dip_p < 0.05 else '✅ Unimodal'}")
    
    print(f"\nCompleted: Run {args.run} analysis")

if __name__ == "__main__":
    main()
