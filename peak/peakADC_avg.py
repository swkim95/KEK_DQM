import os
import argparse
import numpy as np
import pandas as pd

# Data paths
DATA_DIR = "/Volumes/SSD_8TB"
MAPPING_PATH = "/Users/yhep/DRC/KEK/mapping/mapping_KEK_anomaly.csv"

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

def collect_peakADC_for_run(run_num, cs_type, center):
    """Collect peakADC values for a specific run and CS type using explicit center info"""
    import glob
    
    if not os.path.exists(MAPPING_PATH):
        return []
    
    if not center or not str(center).strip():
        return []
    
    mapping_df = pd.read_csv(MAPPING_PATH)
    
    peakADC_values = []
    
    center_key = str(center).strip().upper()
    cs_key = str(cs_type).strip().upper()
    sub_center_key = f"{center_key}-{cs_key}"
    
    mapping_rows = mapping_df[
        mapping_df['pmt'].astype(str).str.strip().str.upper() == sub_center_key
    ]
    
    if mapping_rows.empty:
        return []
    
    for _, mrow in mapping_rows.iterrows():
        if pd.isna(mrow['mid']) or pd.isna(mrow['ch']):
            continue
        mid = int(mrow['mid'])
        ch = int(mrow['ch']) - 1  # Convert to 0-based indexing
        if not (0 <= ch < 32):
            continue
        
        # Construct file path pattern (same as train.py)
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

def calculate_valley_cut_average(run_num, cs_type, center):
    """Calculate average of peakADC values after valley cut"""
    
    # Collect peakADC data
    peakADC_data = collect_peakADC_for_run(run_num, cs_type, center)
    
    if len(peakADC_data) == 0:
        return None, 0
    
    # Apply valley cut (same parameters as peakADC.py)
    filtered_data = smart_valley_cut(peakADC_data, min_valley_height_ratio=0.3, cut_at_zero=True)
    
    if len(filtered_data) == 0:
        return None, 0
    
    # Calculate average
    average = np.mean(filtered_data)
    count = len(filtered_data)
    
    return average, count

def analyze_run(run_num, center):
    """Analyze a specific run with valley cut"""
    
    # Calculate averages for C type
    avg_C, count_C = calculate_valley_cut_average(run_num, 'C', center)
    
    # Calculate averages for S type
    avg_S, count_S = calculate_valley_cut_average(run_num, 'S', center)
    
    # Simple output
    if avg_C is not None:
        print(f"C: {avg_C:.2f}")
    else:
        print("C: No data")
    
    if avg_S is not None:
        print(f"S: {avg_S:.2f}")
    else:
        print("S: No data")
    
    return avg_C, count_C, avg_S, count_S

def main():
    parser = argparse.ArgumentParser(description="Calculate valley-cut peakADC averages for C and S types")
    parser.add_argument("--run", type=int, required=True, help="Run number to analyze")
    parser.add_argument("--center", type=str, required=True, help="Center identifier (e.g., T1)")
    
    args = parser.parse_args()
    
    analyze_run(args.run, args.center)

if __name__ == "__main__":
    main()
