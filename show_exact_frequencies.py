#!/usr/bin/env python3
"""
Simple Anomaly Frequency Viewer
Shows the exact frequencies of current anomalies
"""

import os
import re
from datetime import datetime

def show_current_anomalies():
    """Show anomalies detected today with exact frequencies"""
    
    print("🎯 Current Anomaly Frequencies")
    print("=" * 50)
    
    today = datetime.now().strftime('%Y%m%d')
    log_file = f"logs/anomaly_frequencies_{today}.log"
    
    if not os.path.exists(log_file):
        print(f"❌ No anomaly log found for today: {log_file}")
        print("💡 Run the spectrum monitor first to generate anomalies")
        return
    
    print(f"📄 Reading: {log_file}")
    print()
    
    anomalies = []
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if 'ANOMALY' in line:
                try:
                    # Parse the anomaly line
                    anomaly_data = parse_anomaly_line(line)
                    if anomaly_data:
                        anomalies.append(anomaly_data)
                except Exception as e:
                    print(f"❌ Error parsing line {line_num}: {e}")
    
    if not anomalies:
        print("✅ No anomalies detected today")
        return
    
    print(f"🚨 Found {len(anomalies)} anomalies:")
    print("-" * 80)
    print(f"{'#':<3} {'Time':<12} {'Frequency (MHz)':<15} {'Frequency (Hz)':<15} {'Score':<8} {'Signal':<10}")
    print("-" * 80)
    
    for anomaly in anomalies:
        print(f"{anomaly['number']:<3} {anomaly['time']:<12} {anomaly['freq_mhz']:<15.6f} "
              f"{anomaly['freq_hz']:<15,} {anomaly['score']:<8.4f} {anomaly['signal_db']:<10.2f}")
    
    # Frequency analysis
    print(f"\n📊 Frequency Analysis:")
    
    # Group by frequency bands
    bands = {}
    for anomaly in anomalies:
        freq_mhz = anomaly['freq_mhz']
        
        if 144 <= freq_mhz <= 148:
            band = "2m Ham (144-148 MHz)"
        elif 420 <= freq_mhz <= 450:
            band = "70cm Ham (420-450 MHz)"
        elif 162 <= freq_mhz <= 163:
            band = "Weather (162-163 MHz)"
        elif 460 <= freq_mhz <= 470:
            band = "FRS/GMRS (460-470 MHz)"
        else:
            band = f"Other ({int(freq_mhz)} MHz band)"
        
        if band not in bands:
            bands[band] = []
        bands[band].append(anomaly)
    
    for band, band_anomalies in bands.items():
        print(f"   {band}: {len(band_anomalies)} anomalies")
        for anomaly in band_anomalies[:3]:  # Show first 3
            print(f"      {anomaly['freq_mhz']:.6f} MHz (Score: {anomaly['score']:.4f})")
    
    # Find most frequent frequencies
    freq_counts = {}
    for anomaly in anomalies:
        freq_key = f"{anomaly['freq_mhz']:.3f}"
        freq_counts[freq_key] = freq_counts.get(freq_key, 0) + 1
    
    if freq_counts:
        print(f"\n🎯 Most Common Anomaly Frequencies:")
        sorted_freqs = sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)
        for freq, count in sorted_freqs[:5]:
            print(f"   {freq} MHz: {count} times")

def parse_anomaly_line(line):
    """Parse a single anomaly log line"""
    try:
        # Example: [09:15:32.123] ANOMALY #1: 144,250,000 Hz (144.250000 MHz) | Score: 0.8543 | Signal: -45.2 dB
        
        # Extract time
        time_match = re.search(r'\[(\d+:\d+:\d+)', line)
        time = time_match.group(1) if time_match else "unknown"
        
        # Extract anomaly number
        num_match = re.search(r'ANOMALY #(\d+)', line)
        number = int(num_match.group(1)) if num_match else 0
        
        # Extract frequency in MHz (more reliable)
        freq_mhz_match = re.search(r'\((\d+\.\d+) MHz\)', line)
        freq_mhz = float(freq_mhz_match.group(1)) if freq_mhz_match else 0
        
        # Calculate Hz from MHz if Hz parsing fails
        freq_hz_match = re.search(r'(\d+,?\d*,?\d*) Hz', line)
        if freq_hz_match:
            freq_hz = int(freq_hz_match.group(1).replace(',', ''))
        else:
            freq_hz = int(freq_mhz * 1e6)
        
        # Extract score
        score_match = re.search(r'Score: ([-\d\.]+)', line)
        score = float(score_match.group(1)) if score_match else 0
        
        # Extract signal strength
        signal_match = re.search(r'Signal: ([-\d\.]+) dB', line)
        signal_db = float(signal_match.group(1)) if signal_match else 0
        
        return {
            'time': time,
            'number': number,
            'freq_mhz': freq_mhz,
            'freq_hz': freq_hz,
            'score': score,
            'signal_db': signal_db,
            'raw_line': line.strip()
        }
    
    except Exception as e:
        print(f"❌ Error parsing line: {e}")
        return None

def show_live_anomalies():
    """Show anomalies as they happen"""
    import time
    
    print("🔍 Live Anomaly Monitor")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    today = datetime.now().strftime('%Y%m%d')
    log_file = f"logs/anomaly_frequencies_{today}.log"
    
    last_size = 0
    anomaly_count = 0
    
    try:
        while True:
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                
                if current_size > last_size:
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        
                        for line in new_content.strip().split('\n'):
                            if line and 'ANOMALY' in line:
                                anomaly_count += 1
                                anomaly = parse_anomaly_line(line)
                                if anomaly:
                                    print(f"🚨 #{anomaly_count}: {anomaly['freq_mhz']:.6f} MHz "
                                          f"({anomaly['freq_hz']:,} Hz) | "
                                          f"Score: {anomaly['score']:.4f} | "
                                          f"Signal: {anomaly['signal_db']:.2f} dB")
                    
                    last_size = current_size
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped. Total anomalies seen: {anomaly_count}")

def main():
    print("🎯 Anomaly Frequency Tools")
    print("=" * 30)
    print("1. Show today's anomalies")
    print("2. Live anomaly monitor")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == "1":
        show_current_anomalies()
    elif choice == "2":
        show_live_anomalies()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
