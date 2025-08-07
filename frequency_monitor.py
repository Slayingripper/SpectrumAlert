#!/usr/bin/env python3
"""
Real-time anomaly frequency monitor
Shows exact frequencies of anomalies in real-time
"""

import os
import subprocess
import time
import signal
import sys
from datetime import datetime

def monitor_anomalies():
    """Monitor spectrum and show exact frequencies of anomalies"""
    
    print("🎯 Real-time Anomaly Frequency Monitor")
    print("=" * 50)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    print("🚀 Starting spectrum monitoring...")
    print("📊 When anomalies are detected, exact frequencies will be shown below")
    print("🛑 Press Ctrl+C to stop\n")
    
    # Monitor the anomaly log file
    today = datetime.now().strftime('%Y%m%d')
    log_file = f"logs/anomaly_frequencies_{today}.log"
    
    anomaly_count = 0
    last_size = 0
    
    print(f"📄 Monitoring log file: {log_file}")
    print("🔍 Waiting for anomalies...\n")
    
    try:
        while True:
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                
                if current_size > last_size:
                    # New content added to log file
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_lines = f.read().strip()
                        
                        if new_lines:
                            for line in new_lines.split('\n'):
                                if line.strip():
                                    anomaly_count += 1
                                    print(f"🚨 {line}")
                    
                    last_size = current_size
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped")
        print(f"📊 Total anomalies detected: {anomaly_count}")
        
        if anomaly_count > 0:
            print(f"📄 Full log available in: {log_file}")

def show_instructions():
    """Show instructions for running anomaly detection"""
    print("📋 INSTRUCTIONS:")
    print("1. Start this frequency monitor: python3 frequency_monitor.py")
    print("2. In another terminal, run: python3 main.py")
    print("3. Choose option 3 (Monitor Spectrum)")
    print("4. Select your anomaly detection model")
    print("5. Watch this window for exact anomaly frequencies!")
    print("\n🎯 This will show EXACT frequencies like:")
    print("   🚨 [09:15:32.123] ANOMALY #1: 144,250,000 Hz (144.250000 MHz) | Score: 0.8543")
    print("   🚨 [09:15:34.567] ANOMALY #2: 420,125,000 Hz (420.125000 MHz) | Score: 0.7821")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_instructions()
    else:
        monitor_anomalies()
