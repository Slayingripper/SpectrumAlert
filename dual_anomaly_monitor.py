#!/usr/bin/env python3


import os
import time
import json
import threading
from datetime import datetime
import paho.mqtt.client as mqtt

class DualAnomalyMonitor:
    def __init__(self):
        self.mqtt_anomalies = []
        self.terminal_anomalies = []
        self.mqtt_client = None
        self.connected = False
        
    def setup_mqtt_monitoring(self):
        """Setup MQTT client to monitor anomalies"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_message = self.on_mqtt_message
            
            # Connect to your MQTT broker
            broker = "172.25.96.250"  # From your config
            port = 1883
            
            print(f"🔗 Connecting to MQTT broker: {broker}:{port}")
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            
            return True
        except Exception as e:
            print(f"❌ MQTT setup failed: {e}")
            return False
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            print("✅ Connected to MQTT broker")
            
            # Subscribe to anomaly topic
            topic = "hamradio/anomalies"
            client.subscribe(topic)
            print(f"📡 Subscribed to: {topic}")
        else:
            print(f"❌ MQTT connection failed: {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT anomaly messages"""
        try:
            payload = json.loads(msg.payload.decode())
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Extract key information
            freq_hz = payload.get('metadata', {}).get('exact_frequency_hz', 0)
            freq_mhz = payload.get('frequency_mhz', 0)
            score = payload.get('anomaly_score', 0)
            signal_db = payload.get('metadata', {}).get('signal_strength_db', 0)
            anomaly_num = payload.get('metadata', {}).get('anomaly_number', 0)
            
            mqtt_anomaly = {
                'timestamp': timestamp,
                'source': 'MQTT',
                'frequency_hz': freq_hz,
                'frequency_mhz': freq_mhz,
                'score': score,
                'signal_db': signal_db,
                'anomaly_number': anomaly_num,
                'raw_payload': payload
            }
            
            self.mqtt_anomalies.append(mqtt_anomaly)
            
            # Display MQTT anomaly
            print(f"📡 MQTT: [{timestamp}] ANOMALY #{anomaly_num}: "
                  f"{freq_hz:,} Hz ({freq_mhz:.6f} MHz) | "
                  f"Score: {score:.4f} | "
                  f"Signal: {signal_db:.2f} dB")
            
        except Exception as e:
            print(f"❌ Error processing MQTT message: {e}")
    
    def monitor_terminal_log(self):
        """Monitor the terminal log file for anomalies"""
        today = datetime.now().strftime('%Y%m%d')
        log_file = f"logs/anomaly_frequencies_{today}.log"
        
        print(f"📄 Monitoring terminal log: {log_file}")
        
        last_size = 0
        
        while True:
            try:
                if os.path.exists(log_file):
                    current_size = os.path.getsize(log_file)
                    
                    if current_size > last_size:
                        # New content added
                        with open(log_file, 'r') as f:
                            f.seek(last_size)
                            new_lines = f.read().strip()
                            
                            for line in new_lines.split('\n'):
                                if line.strip() and 'ANOMALY' in line:
                                    self.parse_terminal_anomaly(line)
                        
                        last_size = current_size
                
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error monitoring terminal log: {e}")
                time.sleep(5)
    
    def parse_terminal_anomaly(self, line):
        """Parse terminal anomaly log line"""
        try:
            # Example: [09:15:32.123] ANOMALY #1: 144,250,000 Hz (144.250000 MHz) | Score: 0.8543 | Signal: -45.2 dB
            import re
            
            # Extract timestamp
            timestamp_match = re.search(r'\[(\d+:\d+:\d+\.\d+)\]', line)
            timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
            
            # Extract anomaly number
            anomaly_match = re.search(r'ANOMALY #(\d+)', line)
            anomaly_num = int(anomaly_match.group(1)) if anomaly_match else 0
            
            # Extract frequency in Hz
            freq_hz_match = re.search(r'(\d+,?\d*,?\d*) Hz', line)
            freq_hz = int(freq_hz_match.group(1).replace(',', '')) if freq_hz_match else 0
            
            # Extract frequency in MHz
            freq_mhz_match = re.search(r'\((\d+\.\d+) MHz\)', line)
            freq_mhz = float(freq_mhz_match.group(1)) if freq_mhz_match else 0
            
            # Extract score
            score_match = re.search(r'Score: ([\d\.-]+)', line)
            score = float(score_match.group(1)) if score_match else 0
            
            # Extract signal strength
            signal_match = re.search(r'Signal: ([\d\.-]+) dB', line)
            signal_db = float(signal_match.group(1)) if signal_match else 0
            
            terminal_anomaly = {
                'timestamp': timestamp,
                'source': 'TERMINAL',
                'frequency_hz': freq_hz,
                'frequency_mhz': freq_mhz,
                'score': score,
                'signal_db': signal_db,
                'anomaly_number': anomaly_num,
                'raw_line': line
            }
            
            self.terminal_anomalies.append(terminal_anomaly)
            
            # Display terminal anomaly
            print(f"💻 TERM: [{timestamp}] ANOMALY #{anomaly_num}: "
                  f"{freq_hz:,} Hz ({freq_mhz:.6f} MHz) | "
                  f"Score: {score:.4f} | "
                  f"Signal: {signal_db:.2f} dB")
            
        except Exception as e:
            print(f"❌ Error parsing terminal anomaly: {e}")
    
    def compare_anomalies(self):
        """Compare MQTT vs Terminal anomalies periodically"""
        while True:
            try:
                time.sleep(10)  # Compare every 10 seconds
                
                if len(self.mqtt_anomalies) != len(self.terminal_anomalies):
                    print(f"\n⚠️  MISMATCH: MQTT has {len(self.mqtt_anomalies)} anomalies, "
                          f"Terminal has {len(self.terminal_anomalies)} anomalies")
                
                # Check for frequency mismatches in recent anomalies
                recent_mqtt = self.mqtt_anomalies[-5:] if self.mqtt_anomalies else []
                recent_terminal = self.terminal_anomalies[-5:] if self.terminal_anomalies else []
                
                for mqtt_anom in recent_mqtt:
                    # Find matching terminal anomaly by number
                    matching_terminal = None
                    for term_anom in recent_terminal:
                        if term_anom['anomaly_number'] == mqtt_anom['anomaly_number']:
                            matching_terminal = term_anom
                            break
                    
                    if matching_terminal:
                        # Compare frequencies
                        freq_diff = abs(mqtt_anom['frequency_hz'] - matching_terminal['frequency_hz'])
                        if freq_diff > 1000:  # More than 1kHz difference
                            print(f"\n🔍 FREQUENCY MISMATCH in Anomaly #{mqtt_anom['anomaly_number']}:")
                            print(f"   MQTT: {mqtt_anom['frequency_hz']:,} Hz")
                            print(f"   TERM: {matching_terminal['frequency_hz']:,} Hz")
                            print(f"   Diff: {freq_diff:,} Hz")
                
            except Exception as e:
                print(f"❌ Error comparing anomalies: {e}")
    
    def run(self):
        """Run the dual monitoring system"""
        print("🎯 Dual Anomaly Monitor - MQTT vs Terminal")
        print("=" * 60)
        print("This will show anomalies from both MQTT and terminal logs")
        print("🔍 Look for discrepancies between the two sources\n")
        
        # Setup MQTT monitoring
        if not self.setup_mqtt_monitoring():
            print("❌ Cannot monitor MQTT anomalies")
            return
        
        # Start terminal monitoring in a separate thread
        terminal_thread = threading.Thread(target=self.monitor_terminal_log, daemon=True)
        terminal_thread.start()
        
        # Start comparison thread
        compare_thread = threading.Thread(target=self.compare_anomalies, daemon=True)
        compare_thread.start()
        
        print("🚀 Monitoring started. Anomalies will appear below:")
        print("📡 MQTT = Messages received via MQTT broker")
        print("💻 TERM = Messages found in terminal log files")
        print("-" * 60)
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped")
            print(f"📊 Final counts:")
            print(f"   MQTT anomalies: {len(self.mqtt_anomalies)}")
            print(f"   Terminal anomalies: {len(self.terminal_anomalies)}")
            
            if self.mqtt_client:
                self.mqtt_client.disconnect()

def main():
    monitor = DualAnomalyMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
