#!/usr/bin/env python3

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.model_manager import ModelManager
    from src.core.data_manager import DataManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class AdaptiveAnomalyTuner:
    """Tune anomaly detection parameters based on your specific RF environment"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.data_manager = DataManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_current_models(self):
        """Analyze current model performance and suggest improvements"""
        print("🔍 Analyzing Current Anomaly Detection Models...")
        print("=" * 60)
        
        model_files = self.model_manager.list_models()
        anomaly_models = [f for f in model_files if 'anomaly' in f.lower()]
        
        if not anomaly_models:
            print("❌ No anomaly detection models found!")
            print("   Please train models first using the autonomous mode.")
            return
        
        for model_file in anomaly_models:
            print(f"\n📊 Model: {model_file}")
            metadata = self.model_manager.load_model_metadata(model_file)
            
            if metadata:
                contamination = metadata.get('contamination', 'unknown')
                anomaly_ratio = metadata.get('anomaly_ratio', 'unknown')
                n_samples = metadata.get('n_samples', 'unknown')
                
                print(f"   Current contamination rate: {contamination}")
                print(f"   Detected anomalies in training: {anomaly_ratio:.2%}" if isinstance(anomaly_ratio, float) else f"   Detected anomalies: {anomaly_ratio}")
                print(f"   Training samples: {n_samples}")
                
                # Provide recommendations
                if isinstance(anomaly_ratio, float):
                    if anomaly_ratio > 0.3:
                        print("   ⚠️  ISSUE: Very high anomaly rate in training data!")
                        print("      This suggests your baseline data contains too much variation.")
                        print("      SOLUTION: Retrain with contamination=0.3 or higher")
                    elif anomaly_ratio > 0.15:
                        print("   ⚠️  WARNING: High anomaly rate detected")
                        print("      SOLUTION: Try contamination=0.2-0.25")
                    elif anomaly_ratio < 0.02:
                        print("   ⚠️  WARNING: Very low anomaly rate")
                        print("      This might cause over-sensitivity in monitoring")
                        print("      SOLUTION: Try contamination=0.02-0.05")
                    else:
                        print("   ✅ Reasonable anomaly rate for training data")
                
            else:
                print("   ❌ No metadata available for this model")
    
    def suggest_optimal_contamination(self):
        """Suggest optimal contamination rate based on training data analysis"""
        print("\n🎯 Contamination Rate Recommendations:")
        print("=" * 50)
        
        data_files = self.data_manager.list_data_files()
        if not data_files:
            print("❌ No training data found for analysis")
            return
        
        print("Based on RF environment characteristics:")
        print("• Urban/Dense RF environment: contamination = 0.15-0.25")
        print("• Suburban environment: contamination = 0.08-0.15") 
        print("• Rural/Quiet environment: contamination = 0.03-0.08")
        print("• Research/Lab environment: contamination = 0.02-0.05")
        print()
        print("🔧 Quick Fix Commands:")
        print("1. For immediate relief (if everything is anomaly):")
        print("   Set environment variable: SERVICE_ALERT_THRESHOLD=0.3")
        print()
        print("2. For long-term solution:")
        print("   Retrain models with higher contamination rate")
    
    def create_retrain_script(self):
        """Create a script to retrain models with better parameters"""
        script_content = """#!/usr/bin/env python3
'''
Retrain SpectrumAlert models with adaptive contamination rates
'''

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.model_manager import AnomalyDetectionTrainer, RFFingerprintingTrainer
from src.core.data_manager import DataManager
import numpy as np

def retrain_with_adaptive_contamination():
    data_manager = DataManager()
    
    # Get latest data file
    data_files = data_manager.list_data_files()
    if not data_files:
        print("No data files found!")
        return
    
    latest_file = max(data_files, key=lambda f: os.path.getmtime(
        os.path.join(data_manager.data_dir, f)
    ))
    
    print(f"Using data file: {latest_file}")
    features = data_manager.load_features_csv(latest_file)
    
    if not features:
        print("No features loaded!")
        return
    
    # Calculate adaptive contamination rate
    # Analyze signal strength variation
    signal_strengths = [f.signal_strength for f in features if hasattr(f, 'signal_strength')]
    if signal_strengths:
        ss_std = np.std(signal_strengths)
        ss_range = np.ptp(signal_strengths)  # peak-to-peak
        
        # Adaptive contamination based on signal variation
        if ss_std > 20 or ss_range > 60:  # High variation
            contamination = 0.20
            print(f"High signal variation detected (std={ss_std:.1f}dB). Using contamination=0.20")
        elif ss_std > 10 or ss_range > 30:  # Moderate variation
            contamination = 0.12
            print(f"Moderate signal variation detected (std={ss_std:.1f}dB). Using contamination=0.12")
        else:  # Low variation
            contamination = 0.05
            print(f"Low signal variation detected (std={ss_std:.1f}dB). Using contamination=0.05")
    else:
        contamination = 0.10  # Default fallback
        print("No signal strength data found. Using default contamination=0.10")
    
    # Retrain anomaly detection model
    print(f"\\nRetraining anomaly detection model with contamination={contamination}...")
    trainer = AnomalyDetectionTrainer(lite_mode='lite' in latest_file.lower())
    model, metadata = trainer.train_model(features, contamination=contamination)
    
    if model:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_lite" if 'lite' in latest_file.lower() else "_full"
        model_file = f"adaptive_anomaly_{timestamp}{mode_suffix}.pkl"
        
        trainer.model_manager.save_model(model, model_file, metadata)
        print(f"✅ Adaptive anomaly model saved: {model_file}")
        print(f"   Contamination rate: {contamination}")
        print(f"   Detected anomalies: {metadata.get('anomaly_ratio', 0):.2%}")
    else:
        print("❌ Model training failed!")

if __name__ == "__main__":
    retrain_with_adaptive_contamination()
"""
        
        with open("retrain_adaptive.py", "w") as f:
            f.write(script_content)
        
        os.chmod("retrain_adaptive.py", 0o755)
        print("\n📝 Created retrain_adaptive.py script")
        print("   Run with: python3 retrain_adaptive.py")

def main():
    tuner = AdaptiveAnomalyTuner()
    
    print("🚀 SpectrumAlert Anomaly Detection Tuner")
    print("=" * 50)
    
    tuner.analyze_current_models()
    tuner.suggest_optimal_contamination()
    tuner.create_retrain_script()
    
    print("\n🎯 Quick Actions:")
    print("1. Immediate fix: Increase SERVICE_ALERT_THRESHOLD to 0.3-0.5")
    print("2. Better solution: Run './retrain_adaptive.py' to retrain with optimal settings")
    print("3. Best solution: Collect baseline data in your quiet RF environment first")

if __name__ == "__main__":
    main()
