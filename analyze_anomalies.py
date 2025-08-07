#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from sklearn.ensemble import IsolationForest
    import joblib
    
    print("🔍 SpectrumAlert Anomaly Analysis")
    print("=" * 50)
    
    # Load your training data
    data_file = "data/automated_20250807_085737_full.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    df = pd.read_csv(data_file)
    print(f"📊 Loaded {len(df)} samples from training data")
    
    # Extract features (drop frequency column)
    features = df.drop('Frequency', axis=1).values
    feature_names = df.drop('Frequency', axis=1).columns.tolist()
    
    print(f"🔬 Features: {len(feature_names)} dimensions")
    print(f"   {', '.join(feature_names[:4])}...")
    
    # Load the trained model
    model_files = list(Path("models").glob("anomaly_detection_*.pkl"))
    if not model_files:
        print("❌ No anomaly detection models found!")
        sys.exit(1)
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading model: {latest_model.name}")
    
    model = joblib.load(latest_model)
    
    # Test different contamination rates
    contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print(f"\n🧪 Testing different contamination rates:")
    print(f"{'Rate':<8} {'Anomalies':<10} {'Percentage':<12} {'Recommendation'}")
    print("-" * 60)
    
    for rate in contamination_rates:
        # Create new model with different contamination rate
        test_model = IsolationForest(contamination=rate, random_state=42)
        test_model.fit(features)
        
        # Predict anomalies
        predictions = test_model.predict(features)
        anomaly_count = sum(1 for p in predictions if p == -1)
        percentage = (anomaly_count / len(predictions)) * 100
        
        if percentage < 5:
            recommendation = "✅ Good"
        elif percentage < 15:
            recommendation = "⚠️  Acceptable"
        elif percentage < 30:
            recommendation = "🔶 High"
        else:
            recommendation = "🚨 Too High"
        
        print(f"{rate:<8.2f} {anomaly_count:<10} {percentage:<12.1f}% {recommendation}")
    
    print(f"\n📈 Current Model Analysis:")
    
    # Analyze current model
    current_predictions = model.predict(features)
    current_scores = model.decision_function(features)
    current_anomalies = sum(1 for p in current_predictions if p == -1)
    current_percentage = (current_anomalies / len(current_predictions)) * 100
    
    print(f"   🎯 Current anomaly rate: {current_percentage:.1f}% ({current_anomalies}/{len(features)})")
    print(f"   📊 Score range: {current_scores.min():.3f} to {current_scores.max():.3f}")
    print(f"   🎚️  Decision threshold: ~{np.percentile(current_scores, 10):.3f}")
    
    # Show which samples are anomalous
    print(f"\n🚨 Anomalous samples from your training data:")
    anomaly_indices = [i for i, p in enumerate(current_predictions) if p == -1]
    
    if anomaly_indices:
        for idx in anomaly_indices[:5]:  # Show first 5
            freq = df.iloc[idx]['Frequency'] / 1e6
            score = current_scores[idx]
            signal = df.iloc[idx]['Mean_Amplitude']
            print(f"   📍 Sample {idx+1}: {freq:.1f} MHz, Score: {score:.3f}, Signal: {signal:.4f}")
    else:
        print("   ✅ No anomalies detected in training data")
    
    print(f"\n💡 Recommendations:")
    if current_percentage > 20:
        print(f"   🔧 PROBLEM: {current_percentage:.1f}% anomaly rate is too high!")
        print(f"   🎯 SOLUTION: Increase contamination rate to 0.2-0.25")
        print(f"   ⚡ QUICK FIX: export SERVICE_CONTAMINATION_RATE=0.2")
    elif current_percentage > 10:
        print(f"   ⚠️  WARNING: {current_percentage:.1f}% is borderline high")
        print(f"   🎯 SUGGESTION: Try contamination rate 0.15")
    else:
        print(f"   ✅ Good: {current_percentage:.1f}% anomaly rate is reasonable")
    
    print(f"\n🚀 To fix anomaly detection:")
    print(f"   1. echo 'export SERVICE_CONTAMINATION_RATE=0.2' >> .env")
    print(f"   2. ./run_docker.sh stop && ./run_docker.sh autonomous")
    print(f"   3. Or collect more diverse training data (30+ minutes)")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install scikit-learn pandas")
except Exception as e:
    print(f"❌ Error: {e}")
