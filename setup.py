#!/usr/bin/env python3
"""
Setup and migration script for SpectrumAlert v2.0
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8, 0):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models", 
        "logs",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def migrate_old_data():
    """Migrate data from old structure to new structure"""
    print("Checking for old data to migrate...")
    
    # Map old files to new locations
    migrations = [
        ("collected_data_lite.csv", "data/collected_data_lite_migrated.csv"),
        ("collected_iq_data.csv", "data/collected_iq_data_migrated.csv"),
        ("rf_fingerprinting_model_lite.pkl", "models/rf_model_lite_migrated.pkl"),
        ("rf_fingerprinting_model.pkl", "models/rf_model_full_migrated.pkl"),
        ("anomaly_detection_model_lite.pkl", "models/anomaly_model_lite_migrated.pkl"),
        ("anomaly_detection_model.pkl", "models/anomaly_model_full_migrated.pkl"),
        ("Trainer/config.ini", "config/config_old.ini")
    ]
    
    migrated_count = 0
    for old_path, new_path in migrations:
        if os.path.exists(old_path):
            try:
                shutil.move(old_path, new_path)
                print(f"✓ Migrated: {old_path} -> {new_path}")
                migrated_count += 1
            except Exception as e:
                print(f"✗ Failed to migrate {old_path}: {e}")
    
    if migrated_count > 0:
        print(f"✓ Migrated {migrated_count} files")
    else:
        print("No old files found to migrate")

def setup_configuration():
    """Setup initial configuration"""
    config_file = "config/config.ini"
    
    if os.path.exists(config_file):
        print(f"✓ Configuration file already exists: {config_file}")
        return
    
    # Check if there's an old config to migrate
    if os.path.exists("config/config_old.ini"):
        print("Found old configuration file, you may want to review and update it")
    
    # Create default configuration will be handled by the main app
    print(f"Configuration will be created on first run: {config_file}")

def check_sdr_libraries():
    """Check for available SDR libraries"""
    print("Checking SDR library availability...")
    
    libraries = {
        'pyrtlsdr': 'RTL-SDR support',
        'SoapySDR': 'Multi-SDR support (LimeSDR, HackRF, etc.)'
    }
    
    available = []
    for lib, description in libraries.items():
        try:
            __import__(lib)
            print(f"✓ {lib} - {description}")
            available.append(lib)
        except ImportError:
            print(f"✗ {lib} - {description} (not installed)")
    
    if not available:
        print("Warning: No SDR libraries found. Install pyrtlsdr or SoapySDR")
        print("  For RTL-SDR: pip install pyrtlsdr")
        print("  For SoapySDR: Install SoapySDR and Python bindings")
    
    return available

def cleanup_old_structure():
    """Clean up old project structure"""
    old_items = [
        "Trainer",
        "Main.py"
    ]
    
    print("Cleaning up old project structure...")
    for item in old_items:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    # Check if directory is empty after migration
                    if not os.listdir(item):
                        os.rmdir(item)
                        print(f"✓ Removed empty directory: {item}")
                    else:
                        print(f"⚠ Directory not empty, keeping: {item}")
                else:
                    # Move to backup location
                    backup_name = f"{item}.old"
                    if not os.path.exists(backup_name):
                        shutil.move(item, backup_name)
                        print(f"✓ Moved to backup: {item} -> {backup_name}")
            except Exception as e:
                print(f"✗ Error cleaning up {item}: {e}")

def create_launch_script():
    """Create a convenient launch script"""
    script_content = """#!/bin/bash
# SpectrumAlert Launch Script

echo "Starting SpectrumAlert v2.0..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Launch the application
python3 main.py

echo "SpectrumAlert stopped."
"""
    
    with open("run_spectrum_alert.sh", "w") as f:
        f.write(script_content)
    
    # Make script executable on Unix-like systems
    if os.name != 'nt':
        os.chmod("run_spectrum_alert.sh", 0o755)
    
    print("✓ Created launch script: run_spectrum_alert.sh")

def run_tests():
    """Run basic tests to verify installation"""
    print("Running basic tests...")
    
    try:
        # Test imports
        from src.utils.config_manager import ConfigManager
        from src.core.feature_extraction import FeatureExtractor
        from src.core.data_manager import DataManager
        print("✓ Core modules import successfully")
        
        # Test configuration
        config_manager = ConfigManager()
        if not os.path.exists("config/config.ini"):
            config_manager.create_default_config("config/config.ini")
        config_manager.load_config()
        print("✓ Configuration system working")
        
        # Test feature extraction
        import numpy as np
        extractor = FeatureExtractor(lite_mode=True)
        test_iq = np.random.random(100).astype(np.complex64)
        features = extractor.extract_features(test_iq, 144e6)
        print("✓ Feature extraction working")
        
        print("✓ All basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("SpectrumAlert v2.0 Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed - could not install dependencies")
        sys.exit(1)
    
    # Migrate old data
    migrate_old_data()
    
    # Setup configuration
    setup_configuration()
    
    # Check SDR libraries
    sdr_libs = check_sdr_libraries()
    
    # Clean up old structure
    cleanup_old_structure()
    
    # Create launch script
    create_launch_script()
    
    # Run tests
    if run_tests():
        print("\n" + "=" * 40)
        print("✓ SpectrumAlert v2.0 setup completed successfully!")
        print("\nTo start the application:")
        print("  python3 main.py")
        print("  or")
        print("  ./run_spectrum_alert.sh")
        print("\nFor more information, see README.md")
        
        if not sdr_libs:
            print("\n⚠ Warning: No SDR libraries detected")
            print("Install pyrtlsdr or SoapySDR before using the application")
    else:
        print("\n✗ Setup completed with errors")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
