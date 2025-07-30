#!/usr/bin/env python3
"""
SpectrumAlert v2.0 - Refactored RF Spectrum Monitoring Suite
Enhanced with robust error handling and recovery mechanisms
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
try:
    from src.utils.config_manager import ConfigManager
    from src.core.robust_collector import RobustDataCollector
    from src.core.model_manager import RFFingerprintingTrainer, AnomalyDetectionTrainer, ModelManager
    from src.core.data_manager import DataManager
    from src.core.spectrum_monitor import SpectrumMonitor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have run the setup script: python3 setup.py")
    sys.exit(1)

# Global variables for cleanup
collector = None
monitor = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n\nReceived interrupt signal. Cleaning up...")
    
    if collector:
        collector.stop_collection()
    
    if monitor:
        monitor.stop_monitoring()
    
    print("Cleanup completed. Exiting...")
    sys.exit(0)


def show_banner():
    """Display the application banner"""
    banner = """
▗▄▄▖▗▄▄▖  ▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▖ ▗▖ ▗▖▗▖  ▗▖     ▗▄▖ ▗▖   ▗▄▄▄▖▗▄▄▖▗▄▄▄▖
▐▌   ▐▌ ▐▌▐▌   ▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▛▚▞▜▌    ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌ █  
 ▝▀▚▖▐▛▀▘ ▐▛▀▘▐▌     █  ▐▛▀▚▖▐▌ ▐▌▐▌  ▐▌    ▐▛▀▜▌▐▌   ▐▛▀▘▐▛▀▚▖ █  
▗▄▄▞▘▐▌   ▐▙▄▄▖▝▚▄▄▖  █  ▐▌ ▐▌▝▚▄▞▘▐▌  ▐▌    ▐▌ ▐▌▐▙▄▄▖▐▙▄▄▖▐▌ ▐▌ █  

    SpectrumAlert v2.0 - RF Spectrum Monitoring Suite
    Enhanced with robust error handling and crash recovery
    """
    print(banner)


def check_system_status(config, data_manager, model_manager):
    """Check and display system status"""
    print("\n=== System Status ===")
    
    # Check configuration
    try:
        ham_bands = config.ham_bands
        print(f"✓ Configuration loaded: {len(ham_bands)} ham bands configured")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Check data files
    data_files = data_manager.list_data_files()
    print(f"✓ Data files found: {len(data_files)}")
    for file in data_files[:3]:  # Show first 3
        print(f"  - {file}")
    if len(data_files) > 3:
        print(f"  ... and {len(data_files) - 3} more")
    
    # Check model files
    model_files = model_manager.list_models()
    print(f"✓ Model files found: {len(model_files)}")
    for file in model_files:
        print(f"  - {file}")
    
    # Check RTL-SDR availability
    try:
        from src.core.robust_collector import SafeRTLSDR
        test_sdr = SafeRTLSDR()
        if test_sdr.open():
            print("✓ RTL-SDR device accessible")
            test_sdr.close()
        else:
            print("✗ RTL-SDR device not accessible")
            return False
    except Exception as e:
        print(f"✗ RTL-SDR error: {e}")
        return False
    
    return True


def collect_data_menu(config, data_manager):
    """Data collection menu"""
    global collector
    
    print("\n=== Data Collection ===")
    
    # Ask for lite mode
    lite_mode = input("Enable lite mode for resource-constrained devices? (y/n): ").lower() == 'y'
    
    # Ask for duration
    try:
        duration = float(input("Enter collection duration in minutes: "))
        if duration <= 0:
            print("Duration must be positive")
            return
    except ValueError:
        print("Invalid duration")
        return
    
    # Generate filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_lite" if lite_mode else "_full"
    filename = f"data/collected_data_{timestamp}{mode_suffix}.csv"
    
    print(f"Starting data collection...")
    print(f"Mode: {'Lite' if lite_mode else 'Full'}")
    print(f"Duration: {duration} minutes")
    print(f"Output: {filename}")
    print("Press Ctrl+C to stop early")
    
    collector = RobustDataCollector(config)
    
    try:
        success = collector.collect_data(duration, filename, lite_mode)
        if success:
            print(f"\n✓ Data collection completed successfully!")
            print(f"Data saved to: {filename}")
        else:
            print(f"\n✗ Data collection failed")
    except KeyboardInterrupt:
        print(f"\n⚠ Data collection interrupted by user")
    finally:
        collector = None
        collector = None


def train_models_menu(config, data_manager, model_manager):
    """Model training menu"""
    print("=== Model Training ===")
    
    # List available data files
    data_files = data_manager.list_data_files()
    if not data_files:
        print("No data files found. Please collect data first.")
        return
    
    print("Available data files:")
    for i, file in enumerate(data_files, 1):
        print(f"{i}. {file}")
    
    try:
        choice = int(input("Select data file (number): ")) - 1
        if choice < 0 or choice >= len(data_files):
            print("Invalid selection")
            return
        
        selected_file = data_files[choice]
        lite_mode = "lite" in selected_file.lower()
        
        print(f"Training models using: {selected_file}")
        print(f"Mode: {'Lite' if lite_mode else 'Full'}")
        
        # Load data
        features = data_manager.load_features_csv(selected_file)
        if features is None or len(features) == 0:
            print("No features loaded from data file")
            return
        
        print(f"Loaded {len(features)} feature vectors")
        
        # Prepare timestamps and suffixes for both models
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_lite" if lite_mode else "_full"
        
        # Train RF fingerprinting model
        print("Training RF fingerprinting model...")
        rf_trainer = RFFingerprintingTrainer(lite_mode=lite_mode)
        rf_model, rf_metadata = rf_trainer.train_model(features)
        
        if rf_model:
            model_file = f"rf_fingerprinting_{timestamp}{mode_suffix}.pkl"
            rf_trainer.model_manager.save_model(rf_model, model_file, rf_metadata)
            print(f"✓ RF fingerprinting model saved: {model_file}")
        
        # Train anomaly detection model
        print("Training anomaly detection model...")
        anomaly_trainer = AnomalyDetectionTrainer(lite_mode=lite_mode)
        anomaly_model, anomaly_metadata = anomaly_trainer.train_model(features)
        
        if anomaly_model:
            model_file = f"anomaly_detection_{timestamp}{mode_suffix}.pkl"
            anomaly_trainer.model_manager.save_model(anomaly_model, model_file, anomaly_metadata)
            print(f"✓ Anomaly detection model saved: {model_file}")
        
        print("✓ Model training completed!")
        
    except (ValueError, IndexError):
        print("Invalid selection")
    except Exception as e:
        print(f"Training failed: {e}")


def monitor_spectrum_menu(config, data_manager, model_manager):
    """Spectrum monitoring menu"""
    global monitor
    
    print("\n=== Spectrum Monitoring ===")
    
    # Check for models
    model_files = model_manager.list_models()
    anomaly_models = [f for f in model_files if 'anomaly' in f.lower()]
    
    if not anomaly_models:
        print("No anomaly detection models found. Please train a model first.")
        return
    
    print("Available anomaly detection models:")
    for i, model in enumerate(anomaly_models, 1):
        print(f"{i}. {model}")
    
    try:
        choice = int(input("Select model (number): ")) - 1
        if choice < 0 or choice >= len(anomaly_models):
            print("Invalid selection")
            return
        
        selected_anomaly_model = anomaly_models[choice]
        
        # Get lite_mode from model metadata instead of filename
        anomaly_metadata = model_manager.load_model_metadata(selected_anomaly_model)
        if anomaly_metadata and 'lite_mode' in anomaly_metadata:
            lite_mode = anomaly_metadata['lite_mode']
        else:
            # Fallback to filename check if no metadata
            lite_mode = "lite" in selected_anomaly_model.lower()
        
        # Find corresponding RF fingerprinting model
        # Extract timestamp and mode from anomaly model name
        if "test_anomaly" in selected_anomaly_model:
            selected_rf_model = selected_anomaly_model.replace("test_anomaly", "test_rf")
        elif "anomaly_detection" in selected_anomaly_model:
            selected_rf_model = selected_anomaly_model.replace("anomaly_detection", "rf_fingerprinting")
        else:
            # Fallback: look for any RF model with similar timestamp
            rf_models = [f for f in model_files if 'rf' in f.lower() or 'fingerprint' in f.lower()]
            if rf_models:
                selected_rf_model = rf_models[0]  # Use first available RF model
            else:
                print("No RF fingerprinting model found. Cannot start monitoring.")
                return
        
        print(f"Starting spectrum monitoring...")
        print(f"Anomaly Model: {selected_anomaly_model}")
        print(f"RF Model: {selected_rf_model}")
        print(f"Mode: {'Lite' if lite_mode else 'Full'}")
        print("Press Ctrl+C to stop monitoring")
        
        # Load models to verify they exist
        anomaly_model = model_manager.load_model(selected_anomaly_model)
        rf_model = model_manager.load_model(selected_rf_model)
        if not (anomaly_model and rf_model):
            print("Failed to load one or both models")
            return
        
        # Start monitoring with specific model files
        monitor = SpectrumMonitor(config, lite_mode=lite_mode, 
                                rf_model_file=selected_rf_model,
                                anomaly_model_file=selected_anomaly_model)
        monitor.start_monitoring()
        
    except (ValueError, IndexError):
        print("Invalid selection")
    except KeyboardInterrupt:
        print("\n⚠ Monitoring stopped by user")
    except Exception as e:
        print(f"Monitoring failed: {e}")
    finally:
        if monitor:
            monitor.stop_monitoring()
        monitor = None


def automated_workflow_menu(config, data_manager, model_manager):
    """Automated workflow menu"""
    print("\n=== Automated Workflow ===")
    print("This will: collect data → train models → start monitoring")
    
    lite_mode = input("Enable lite mode? (y/n): ").lower() == 'y'
    
    try:
        duration = float(input("Data collection duration (minutes): "))
        if duration <= 0:
            print("Duration must be positive")
            return
    except ValueError:
        print("Invalid duration")
        return
    
    print(f"\nStarting automated workflow (lite_mode={lite_mode})...")
    
    # Step 1: Collect data
    print("\nStep 1: Collecting data...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_lite" if lite_mode else "_full"
    data_file = f"data/automated_{timestamp}{mode_suffix}.csv"
    
    collector = RobustDataCollector(config)
    success = collector.collect_data(duration, data_file, lite_mode)
    
    if not success:
        print("✗ Data collection failed. Aborting workflow.")
        return
    
    print(f"✓ Data collected: {data_file}")
    
    # Step 2: Train models
    print("\nStep 2: Training models...")
    
    # Load data
    features = data_manager.load_features_csv(data_file)
    if features is None or len(features) == 0:
        print("✗ No features loaded. Aborting workflow.")
        return
    
    # Train models
    rf_trainer = RFFingerprintingTrainer(lite_mode=lite_mode)
    rf_model, rf_metadata = rf_trainer.train_model(features)
    
    anomaly_trainer = AnomalyDetectionTrainer(lite_mode=lite_mode)
    anomaly_model, anomaly_metadata = anomaly_trainer.train_model(features)
    
    if not (rf_model and anomaly_model):
        print("✗ Model training failed. Aborting workflow.")
        return
    
    # Save models
    rf_model_file = f"automated_rf_{timestamp}{mode_suffix}.pkl"
    anomaly_model_file = f"automated_anomaly_{timestamp}{mode_suffix}.pkl"
    
    rf_trainer.model_manager.save_model(rf_model, rf_model_file, rf_metadata)
    anomaly_trainer.model_manager.save_model(anomaly_model, anomaly_model_file, anomaly_metadata)
    
    print(f"✓ Models trained and saved")
    
    # Step 3: Start monitoring
    print("\nStep 3: Starting monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        monitor = SpectrumMonitor(config, lite_mode=lite_mode,
                                rf_model_file=rf_model_file,
                                anomaly_model_file=anomaly_model_file)
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⚠ Monitoring stopped by user")
    except Exception as e:
        print(f"Monitoring failed: {e}")
    
    print("\n✓ Automated workflow completed!")


def cleanup_data_menu(data_manager, model_manager):
    """Data cleanup menu"""
    print("\n=== Data Cleanup ===")
    
    data_files = data_manager.list_data_files()
    model_files = model_manager.list_models()
    
    print(f"Found {len(data_files)} data files and {len(model_files)} model files")
    
    if not data_files and not model_files:
        print("No files to clean up")
        return
    
    print("\nCleanup options:")
    print("1. Delete old data files (keep models)")
    print("2. Delete old model files (keep data)")
    print("3. Delete everything")
    print("4. Cancel")
    
    try:
        choice = int(input("Select option: "))
        
        if choice == 1:
            confirm = input(f"Delete {len(data_files)} data files? (yes/no): ")
            if confirm.lower() == 'yes':
                count = 0
                for data_file in data_files:
                    try:
                        data_path = os.path.join(data_manager.data_dir, data_file)
                        os.remove(data_path)
                        count += 1
                        print(f"Deleted: {data_file}")
                    except Exception as e:
                        print(f"Failed to delete {data_file}: {e}")
                print(f"✓ Deleted {count} data files")
        
        elif choice == 2:
            confirm = input(f"Delete {len(model_files)} model files? (yes/no): ")
            if confirm.lower() == 'yes':
                count = 0
                for model_file in model_files:
                    try:
                        if model_manager.delete_model(model_file):
                            count += 1
                            print(f"Deleted: {model_file}")
                        else:
                            print(f"Failed to delete: {model_file}")
                    except Exception as e:
                        print(f"Error deleting {model_file}: {e}")
                print(f"✓ Deleted {count} model files")
        
        elif choice == 3:
            confirm = input("Delete ALL files? (yes/no): ")
            if confirm.lower() == 'yes':
                total_count = 0
                
                # Delete data files
                for data_file in data_files:
                    try:
                        import os
                        data_path = os.path.join(data_manager.data_dir, data_file)
                        os.remove(data_path)
                        total_count += 1
                        print(f"Deleted data: {data_file}")
                    except Exception as e:
                        print(f"Failed to delete data {data_file}: {e}")
                
                # Delete model files
                for model_file in model_files:
                    try:
                        if model_manager.delete_model(model_file):
                            total_count += 1
                            print(f"Deleted model: {model_file}")
                        else:
                            print(f"Failed to delete model: {model_file}")
                    except Exception as e:
                        print(f"Error deleting model {model_file}: {e}")
                
                print(f"✓ Deleted {total_count} files")
        
        elif choice == 4:
            print("Cleanup cancelled")
        
        else:
            print("Invalid option")
    
    except ValueError:
        print("Invalid selection")


def main():
    """Main application loop"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/spectrum_alert.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("SpectrumAlert v2.0 starting...")
    
    # Initialize configuration and data manager
    try:
        config = ConfigManager()
        config.load_config()
        
        data_manager = DataManager()
        model_manager = ModelManager()
        
        logger.info("System initialized successfully")
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)
    
    # Main menu loop
    while True:
        try:
            show_banner()
            
            print("\n=== Main Menu ===")
            print("1. Collect RF Data")
            print("2. Train ML Models")
            print("3. Monitor Spectrum")
            print("4. Automated Workflow")
            print("5. Check Data/Models")
            print("6. Cleanup Old Data")
            print("7. System Information")
            print("8. Exit")
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                collect_data_menu(config, data_manager)
            
            elif choice == '2':
                train_models_menu(config, data_manager, model_manager)
            
            elif choice == '3':
                monitor_spectrum_menu(config, data_manager, model_manager)
            
            elif choice == '4':
                automated_workflow_menu(config, data_manager, model_manager)
            
            elif choice == '5':
                check_system_status(config, data_manager, model_manager)
            
            elif choice == '6':
                cleanup_data_menu(data_manager, model_manager)
            
            elif choice == '7':
                print("\n=== System Information ===")
                print(f"SpectrumAlert v2.0")
                print(f"Configuration: {config.config_file}")
                print(f"Data directory: data/")
                print(f"Models directory: models/")
                print(f"Logs directory: logs/")
                ham_bands = config.ham_bands
                print(f"Ham bands: {len(ham_bands)} configured")
                for band in ham_bands:
                    print(f"  {band.start_freq/1e6:.1f} - {band.end_freq/1e6:.1f} MHz")
            
            elif choice == '8':
                print("Exiting SpectrumAlert...")
                logger.info("SpectrumAlert shutting down normally")
                break
            
            else:
                print("Invalid option. Please try again.")
            
            if choice != '8':
                input("\nPress Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\nExiting SpectrumAlert...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            print(f"An error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()