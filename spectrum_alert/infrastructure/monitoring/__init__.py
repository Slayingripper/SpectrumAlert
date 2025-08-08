"""
System monitoring infrastructure for SpectrumAlert v3.0
"""

import psutil
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from spectrum_alert.core.exceptions import MonitoringError

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self._last_update = time.time()
        self._metrics_history = []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            current_time = time.time()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network I/O
            network = psutil.net_io_counters()
            bytes_sent = network.bytes_sent
            bytes_recv = network.bytes_recv
            
            # System uptime
            uptime_seconds = current_time - self.start_time
            
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'uptime_seconds': uptime_seconds,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'available_gb': round(memory_available_gb, 2),
                    'total_gb': round(memory_total_gb, 2),
                    'used_gb': round(memory_total_gb - memory_available_gb, 2)
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': round(disk_free_gb, 2),
                    'total_gb': round(disk_total_gb, 2),
                    'used_gb': round(disk_total_gb - disk_free_gb, 2)
                },
                'network': {
                    'bytes_sent': bytes_sent,
                    'bytes_received': bytes_recv
                }
            }
            
            self._last_update = current_time
            return metrics
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            raise MonitoringError(f"Failed to get system metrics: {e}")
    
    def get_process_metrics(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics for a specific process (current process if pid not specified)"""
        try:
            if pid is None:
                process = psutil.Process()
            else:
                process = psutil.Process(pid)
            
            # Process info
            info = process.as_dict(attrs=[
                'pid', 'name', 'status', 'create_time',
                'cpu_percent', 'memory_info', 'num_threads'
            ])
            
            # Convert memory to MB
            memory_info = info.get('memory_info')
            if memory_info:
                info['memory_mb'] = round(memory_info.rss / (1024**2), 2)
                info['memory_vms_mb'] = round(memory_info.vms / (1024**2), 2)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'process': info
            }
        except Exception as e:
            logger.error(f"Error getting process metrics: {e}")
            raise MonitoringError(f"Failed to get process metrics: {e}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            metrics = self.get_system_metrics()
            
            # Define health thresholds
            health_status = {
                'overall': 'healthy',
                'warnings': [],
                'errors': [],
                'metrics': metrics
            }
            
            # Check CPU usage
            if metrics['cpu']['percent'] > 90:
                health_status['errors'].append('High CPU usage (>90%)')
                health_status['overall'] = 'critical'
            elif metrics['cpu']['percent'] > 70:
                health_status['warnings'].append('Elevated CPU usage (>70%)')
                if health_status['overall'] == 'healthy':
                    health_status['overall'] = 'warning'
            
            # Check memory usage
            if metrics['memory']['percent'] > 95:
                health_status['errors'].append('Critical memory usage (>95%)')
                health_status['overall'] = 'critical'
            elif metrics['memory']['percent'] > 80:
                health_status['warnings'].append('High memory usage (>80%)')
                if health_status['overall'] == 'healthy':
                    health_status['overall'] = 'warning'
            
            # Check disk usage
            if metrics['disk']['percent'] > 95:
                health_status['errors'].append('Critical disk usage (>95%)')
                health_status['overall'] = 'critical'
            elif metrics['disk']['percent'] > 85:
                health_status['warnings'].append('High disk usage (>85%)')
                if health_status['overall'] == 'healthy':
                    health_status['overall'] = 'warning'
            
            # Check available memory
            if metrics['memory']['available_gb'] < 0.5:
                health_status['errors'].append('Low available memory (<0.5GB)')
                health_status['overall'] = 'critical'
            elif metrics['memory']['available_gb'] < 1.0:
                health_status['warnings'].append('Low available memory (<1GB)')
                if health_status['overall'] == 'healthy':
                    health_status['overall'] = 'warning'
            
            return health_status
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'overall': 'error',
                'warnings': [],
                'errors': [f'Failed to check system health: {e}'],
                'metrics': {}
            }
    
    def log_metrics(self, level: str = 'INFO') -> None:
        """Log current system metrics"""
        try:
            metrics = self.get_system_metrics()
            
            log_msg = (
                f"System Metrics - "
                f"CPU: {metrics['cpu']['percent']:.1f}%, "
                f"Memory: {metrics['memory']['percent']:.1f}% "
                f"({metrics['memory']['used_gb']:.1f}GB/{metrics['memory']['total_gb']:.1f}GB), "
                f"Disk: {metrics['disk']['percent']:.1f}% "
                f"({metrics['disk']['used_gb']:.1f}GB/{metrics['disk']['total_gb']:.1f}GB)"
            )
            
            if level.upper() == 'DEBUG':
                logger.debug(log_msg)
            elif level.upper() == 'WARNING':
                logger.warning(log_msg)
            elif level.upper() == 'ERROR':
                logger.error(log_msg)
            else:
                logger.info(log_msg)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def get_uptime_string(self) -> str:
        """Get formatted uptime string"""
        try:
            uptime_seconds = time.time() - self.start_time
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except Exception as e:
            logger.error(f"Error getting uptime: {e}")
            return "Unknown"
