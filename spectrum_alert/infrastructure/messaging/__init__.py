"""
MQTT messaging interface for SpectrumAlert v3.0
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import paho.mqtt.client as mqtt
from spectrum_alert.core.exceptions import NetworkError

logger = logging.getLogger(__name__)


class MQTTManager:
    """Manages MQTT communications for SpectrumAlert"""
    
    def __init__(self, broker: str, port: int = 1883, username: Optional[str] = None, 
                 password: Optional[str] = None):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.client = mqtt.Client()
        self.connected = False
        self.message_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Setup MQTT client with callbacks"""
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        
        # Set authentication if provided
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection"""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
        else:
            self.connected = False
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for disconnection"""
        self.connected = False
        if rc != 0:
            logger.warning("Unexpected disconnection from MQTT broker")
        else:
            logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """Callback for received messages"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.debug(f"Received message on topic {topic}: {payload}")
            
            # Call registered callback if available
            if topic in self.message_callbacks:
                self.message_callbacks[topic](topic, payload)
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for published messages"""
        logger.debug(f"Message {mid} published successfully")
    
    def connect(self, timeout: int = 60) -> bool:
        """Connect to MQTT broker"""
        try:
            logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}")
            self.client.connect(self.broker, self.port, timeout)
            self.client.loop_start()
            
            # Wait for connection
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self.connected
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            raise NetworkError(f"Failed to connect to MQTT broker: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Disconnected from MQTT broker")
        except Exception as e:
            logger.error(f"Error disconnecting from MQTT broker: {e}")
    
    def publish_anomaly(self, frequency: float, score: float, 
                       details: Optional[Dict[str, Any]] = None) -> bool:
        """Publish anomaly detection to MQTT"""
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        try:
            message = {
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'anomaly',
                'frequency_hz': frequency,
                'confidence_score': score,
                'details': details or {}
            }
            
            topic = "spectrum_alert/anomaly"
            result = self.client.publish(topic, json.dumps(message))
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Anomaly published to {topic}")
                return True
            else:
                logger.error(f"Failed to publish anomaly: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing anomaly: {e}")
            return False
    
    def publish_status(self, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Publish system status to MQTT"""
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        try:
            message = {
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'status',
                'status': status,
                'details': details or {}
            }
            
            topic = "spectrum_alert/status"
            result = self.client.publish(topic, json.dumps(message))
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Status published to {topic}")
                return True
            else:
                logger.error(f"Failed to publish status: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
            return False
    
    def subscribe(self, topic: str, callback: Optional[Callable] = None) -> bool:
        """Subscribe to MQTT topic"""
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        try:
            result = self.client.subscribe(topic)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Subscribed to topic: {topic}")
                if callback:
                    self.message_callbacks[topic] = callback
                return True
            else:
                logger.error(f"Failed to subscribe to topic {topic}: {result[0]}")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to topic {topic}: {e}")
            return False
