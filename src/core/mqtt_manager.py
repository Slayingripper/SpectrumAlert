"""
MQTT communication manager for SpectrumAlert
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import paho.mqtt.client as mqtt
from src.utils.config_manager import MQTTConfig

logger = logging.getLogger(__name__)


class MQTTManager:
    """Manages MQTT communications for SpectrumAlert"""
    
    def __init__(self, config: MQTTConfig):
        self.config = config
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
        if self.config.username and self.config.password:
            self.client.username_pw_set(self.config.username, self.config.password)
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection"""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker at {self.config.broker}:{self.config.port}")
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
            logger.info(f"Connecting to MQTT broker at {self.config.broker}:{self.config.port}")
            self.client.connect(self.config.broker, self.config.port, timeout)
            self.client.loop_start()
            
            # Wait for connection
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self.connected
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Disconnected from MQTT broker")
        except Exception as e:
            logger.error(f"Error disconnecting from MQTT broker: {e}")
    
    def publish_anomaly(self, frequency: float, score: float, 
                       coordinates: Optional[tuple] = None, 
                       metadata: Optional[Dict] = None) -> bool:
        """Publish anomaly detection result"""
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'frequency_mhz': frequency / 1e6,
                'anomaly_score': score,
                'metadata': metadata or {}
            }
            
            if coordinates:
                message['latitude'] = coordinates[0]
                message['longitude'] = coordinates[1]
            
            return self._publish_json(self.config.topic_anomalies, message)
        except Exception as e:
            logger.error(f"Error publishing anomaly: {e}")
            return False
    
    def publish_signal_strength(self, frequency: float, strength_db: float,
                              coordinates: Optional[tuple] = None) -> bool:
        """Publish signal strength measurement"""
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'frequency_mhz': frequency / 1e6,
                'signal_strength_db': strength_db
            }
            
            if coordinates:
                message['latitude'] = coordinates[0]
                message['longitude'] = coordinates[1]
            
            return self._publish_json(self.config.topic_signal_strength, message)
        except Exception as e:
            logger.error(f"Error publishing signal strength: {e}")
            return False
    
    def publish_modulation(self, frequency: float, modulation_type: str,
                          confidence: float = 1.0) -> bool:
        """Publish modulation classification result"""
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'frequency_mhz': frequency / 1e6,
                'modulation_type': modulation_type,
                'confidence': confidence
            }
            
            return self._publish_json(self.config.topic_modulation, message)
        except Exception as e:
            logger.error(f"Error publishing modulation: {e}")
            return False
    
    def publish_coordinates(self, latitude: float, longitude: float,
                          metadata: Optional[Dict] = None) -> bool:
        """Publish receiver coordinates"""
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'latitude': latitude,
                'longitude': longitude,
                'metadata': metadata or {}
            }
            
            return self._publish_json(self.config.topic_coordinates, message)
        except Exception as e:
            logger.error(f"Error publishing coordinates: {e}")
            return False
    
    def publish_custom(self, topic: str, data: Dict[str, Any]) -> bool:
        """Publish custom data to specified topic"""
        try:
            return self._publish_json(topic, data)
        except Exception as e:
            logger.error(f"Error publishing custom data: {e}")
            return False
    
    def _publish_json(self, topic: str, data: Dict[str, Any]) -> bool:
        """Publish JSON data to topic"""
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        try:
            with self._lock:
                payload = json.dumps(data, default=str)
                result = self.client.publish(topic, payload)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.debug(f"Published to {topic}: {payload[:100]}...")
                    return True
                else:
                    logger.error(f"Failed to publish to {topic}, return code: {result.rc}")
                    return False
        except Exception as e:
            logger.error(f"Error publishing JSON to {topic}: {e}")
            return False
    
    def subscribe(self, topic: str, callback: Callable[[str, str], None]) -> bool:
        """Subscribe to a topic with callback"""
        try:
            if not self.connected:
                logger.warning("Not connected to MQTT broker")
                return False
            
            result = self.client.subscribe(topic)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.message_callbacks[topic] = callback
                logger.info(f"Subscribed to topic: {topic}")
                return True
            else:
                logger.error(f"Failed to subscribe to {topic}, return code: {result[0]}")
                return False
        except Exception as e:
            logger.error(f"Error subscribing to {topic}: {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from a topic"""
        try:
            if not self.connected:
                return True  # Already disconnected
            
            result = self.client.unsubscribe(topic)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                if topic in self.message_callbacks:
                    del self.message_callbacks[topic]
                logger.info(f"Unsubscribed from topic: {topic}")
                return True
            else:
                logger.error(f"Failed to unsubscribe from {topic}, return code: {result[0]}")
                return False
        except Exception as e:
            logger.error(f"Error unsubscribing from {topic}: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to MQTT broker"""
        return self.connected
    
    def get_status(self) -> Dict[str, Any]:
        """Get MQTT manager status"""
        return {
            'connected': self.connected,
            'broker': self.config.broker,
            'port': self.config.port,
            'subscribed_topics': list(self.message_callbacks.keys()),
            'client_id': getattr(self.client, '_client_id', 'unknown')
        }


class MQTTMessageBuffer:
    """Buffer for MQTT messages when connection is unavailable"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        self._lock = threading.Lock()
    
    def add_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Add message to buffer"""
        with self._lock:
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)  # Remove oldest message
            
            message = {
                'topic': topic,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            self.buffer.append(message)
    
    def flush_to_mqtt(self, mqtt_manager: MQTTManager) -> int:
        """Flush buffered messages to MQTT"""
        if not mqtt_manager.is_connected():
            return 0
        
        sent_count = 0
        with self._lock:
            while self.buffer:
                message = self.buffer.pop(0)
                success = mqtt_manager.publish_custom(message['topic'], message['data'])
                if success:
                    sent_count += 1
                else:
                    # Put message back if send failed
                    self.buffer.insert(0, message)
                    break
        
        if sent_count > 0:
            logger.info(f"Flushed {sent_count} buffered messages to MQTT")
        
        return sent_count
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        with self._lock:
            return len(self.buffer)
    
    def clear_buffer(self) -> None:
        """Clear message buffer"""
        with self._lock:
            self.buffer.clear()
