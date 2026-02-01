import pygetwindow as gw
import mss
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
import time
import re
from datetime import datetime

class TrackmaniaAI:
    def __init__(self):
        self.window_title = "Trackmania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.10)"
        self.window = None
        self.start_time = None
        self.race_started = False
        self.checkpoint_count = 0
        self.best_time = float('inf')
        self.current_lap_start_time = None
        self.lap_times = []
        
        # Track coordinates (x, z, y)
        self.start_coords = (171.201, 90.005, 688.001)
        self.finish_coords = (22.334, 113.359, 176.363)
        
        # Waypoints for the track (x, z, y)
        self.waypoints = [
            (243.983, 88.014, 687.997),   # Start
            (281.946, 88.014, 741.501),   # Turn 1
            (407.122, 88.014, 749.829),   # Turn 2
            (444.749, 88.014, 695.584),   # Turn 3
            (970.589, 24.014, 673.361),   # Turn 4
            (971.381, 24.014, 209.084),   # Turn 5
            (639.263, 24.013, 176.736),   # Turn 6
            (22.334, 113.359, 176.363)    # Finish
        ]
        
        # Track width for positioning checks
        self.track_width = 37
        
    def get_window(self):
        """Get the Trackmania window object"""
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            raise ValueError(f"Window with title '{self.window_title}' not found.")
        return windows[0]
    
    def capture_window_screenshot(self):
        """Capture screenshot of the Trackmania window"""
        try:
            self.window = self.get_window()
            left, top, width, height = self.window.left, self.window.top, self.window.width, self.window.height
            
            with mss.mss() as sct:
                screenshot = sct.grab((left, top, left + width, top + height))
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            
            return img
        except Exception as e:
            print(f"Error capturing window: {e}")
            return None
    
    def extract_coordinates_from_text(self, text):
        """Extract position and velocity coordinates from OCR text"""
        try:
            # Extract position coordinates (x, z, y)
            pos_match = re.search(r'Position\?\s+(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)', text)
            
            # Extract velocity components
            vel_match = re.search(r'VEER\s+(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)', text)
            
            position = None
            velocity = None
            
            if pos_match:
                # Reorder coordinates from (x, z, y)
                x, z, y = map(float, pos_match.groups())
                position = (x, z, y)
                
            if vel_match:
                # Reorder velocity components from (x, z, y)
                x, z, y = map(float, vel_match.groups())
                velocity = (x, z, y)
                
            return position, velocity
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return None, None
    
    def extract_checkpoint_info(self, text):
        """Extract checkpoint information from OCR text"""
        try:
            # Extract checkpoint count (e.g., "Checkpoints: 1/3")
            checkpoint_match = re.search(r'Checkpoints:\s+(\d+)/(\d+)', text)
            
            if checkpoint_match:
                current_checkpoint = int(checkpoint_match.group(1))
                total_checkpoints = int(checkpoint_match.group(2))
                return current_checkpoint, total_checkpoints
            return 0, 0
        except Exception as e:
            print(f"Error extracting checkpoints: {e}")
            return 0, 0
    
    def extract_speed_info(self, text):
        """Extract speed information from OCR text"""
        try:
            # Extract real speed
            speed_match = re.search(r'Real Speed:\s+(\d+\.\d+)', text)
            
            if speed_match:
                return float(speed_match.group(1))
            return 0.0
        except Exception as e:
            print(f"Error extracting speed: {e}")
            return 0.0
    
    def extract_yaw_pitch_roll(self, text):
        """Extract yaw, pitch, and roll information"""
        try:
            yaw_match = re.search(r'Yaw:\s+(\d+\.\d+)', text)
            pitch_match = re.search(r'Pitch:\s+(\d+\.\d+)', text)
            roll_match = re.search(r'Roll:\s+(\d+\.\d+)', text)
            
            yaw = float(yaw_match.group(1)) if yaw_match else 0.0
            pitch = float(pitch_match.group(1)) if pitch_match else 0.0
            roll = float(roll_match.group(1)) if roll_match else 0.0
            
            return yaw, pitch, roll
        except Exception as e:
            print(f"Error extracting orientation: {e}")
            return 0.0, 0.0, 0.0
    
    def process_ocr_text(self, text):
        """Process OCR text to extract all relevant information"""
        position, velocity = self.extract_coordinates_from_text(text)
        checkpoint, total_checkpoints = self.extract_checkpoint_info(text)
        speed = self.extract_speed_info(text)
        yaw, pitch, roll = self.extract_yaw_pitch_roll(text)
        
        return {
            'position': position,
            'velocity': velocity,
            'checkpoint': checkpoint,
            'total_checkpoints': total_checkpoints,
            'speed': speed,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two 3D points"""
        if pos1 is None or pos2 is None:
            return float('inf')
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
    
    def check_checkpoint_completion(self, position):
        """Check if we've passed a checkpoint"""
        if position is None:
            return False
            
        # Check if we're near the finish line
        distance_to_finish = self.calculate_distance(position, self.finish_coords)
        if distance_to_finish < 10:  # Within 10 units of finish
            return True
        return False
    
    def calculate_reward(self, position, checkpoint_count, speed):
        """Calculate reward based on position, checkpoints, and speed"""
        if position is None:
            return -100
            
        # Base reward for checkpoint completion
        checkpoint_reward = 0
        if checkpoint_count > self.checkpoint_count:
            checkpoint_reward = 50 * (checkpoint_count - self.checkpoint_count)
            self.checkpoint_count = checkpoint_count
            
        # Distance to finish line reward (closer is better)
        distance_to_finish = self.calculate_distance(position, self.finish_coords)
        finish_reward = max(0, 1000 - distance_to_finish) / 10
        
        # Speed reward (higher speed is generally better, but not too fast for control)
        speed_reward = min(speed * 10, 50) if speed > 0.01 else 0
        
        # Time-based reward (faster completion gets higher reward)
        time_reward = 0
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0 and checkpoint_count == 3:  # Finished race
                time_reward = max(0, 1000 - elapsed_time * 10)  # Higher reward for faster times
                
        total_reward = checkpoint_reward + finish_reward + speed_reward + time_reward
        
        return total_reward
    
    def is_race_finished(self, checkpoint_count, total_checkpoints):
        """Check if race is finished (all checkpoints passed)"""
        return checkpoint_count == total_checkpoints and total_checkpoints > 0
    
    def run_continuous_monitoring(self):
        """Continuously monitor Trackmania for position and time data"""
        print("Starting continuous monitoring...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Capture screenshot
                screenshot = self.capture_window_screenshot()
                if not screenshot:
                    time.sleep(1)
                    continue
                
                # Crop area where text information is likely located
                # Adjust these coordinates based on your actual window layout
                crop_coords = (15, 60, 345, 285)  # x1, y1, x2, y2
                cropped_image = screenshot.crop(crop_coords)
                
                # Preprocess image for OCR
                gray_image = cropped_image.convert('L')
                np_image = np.array(gray_image)
                
                # Apply thresholding
                _, thresh = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_image = Image.fromarray(thresh)
                
                # Perform OCR
                text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 6')
                
                # Process the extracted text
                data = self.process_ocr_text(text)
                
                if data['position']:
                    position = data['position']
                    print(f"Position: {position}")
                    print(f"Checkpoint: {data['checkpoint']}/{data['total_checkpoints']}")
                    print(f"Speed: {data['speed']}")
                    print(f"Yaw: {data['yaw']}")
                    
                    # Start race timer when first speed change is detected
                    if not self.race_started and data['speed'] > 0.01:
                        self.start_time = time.time()
                        self.race_started = True
                        print("Race started!")
                        
                    # Check for lap completion
                    if self.is_race_finished(data['checkpoint'], data['total_checkpoints']):
                        if self.start_time:
                            race_time = time.time() - self.start_time
                            print(f"Race completed in {race_time:.2f} seconds")
                            
                            # Update best time
                            if not hasattr(self, 'best_time') or race_time < self.best_time:
                                self.best_time = race_time
                                print(f"New best time: {race_time:.2f} seconds")
                                
                        self.race_started = False
                        self.checkpoint_count = 0
                        
                    # Calculate reward (optional - for reinforcement learning)
                    reward = self.calculate_reward(position, data['checkpoint'], data['speed'])
                    print(f"Reward: {reward:.2f}")
                    
                else:
                    print("No position data detected")
                
                time.sleep(1)  # Wait before next capture
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error during monitoring: {e}")

# You'll need to add this class definition at the top of your file
class TrackmaniaMonitor:
    def __init__(self):
        self.start_time = None
        self.race_started = False
        self.checkpoint_count = 0
        self.best_time = float('inf')
        

if __name__ == "__main__":
    monitor = TrackmaniaAI()
    monitor.run_continuous_monitoring()
