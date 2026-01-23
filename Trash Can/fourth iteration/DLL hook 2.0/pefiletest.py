import json
import time
import os
from datetime import datetime

class TMInterfaceDataReader:
    def __init__(self, output_file="game_data.json"):
        self.output_file = output_file
        self.last_modified = 0
        
    def is_game_running(self):
        """Check if TMInterface is running"""
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                if 'tminterface' in proc.info['name'].lower():
                    return True
        except:
            pass
        return False
    
    def read_latest_data(self):
        """Read the latest game data from file"""
        try:
            # Check if file exists and has been modified
            if not os.path.exists(self.output_file):
                return None
                
            current_modified = os.path.getmtime(self.output_file)
            
            # Only read if file has changed since last read
            if current_modified > self.last_modified:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                
                self.last_modified = current_modified
                return data
            else:
                return None
                
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
    
    def get_simple_data(self):
        """Get simplified, easy-to-use data"""
        data = self.read_latest_data()
        if not data:
            return None
            
        # Extract what you need
        simple_data = {
            "timestamp": datetime.fromtimestamp(data.get("timestamp", 0)),
            "speed": data.get("player", {}).get("speed", 0),
            "lap": data.get("player", {}).get("lap", 0),
            "checkpoint": data.get("player", {}).get("checkpoint", 0),
            "position": data.get("player", {}).get("position", [0,0,0])
        }
        
        return simple_data
    
    def monitor_continuous(self):
        """Continuously monitor and print new data"""
        print("Monitoring TMInterface data...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                data = self.get_simple_data()
                if data:
                    print(f"Speed: {data['speed']:.1f} km/h | "
                          f"Lap: {data['lap']} | "
                          f"Checkpoint: {data['checkpoint']}")
                
                time.sleep(0.1)  # Check every 100ms
                
        except KeyboardInterrupt:
            print("\nStopped monitoring")

# Usage examples
def main():
    reader = TMInterfaceDataReader("game_data.json")
    
    # Single read
    data = reader.get_simple_data()
    if data:
        print("Current game data:")
        print(f"Speed: {data['speed']}")
        print(f"Lap: {data['lap']}")
        print(f"Checkpoint: {data['checkpoint']}")
    
    # Continuous monitoring
    # reader.monitor_continuous()

if __name__ == "__main__":
    main()
