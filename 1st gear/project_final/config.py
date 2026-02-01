# config.py
class TrainingConfig:
    # Reward multipliers (tune these to change learning emphasis)
    SPEED_REWARD_MULT = 0.15        # Default 0.03, increase to prioritize speed
    ACCELERATION_BONUS = 0.15       # Default 0.08
    CHECKPOINT_REWARD = 1000.0      # Default 300, massive boost for progress
    FINISH_REWARD = 5000.0          # Default 1500
    OSCILLATION_PENALTY = -20.0     # Penalty for steering left-right spam
    
    # Detection parameters
    FINISH_CONFIRMATION_FRAMES = 5  # Frames to confirm finish
    CHECKPOINT_CONFIRMATION = 3     # Consecutive frames to confirm checkpoint
    POSITION_TOLERANCE = 3.0        # Meters from finish to consider "locked"
    
    # Memory
    MAX_MEMORY = 200_000            # Increased from 100k
    CHECKPOINT_PRIORITY_WEIGHT = 3.0  # How much to boost checkpoint experiences
    
    # Training
    BATCH_SIZE = 128                # Larger batch for more stable learning
    TARGET_UPDATE_EVERY = 10        # Less frequent target updates for stability
    EPSILON_DECAY = 0.998           # Slower decay for more exploration
    LEARNING_RATE = 0.0005          # Lower LR for stability with larger batches