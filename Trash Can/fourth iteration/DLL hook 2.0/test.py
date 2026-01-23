import ctypes

# Load the DLL using a raw string
tminterface = ctypes.CDLL(r'C:\Users\Kylmä Sami\AppData\Local\TMLoader\database\TmForever\products\TMInterface\2.2.1\TMInterface.dll')

# Define function signatures
tminterface.GetPlayerInfo.argtypes = [ctypes.c_int]
tminterface.GetPlayerInfo.restype = ctypes.c_char_p

tminterface.GetMapName.argtypes = []
tminterface.GetMapName.restype = ctypes.c_char_p

tminterface.GetGameState.argtypes = []
tminterface.GetGameState.restype = ctypes.c_int

def get_player_info(player_id):
    player_info = tminterface.GetPlayerInfo(player_id)
    return player_info.decode('utf-8') if player_info else None

def get_map_name():
    map_name = tminterface.GetMapName()
    return map_name.decode('utf-8') if map_name else None

def get_game_state():
    game_state = tminterface.GetGameState()
    state_mapping = {0: "Menu", 1: "In Race"}
    return state_mapping.get(game_state, "Unknown State")

# Example usage
player_id = 1  # Replace with the actual player ID you want to query
player_info = get_player_info(player_id)
if player_info:
    print(f"Player Info: {player_info}")

map_name = get_map_name()
if map_name:
    print(f"Current Map: {map_name}")

game_state = get_game_state()
print(f"Game State: {game_state}")
