import os
import json
import numpy as np
from tqdm import tqdm

def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "attack": 0,
    "use": 0,
    "pickItem": 0,
    "camera": np.array([0, 0]),
}
CAMERA_SCALER = 360.0 / 2400.0
filesname = './mnt/all_files.txt'
with open(filesname) as f:
        files = f.readlines()
files = [x.strip() for x in files]
video_files = sorted([f for f in files if f.endswith('.mp4')])
metadata_files = sorted([f for f in files if f.endswith('.jsonl')])

# Define the list to hold the actions
actions = []
min_camera_x, max_camera_x = float('inf'), float('-inf')
min_camera_y, max_camera_y = float('inf'), float('-inf')
# Make sure to check that the directory exists and handle any potential errors
metadata_dir = '/mnt/dataset_mnt/'

for metadata_file in tqdm(metadata_files):
    metadata_path = os.path.join(metadata_dir, metadata_file)

    try:
        with open(metadata_path) as json_file:
            json_lines = json_file.readlines()
            metadata = "[" + ",".join(json_lines) + "]"
            metadata = json.loads(metadata)   
    except:
        continue

    attack_is_stuck = False
    last_hotbar = 0
    for i, step_data in enumerate(metadata):                
        if i == 0 and step_data["mouse"]["newButtons"] == [0]:
            attack_is_stuck = True
        elif attack_is_stuck and 0 in step_data["mouse"]["newButtons"]:
            attack_is_stuck = False
        if attack_is_stuck:
            step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

        action, is_null_action = json_action_to_env_action(step_data)

        # Update hotbar selection
        current_hotbar = step_data["hotbar"]
        if current_hotbar != last_hotbar:
            action["hotbar.{}".format(current_hotbar + 1)] = 1
        last_hotbar = current_hotbar
        actions.append(action)

        camera_x, camera_y = action["camera"]
        min_camera_x = min(min_camera_x, camera_x)
        max_camera_x = max(max_camera_x, camera_x)
        min_camera_y = min(min_camera_y, camera_y)
        max_camera_y = max(max_camera_y, camera_y)



print(f"Minimum camera x (yaw): {min_camera_x}")
print(f"Maximum camera x (yaw): {max_camera_x}")
print(f"Minimum camera y (pitch): {min_camera_y}")
print(f"Maximum camera y (pitch): {max_camera_y}")