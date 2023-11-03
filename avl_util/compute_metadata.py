import os
import json
import cv2
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.

Press any button the window to proceed to the next frame.
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0



def top_k_common_items(lst, k):
    count = Counter(lst)
    return [item for item, _ in count.most_common(k)]
def convert_to_text(action_dict):
    # Initialize an empty list to store the non-zero action strings.
    non_zero_actions = []
    
    for key, value in action_dict.items():
        # Handling the "camera" key separately.
        if key == "camera":
                if any(value != 0):
                    non_zero_actions.append('[mouse dx:' + str(int(value[0])) + ', mouse dy:' + str(int(value[1])) + ']')

        else:
            # Add the key to the list if the value is non-zero.
            if value != 0:
                non_zero_actions.append(key)
    
    return non_zero_actions

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

def process_file(metadata_dir, video_file, metadata_file, chunk_size_frames):
    video_path = os.path.join(metadata_dir, video_file)
    metadata_path = os.path.join(metadata_dir, metadata_file)
    data_chunk = {
        "video": [],
        "start_frame": [],
        "end_frame": [],
        "actions": [],
        "caption": []
    }

    try:
        with open(metadata_path) as json_file:
            json_lines = json_file.readlines()
            metadata = "[" + ",".join(json_lines) + "]"
            metadata = json.loads(metadata)   
    except:
        return None  # Return None if failed to load metadata

    # ... rest of the code to process the metadata and fill the data_chunk dictionary ...
        
    attack_is_stuck = False
    last_hotbar = 0
    actions = []
    duration_s, fps, total_frames = len(metadata) / 20.0, 20, len(metadata)
    num_chunks = int(total_frames // chunk_size_frames) #+ (1 if total_frames % chuck_size_frames > 0 else 0)

    for i in range(len(metadata)):
        
        step_data = metadata[i]

        if i == 0:
            # Check if attack will be stuck down
            if step_data["mouse"]["newButtons"] == [0]:
                attack_is_stuck = True
        elif attack_is_stuck:
            # Check if we press attack down, then it might not be stuck
            if 0 in step_data["mouse"]["newButtons"]:
                attack_is_stuck = False
        # If still stuck, remove the action
        if attack_is_stuck:
            step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

        action, is_null_action = json_action_to_env_action(step_data)

        # Update hotbar selection
        if 'hotbar' in step_data:
            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar
        actions.append(action)

    # Append video path and metadata in chunks
    for i in range(num_chunks):
        data_chunk["video"].append(video_path)
        start_frame = i * chunk_size_frames
        stop_frame = min((i + 1) * chunk_size_frames, total_frames) 
        if start_frame == stop_frame:
            continue
        
        data_chunk["start_frame"].append(start_frame)
        data_chunk['end_frame'].append(stop_frame)

        
        
        # Assuming 20 FPS, get actions corresponding to the time chunk
        start_frame, stop_frame = int(start_frame ), int(stop_frame)
        tmp = actions[start_frame:stop_frame]

        
        if isinstance(tmp, list):
            data_chunk["actions"].append(actions[start_frame:stop_frame])
        elif isinstance(tmp, str):
            data_chunk["actions"].append([actions[start_frame:stop_frame]])
            tmp = [tmp]
        
        
        
        all_actions = []
        captions = ""
        for idx , action in enumerate(tmp):
            text_actions = convert_to_text(action)
            if text_actions:
                captions += f'frame {idx}: ' + ' '.join(text_actions) + '\n'
            else:
                captions += f"frame {idx}:  No action\n" 

        if captions:
            data_chunk["caption"].append(captions)
        else:
            data_chunk["caption"].append("No action in the video")

    return data_chunk

def load_metadata(chunk_size_frames=4, metadata_dir='./mnt/dataset_mnt/', filesname='./mnt/all_files.txt'):
    with open(filesname) as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    video_files = sorted([f for f in files if f.endswith('.mp4')])
    metadata_files = sorted([f for f in files if f.endswith('.jsonl')])

    data = {
        "video": [],
        "start_frame": [],
        "end_frame": [],
        "actions": [],
        "caption": []
    }

    # Using ThreadPoolExecutor to parallelize file processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Schedule the processing of each file.
        futures = [executor.submit(process_file, metadata_dir, video_file, metadata_file, chunk_size_frames)
                   for video_file, metadata_file in zip(video_files, metadata_files)]

        # Progress bar for futures
        for future in tqdm(as_completed(futures), total=len(futures)):
            data_chunk = future.result()
            if data_chunk:
                data["video"].extend(data_chunk["video"])
                data["start_frame"].extend(data_chunk["start_frame"])
                data["end_frame"].extend(data_chunk["end_frame"])
                data["actions"].extend(data_chunk["actions"])
                data["caption"].extend(data_chunk["caption"])
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(metadata_dir, 'metadata.csv'), index=False)


if __name__ == "__main__":
    load_metadata(chunk_size_frames=4, metadata_dir= './mnt/datasets_mnt/')