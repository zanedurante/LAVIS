from transformers import  AutoTokenizer

def init_tokenizer(base_model_name):
        tokenizer =  AutoTokenizer.from_pretrained(base_model_name)

        # actions for language table
        for i in range(21):
            tokenizer.add_tokens([f"[ROBOTACTIONX{i}]", f"[ROBOTACTIONY{i}]"])
            tokenizer.add_tokens([f"[ROBOTEETX{i}]", f"[ROBOTEETY{i}]"])
            tokenizer.add_tokens([f"[ROBOTEETTX{i}]", f"[ROBOTEETTY{i}]"])
        
        tokenizer.add_tokens(['[ENDOFACTION]'])
        tokenizer.add_tokens(['[STARTACTION]'])
        tokenizer.add_tokens(['[TERMINAL]'])
        
        tokenizer.add_tokens(['[STARTEET]'])
        tokenizer.add_tokens(['[ENDOFEET]'])

        tokenizer.add_tokens(['[STARTEETT]'])
        tokenizer.add_tokens(['[ENDOFEETT]'])
        tokenizer.pad_token = tokenizer.eos_token

        # actions for minecraft

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
        }
        new_tokens = list(NOOP_ACTION.keys())
        new_tokens = [ f'[{new_token}]' for new_token in new_tokens ]
        tokenizer.add_tokens(new_tokens)
        camera_actions = []
        for i in range(-49, 50):
            camera_actionx = f'[CAMERAX{i}]'
            camera_actiony = f'[CAMERAY{i}]'
            camera_actions.extend([camera_actionx, camera_actiony])
        
        tokenizer.add_tokens(camera_actions)

        # actions for calvin
        for i in range(101):
            tokenizer.add_tokens([f"[ROBOTACTION0_{i}]", f"[ROBOTACTION1_{i}]", f"[ROBOTACTION2_{i}]",
                                       f"[ROBOTACTION3_{i}]", f"[ROBOTACTION4_{i}]", f"[ROBOTACTION5_{i}]"])
        
        for i in range(14):
            for j in range(101):
                tokenizer.add_tokens([f"[ROBOTSTATE{i}_{j}]"])
        
        tokenizer.add_tokens(['[GRIPPER_OPEN]', '[GRIPPER_CLOSE]', '[GRIPPER_OPENED]', '[GRIPPER_CLOSED]'])

        return tokenizer