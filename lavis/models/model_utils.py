from transformers import  AutoTokenizer
BIN_SIZES = {
    'language_table': -1,
    'calvin': -1
}

def init_tokenizer(base_model_name, bin_sizes=None):
    global BIN_SIZES

    tokenizer =  AutoTokenizer.from_pretrained(base_model_name)
    if bin_sizes is None:
        bin_sizes = BIN_SIZES

    BIN_SIZES = bin_sizes

    language_table_bin_size = bin_sizes['language_table']
    calvin_bin_size = bin_sizes['calvin']

    assert language_table_bin_size >= 0
    assert calvin_bin_size >= 0

    # actions for language table
    for i in range(language_table_bin_size+1):
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
    for i in range(-49, 51):
        camera_actionx = f'[CAMERAX{i}]'
        camera_actiony = f'[CAMERAY{i}]'
        camera_actions.extend([camera_actionx, camera_actiony])

    tokenizer.add_tokens(camera_actions)

    # actions for calvin
    for i in range(calvin_bin_size+1):
        tokenizer.add_tokens([f"[ROBOTACTION0_{i}]", f"[ROBOTACTION1_{i}]", f"[ROBOTACTION2_{i}]",
                                   f"[ROBOTACTION3_{i}]", f"[ROBOTACTION4_{i}]", f"[ROBOTACTION5_{i}]"])

    for i in range(14):
        for j in range(calvin_bin_size+1):
            tokenizer.add_tokens([f"[ROBOTSTATE{i}_{j}]"])

    tokenizer.add_tokens(['[GRIPPER_OPEN]', '[GRIPPER_CLOSE]', '[GRIPPER_OPENED]', '[GRIPPER_CLOSED]'])

    # actions for be
    BE_ACTIONS_LIST = ["Evade", "Jump", "LockOn", "Mount", "MeleeAttack", "SpecialAbility1", "SpecialAbility2", "SpecialAbility3", "SuperAbility", "SwitchLockOnTarget", "Taunt"]

    tokenizer.add_tokens([f'[{x}]' for x in BE_ACTIONS_LIST])
    tokenizer.add_tokens([f'[lrot{rot + 1}]' for rot in range(256)])
    tokenizer.add_tokens([f'[lmag{mag + 1}]' for mag in range(4)])
    tokenizer.add_tokens([f'[rrot{rot + 1}]' for rot in range(256)])
    tokenizer.add_tokens([f'[rmag{mag + 1}]' for mag in range(4)])
    return tokenizer
