from helper.helpers import load_model
import torch
import subprocess
import sys

common_configs = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset_name": "Anthropic/hh-rlhf",
    "random_seed": 42,
    "train_size": 10000,
    "test_size": 2500,}


finetuning_configs  = {
    "lr": 1e-5,
    "batch_size": 2,
    "num_epochs": 3,
    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",}


reward_configs = {
    "model_weights_path": "checkpoints/sft_epoch_1",
    "num_epochs": 4,
    "lr": 1e-5,
    "batch_size": 2,}

ppo_configs = {
    'gamma': 1,
    'lambda': 0.95,
    'kl_beta': 0.05,
    'val_weight':1,
    'num_epochs':5,
    'value_lr':1e-5,
    'policy_lr':1e-6,
    "ref_model_weights_path": "checkpoints/sft_epoch_1",
    "reward_model_weights_path": "checkpoints/reward_model",
    "rpr":2,      #responses per rollout
    "epr":4,       #epoch per rollout
    "mini_batch_size":2,
    "clip_eps":0.2,
    "epsilon_v":0.2,
    "vf_coef":0.5,
    "max_new_tokens":1024

}




Train_options = [
    "FineTune Only", 
    "Reward Only", 
    "FineTune + Reward", 
    "FineTune + Reward + PPO"
]

def select_training_mode(options):
    print("-" * 100)
    print("TRAINING PIPELINE CONFIGURATION")
    print("-" * 100)
    for idx, option in enumerate(options, 1):
        print(f"  [{idx}] {option}")
    print("-" * 100)

    while True:
        try:
            choice = input(f"Select an option (1-{len(options)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(options):
                selected_option = options[choice_idx]
                print(f"Selected: {selected_option}\n")
                return selected_option
            else:
                print(f"Error: Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Error: Invalid input. Please enter a valid number.")

if __name__ == "__main__":

    model,tokenizer = load_model(finetuning_configs["model_name"])
    print("-" * 100)
    print("Pre-eliminary checks")
    print("-" * 100)
    if tokenizer.pad_token is None:
        print("No default padding token defined in tokenizer")
    else:
        print(f"Default padding token of tokenizer: {tokenizer.pad_token}")

    if model.config.pad_token_id is None:
        print("No default padding token defined in model config")
    else:
        print(f"Default padding token of model config: {tokenizer.decode(model.config.pad_token_id)}")
    print(f"Model max length:{tokenizer.model_max_length}")
    print(f"Model padding side: {tokenizer.padding_side}")
    print(f"Tokenizer eos token:{tokenizer.eos_token}")

    del model,tokenizer
    torch.cuda.empty_cache()
    selected_mode = select_training_mode(Train_options)


    if selected_mode == "FineTune Only":
        subprocess.run([sys.executable, "-m", "finetuning.SFH"], check=True)

    elif selected_mode == "Reward Only":
        subprocess.run([sys.executable, "-m", "reward_model.reward_model"], check=True)

    elif selected_mode == "FineTune + Reward":
        print("▶ Running Supervised Fine-Tuning...")
        subprocess.run([sys.executable, "-m", "finetuning.SFH"], check=True)
        
        print("\n▶ Running Reward Model Training...")
        subprocess.run([sys.executable, "-m", "reward_model.reward_model"], check=True)

    elif selected_mode == "FineTune + Reward + PPO":
        print("▶ Running Supervised Fine-Tuning...")
        subprocess.run([sys.executable, "-m", "finetuning.SFH"], check=True)
        
        print("\n▶ Running Reward Model Training...")
        subprocess.run([sys.executable, "-m", "reward_model.reward_model"], check=True)
        
        # print("\n▶ Running PPO Training...")
        # subprocess.run([sys.executable, "-m", "ppo.ppo_training"], check=True)


