
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from helper.helpers import load_finetuned_model
from helper.helpers import RewardModel
from main import reward_configs,common_configs
import os

random.seed(common_configs["random_seed"])
torch.manual_seed(common_configs["random_seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SFT model and tokenizer
backbone, tokenizer = load_finetuned_model(reward_configs["model_weights_path"])

max_length = tokenizer.model_max_length
backbone.config.pad_token_id = tokenizer.pad_token_id
backbone.config.use_cache = False

# Import RewardModel from the module where you defined it.
# Alternatively, keep the LinearHead and RewardModel classes above this code.
hidden_size = backbone.config.hidden_size
reward_model = RewardModel(backbone, hidden_size).to(device)

print("Reward model initialized!")

# Load paired preference data
dataset = load_dataset(common_configs["dataset_name"])

train_data = random.sample(list(dataset["train"]),min(common_configs["train_size"], len(dataset["train"])))

test_data = random.sample(list(dataset["test"]),min(common_configs["test_size"], len(dataset["test"])))


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

reward_model.backbone.config.pad_token_id = tokenizer.pad_token_id


def score_responses(texts):
    """Return one scalar reward per conversation."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",)

    encoded = {k: v.to(device) for k, v in encoded.items()}

    return reward_model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],)


def run_epoch(pairs, training):
    reward_model.train(training)

    random.shuffle(pairs) if training else None

    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, len(pairs), reward_configs["batch_size"]):
        batch = pairs[start:start + reward_configs["batch_size"]]

        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]

        # Process both sets of responses.
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            chosen_scores = score_responses(chosen)
            rejected_scores = score_responses(rejected)

            # Bradley-Terry Model
            loss = -F.logsigmoid(
                chosen_scores - rejected_scores
            ).mean()

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    reward_model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        total_loss += loss.detach().item()
        correct += (
            chosen_scores.detach() > rejected_scores.detach()
        ).sum().item()
        total += len(batch)

        # Release batch intermediates
        del chosen_scores, rejected_scores, loss

    return total_loss / max(1, (len(pairs) + reward_configs["batch_size"] - 1)
                            // reward_configs["batch_size"]), correct / max(total, 1)


optimizer = torch.optim.AdamW(
    reward_model.parameters(),
    lr=reward_configs["lr"],
)

for epoch in range(reward_configs["num_epochs"]):
    train_loss, train_acc = run_epoch(train_data, training=True)
    val_loss, val_acc = run_epoch(test_data, training=False)

    print(
        f"Epoch {epoch + 1}/{reward_configs['num_epochs']} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_acc:.4f}"
    )

# Save backbone and reward head separately
save_path = "checkpoints/reward_model"
os.makedirs(save_path, exist_ok=True)

reward_model.backbone.save_pretrained(
    os.path.join(save_path, "backbone")
)
tokenizer.save_pretrained(os.path.join(save_path, "backbone"))
torch.save(
    reward_model.reward_head.state_dict(),
    os.path.join(save_path, "reward_head.pt"),
)

print("Reward model training completed!")
