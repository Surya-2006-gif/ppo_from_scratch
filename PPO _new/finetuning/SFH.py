
from helper.helpers import load_model, tokenize_chat
from datasets import load_dataset
import torch
import random
import gc
from tqdm import tqdm
from pathlib import Path
from main import finetuning_configs,common_configs

random.seed(common_configs["random_seed"])
torch.manual_seed(common_configs["random_seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(common_configs["random_seed"])

device = torch.device(common_configs["device"])

model, tokenizer = load_model(finetuning_configs["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Model loaded on {device}")

# Load paired preference data
dataset = load_dataset(common_configs["dataset_name"])
print("Dataset Loaded!")

train_data = random.sample(
    list(dataset["train"]),
    min(common_configs["train_size"], len(dataset["train"])))

test_data = random.sample(
    list(dataset["test"]),
    min(common_configs["test_size"], len(dataset["test"])))

train_data = [item["chosen"] for item in train_data]
test_data = [item["chosen"] for item in test_data]

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=finetuning_configs["lr"]
)

# Create checkpoint directory if it doesn't exist.
Path("checkpoints").mkdir(parents=True, exist_ok=True)


def run_epoch(texts, training):
    if training:
        model.train()
        random.shuffle(texts)
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    for i in tqdm(range(0, len(texts), finetuning_configs["batch_size"]),leave=False):
        batch_texts = texts[i:i + finetuning_configs["batch_size"]]

        encoded = None
        input_ids = None
        attention_mask = None
        labels = None
        outputs = None
        loss = None

        try:
            encoded = tokenize_chat(batch_texts, tokenizer)

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels = encoded["labels"].to(device)

            # Skip batches without valid assistant targets.
            if not (labels != -100).any():
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss

                if training:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )

                    optimizer.step()

            
            total_loss += loss.detach().item()
            total_batches += 1

        finally:
            # Release gradients and references to batch intermediates.
            if training:
                optimizer.zero_grad(set_to_none=True)

            del encoded, input_ids, attention_mask, labels
            del outputs, loss

            # Collect unreferenced Python objects.
            gc.collect()

            # Release unused CUDA cache blocks.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / max(total_batches, 1)


for epoch in range(finetuning_configs["num_epochs"]):
    train_loss = run_epoch(train_data, training=True)
    val_loss = run_epoch(test_data, training=False)

    print(
        f"Epoch {epoch + 1}/{finetuning_configs['num_epochs']} | "
        f"Train loss: {train_loss:.4f} | "
        f"Validation loss: {val_loss:.4f}"
    )

    checkpoint_path = f"checkpoints/sft_epoch_{epoch + 1}"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

print("SFT completed")

del optimizer, dataset, train_data, test_data
del model, tokenizer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
