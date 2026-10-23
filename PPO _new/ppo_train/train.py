
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset

from main import ppo_configs, common_configs
from helper.helpers import (
    load_finetuned_model,
    load_reward_model,
    ValueModel,
    split_into_ppo_turns,
    generate_response,
    get_token_values,
    get_reference_logprobs,
    get_sequence_rewards,
    make_completed_texts,
)



seed = common_configs["random_seed"]
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device(common_configs["device"])

# Frozen reference policy.
ref_model, _ = load_finetuned_model(ppo_configs["ref_model_weights_path"])
ref_model.to(device)
ref_model.eval()

for p in ref_model.parameters():
    p.requires_grad_(False)


# Frozen reward model.
reward_model, _ = load_reward_model(ppo_configs["reward_model_weights_path"])
reward_model.to(device)
reward_model.eval()

for p in reward_model.parameters():
    p.requires_grad_(False)


# Trainable policy, initialized from the SFT checkpoint.
policy_model, tokenizer = load_finetuned_model(ppo_configs["ref_model_weights_path"])
policy_model.to(device)
policy_model.train()


# Trainable critic, initialized from the SFT backbone.
value_backbone, _ = load_finetuned_model(ppo_configs["ref_model_weights_path"])
value_backbone.to(device)

value_model = ValueModel(value_backbone,value_backbone.config.hidden_size,).to(device)
value_model.train()


if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = tokenizer.model_max_length



dataset = load_dataset(common_configs["dataset_name"])

train_rows = random.sample(list(dataset["train"]),min(common_configs["train_size"],len(dataset["train"]),),)

# Expand preference pairs into conversation strings.
train_conversations = ([row["chosen"] for row in train_rows]+ [row["rejected"] for row in train_rows])

del train_rows

all_turns = split_into_ppo_turns(train_conversations)

# Keep only examples that contain a subsequent human turn.
# row[0] = first-turn prefix ending with Assistant:
# row[1] = second human turn ending with Assistant:
ppo_train_data = [(row[0], row[1]) for row in all_turns if row[1] is not None]

del train_conversations, all_turns

random.shuffle(ppo_train_data)

print(f"Second-turn PPO examples: {len(ppo_train_data)}")


# ============================================================
# 4. PPO loss helpers
# ============================================================

def evaluate_response_logprobs(
    model,
    full_ids,
    attention_mask,
    prompt_len,
):
    """
    Differentiable log-probabilities for generated response tokens.
    Uses raw model logits, without generation-time sampling filters.
    """
    out = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=False,)

    # Logits at position t-1 predict token at position t.
    logits = out.logits[:, prompt_len - 1:-1, :].float()
    targets = full_ids[:, prompt_len:]

    log_probs = F.log_softmax(logits, dim=-1)

    return log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)


def compute_post_kl_rewards(
    old_logprobs,
    ref_logprobs,
    action_mask,
    sequence_rewards,
    beta,
):
    """
    Per-token reward = -beta * sampled log-ratio.
    Add the sequence-level reward to the final valid response token.
    """
    mask = action_mask.bool()

    post_kl_rewards = (
        -beta * (old_logprobs - ref_logprobs)
    )
    post_kl_rewards = post_kl_rewards * mask.float()

    lengths = mask.long().sum(dim=1)
    valid_rows = lengths > 0

    if valid_rows.any():
        rows = torch.arange(
            mask.size(0), device=mask.device
        )
        last_positions = (lengths - 1).clamp_min(0)

        post_kl_rewards[
            rows[valid_rows],
            last_positions[valid_rows],
        ] += sequence_rewards[valid_rows]

    return post_kl_rewards


def compute_returns(
    rewards,
    action_mask,
    gamma=1.0,
):
    """
    Monte Carlo returns for a single-response trajectory.
    The last valid response token is terminal.
    """
    B, T = rewards.shape

    returns = torch.zeros_like(rewards)
    running = torch.zeros(
        B,
        dtype=rewards.dtype,
        device=rewards.device,
    )

    lengths = action_mask.long().sum(dim=1)
    positions = torch.arange(
        T, device=rewards.device
    ).unsqueeze(0)

    done_mask = (
        action_mask.bool()
        & (positions == (lengths - 1).unsqueeze(1))
    )

    for t in reversed(range(T)):
        running = (
            rewards[:, t]
            + gamma
            * (1.0 - done_mask[:, t].float())
            * running
        )

        running = running * action_mask[:, t].float()
        returns[:, t] = running

    return returns.detach(), done_mask.float()


def compute_value_loss(
    values,
    old_values,
    returns,
    action_mask,
    epsilon_v=0.2,
):
    v_clipped = old_values + torch.clamp(
        values - old_values,
        min=-epsilon_v,
        max=epsilon_v,
    )

    loss_unclipped = 0.5 * (values - returns).pow(2)
    loss_clipped = 0.5 * (v_clipped - returns).pow(2)

    token_loss = torch.maximum(
        loss_unclipped, loss_clipped
    )

    mask = action_mask.float()

    return (
        (token_loss * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1)
    ).mean()


def compute_policy_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    action_mask,
    clip_eps=0.2,
):
    # PPO importance ratio.
    ratio = torch.exp(new_logprobs - old_logprobs)

    objective_unclipped = ratio * advantages
    objective_clipped = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    token_loss = -torch.minimum(
        objective_unclipped,
        objective_clipped,
    )

    mask = action_mask.float()

    return (
        (token_loss * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1)
    ).mean()


# ============================================================
# 5. Optimizers and PPO hyperparameters
# ============================================================

policy_optimizer = torch.optim.AdamW(
    policy_model.parameters(),lr=ppo_configs.get("policy_lr", 1e-5),)

value_optimizer = torch.optim.AdamW(
    value_model.parameters(),lr=ppo_configs.get("value_lr", 1e-5),)

rpr = ppo_configs["rpr"]
mini_batch_size = ppo_configs["mini_batch_size"]
epochs_per_rollout = ppo_configs["epr"]

beta = ppo_configs.get("kl_beta", 0.05)
gamma = ppo_configs.get("gamma", 1.0)
clip_eps = ppo_configs.get("clip_eps", 0.2)
epsilon_v = ppo_configs.get("epsilon_v", 0.2)
vf_coef = ppo_configs.get("vf_coef", 0.5)
max_new_tokens = ppo_configs.get("max_new_tokens", 1024)


# ============================================================
# 6. PPO training loop -- SECOND TURN ONLY
# ============================================================

for rollout_start in range(0, len(ppo_train_data), rpr):

    batch_rows = ppo_train_data[
        rollout_start:rollout_start + rpr
    ]

    prompts1 = [row[0] for row in batch_rows]
    prompts2 = [row[1] for row in batch_rows]

    # --------------------------------------------------------
    # A. Generate first response to construct second-turn context
    # --------------------------------------------------------

    turn1 = generate_response(
        prompts1,
        policy_model,
        tokenizer,
        device,
        max_prompt_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    responses1 = []

    for ids, mask in zip(
        turn1["response_ids"],
        turn1["response_mask"],
    ):
        responses1.append(
            tokenizer.decode(
                ids[mask.bool()],
                skip_special_tokens=False,
            )
        )

    # Construct the complete context for the second response.
    prompts2_full = [
        prompt1 + response1 + next_prompt
        for prompt1, response1, next_prompt in zip(
            prompts1,
            responses1,
            prompts2,
        )
    ]

    # --------------------------------------------------------
    # B. Generate second response -- THIS IS THE PPO ACTION
    # --------------------------------------------------------

    turn2 = generate_response(
        prompts2_full,
        policy_model,
        tokenizer,
        device,
        max_prompt_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    # Freeze the sampled action data for this PPO rollout.
    old_logprobs = turn2["old_logprobs"].detach()
    action_mask = turn2["response_mask"].bool()

    full_ids = turn2["full_ids"].detach()
    full_attention_mask = (
        turn2["full_attention_mask"].detach()
    )
    prompt_len = turn2["prompt_ids"].shape[1]

    # --------------------------------------------------------
    # C. Reference log-probabilities on the same sampled tokens
    # --------------------------------------------------------

    ref_logprobs = get_reference_logprobs(
        turn2, ref_model
    ).detach()

    # --------------------------------------------------------
    # D. Score the completed second-turn conversation
    # --------------------------------------------------------

    completed_texts = make_completed_texts(
        prompts2_full,
        turn2,
        tokenizer,
    )

    sequence_rewards = get_sequence_rewards(
        completed_texts,
        reward_model,
        tokenizer,
        device,
        max_length,
    ).detach()

    # --------------------------------------------------------
    # E. Old critic values, evaluated before any PPO update
    # --------------------------------------------------------

    with torch.no_grad():
        old_values = get_token_values(
            value_model,
            full_ids,
            full_attention_mask,
            prompt_len,
        ).detach()

    # --------------------------------------------------------
    # F. Post-KL rewards, returns, and advantages
    # --------------------------------------------------------

    post_kl_rewards = compute_post_kl_rewards(
        old_logprobs,
        ref_logprobs,
        action_mask,
        sequence_rewards,
        beta,
    )

    returns, done_mask = compute_returns(
        post_kl_rewards,
        action_mask,
        gamma,
    )

    advantages = (returns - old_values).detach()

    # Normalize over valid action tokens in this rollout.
    valid_advantages = advantages[action_mask]

    if valid_advantages.numel() > 1:
        mean_adv = valid_advantages.mean()
        std_adv = valid_advantages.std(unbiased=False)

        advantages = (
            (advantages - mean_adv)
            / (std_adv + 1e-8)
        )

    advantages = advantages * action_mask.float()
    advantages = advantages.detach()

    # --------------------------------------------------------
    # G. Multiple PPO optimization epochs on this fixed rollout
    # --------------------------------------------------------

    B = full_ids.size(0)

    for epoch in range(epochs_per_rollout):

        permutation = torch.randperm(B, device=device)

        for mb_start in range(0, B, mini_batch_size):

            indices = permutation[
                mb_start:mb_start + mini_batch_size
            ]

            mb_full_ids = full_ids[indices]
            mb_attention = full_attention_mask[indices]

            mb_old_logprobs = old_logprobs[indices]
            mb_ref_logprobs = ref_logprobs[indices]
            mb_old_values = old_values[indices]
            mb_returns = returns[indices]
            mb_advantages = advantages[indices]
            mb_mask = action_mask[indices]

            # Current policy log-probabilities.
            new_logprobs = evaluate_response_logprobs(
                policy_model,
                mb_full_ids,
                mb_attention,
                prompt_len,
            )

            # Current critic predictions.
            new_values = get_token_values(
                value_model,
                mb_full_ids,
                mb_attention,
                prompt_len,
            )

            # Clipped PPO policy loss.
            policy_loss = compute_policy_loss(
                new_logprobs,
                mb_old_logprobs,
                mb_advantages,
                mb_mask,
                clip_eps,
            )

            # Clipped critic loss.
            value_loss = compute_value_loss(
                new_values,
                mb_old_values,
                mb_returns,
                mb_mask,
                epsilon_v,
            )

            total_loss = policy_loss + vf_coef * value_loss

            policy_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), 1.0
            )
            torch.nn.utils.clip_grad_norm_(
                value_model.parameters(), 1.0
            )

            policy_optimizer.step()
            value_optimizer.step()

            print(
                f"rollout={rollout_start // rpr} "
                f"epoch={epoch + 1}/{epochs_per_rollout} "
                f"policy_loss={policy_loss.item():.4f} "
                f"value_loss={value_loss.item():.4f} "
                f"reward={sequence_rewards.mean().item():.4f} "
                f"valid_tokens={int(mb_mask.sum().item())}"
            )

    # End of this rollout: the policy has been updated.
    # The next iteration samples new second-turn responses.

print("PPO training complete.")
