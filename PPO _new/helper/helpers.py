import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch.nn as nn
import torch.nn.functional as F
import os

#----------------- Loading ------------------------------------------------------------------ 

def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    return model,tokenizer



def load_finetuned_model(model_weights_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_weights_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_weights_path
    ).to(device)

    return model,tokenizer

def load_reward_model(model_weights_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone,tokenizer = load_finetuned_model(
    os.path.join(model_weights_path,"backbone"))
    reward_model = RewardModel(backbone,backbone.config.hidden_size).to(device)

    state_dict = torch.load(os.path.join(model_weights_path,"reward_head.pt"), map_location=device)
    reward_model.reward_head.load_state_dict(state_dict)

    return reward_model,tokenizer

#---------------------chat and tokens----------------------------------------------------------------------
def tokenize_chat(texts, tokenizer):
    """
    texts: list of HH-RLHF conversation strings.

    Returns:
        input_ids
        attention_mask
        labels: assistant tokens + EOS retained;
                human tokens and padding masked.
    """

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token = tokenizer.eos_token

    # Append EOS to the end of every assistant turn.
    processed_texts = []

    for text in texts:
        # Match each Assistant turn up to the next Human turn or end.
        pattern = r"(Assistant:\s*)(.*?)(?=\n\nHuman:|$)"

        def add_eos(match):
            prefix = match.group(1)
            response = match.group(2)

            # Avoid adding EOS twice.
            response = response.rstrip()
            if not response.endswith(eos_token):
                response += eos_token

            return prefix + response

        text = re.sub(pattern, add_eos, text, flags=re.DOTALL)
        processed_texts.append(text)

    max_length = tokenizer.model_max_length

    encoded = tokenizer(
        processed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]

    labels = torch.full_like(input_ids, -100)

    for b, text in enumerate(processed_texts):
        matches = list(
            re.finditer(
                r"(?:^|\n\n)(Human|Assistant):",
                text
            )
        )

        assistant_spans = []

        for i, match in enumerate(matches):
            if match.group(1) != "Assistant":
                continue

            # Exclude the preceding newline, if any, from the span.
            start = match.start(1)

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            assistant_spans.append((start, end))

        for j, (start, end) in enumerate(offsets[b].tolist()):
            if attention_mask[b, j].item() == 0:
                continue

            if start == end:
                continue

            # Keep assistant response tokens and its EOS token.
            for span_start, span_end in assistant_spans:
                if start < span_end and end > span_start:
                    labels[b, j] = input_ids[b, j]
                    break

    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }




def split_into_ppo_turns(data):
    """
    Returns samples of shape (n_samples, 2).

    Each sample:
        [
            conversation_prefix_ending_with_Assistant:,
            next_human_turn_ending_with_Assistant:
        ]

    The first element is the context for generating an assistant response.
    The second element is the next human turn formatted as a prompt.
    If no next human turn exists, the second element is None.
    """
    samples = []

    for conversation in data:
        # Normalize line endings.
        conversation = conversation.replace("\r\n", "\n").replace("\r", "\n")

        # Find all turn markers.
        matches = list(
            re.finditer(r"(?:^|\n\n)(Human|Assistant):", conversation)
        )

        for i, match in enumerate(matches):
            if match.group(1) != "Assistant":
                continue

            # Prefix ending immediately after Assistant:
            prefix = conversation[:match.end()].strip()
            prefix = "\n\n" + prefix.lstrip("\n")
            
            # Find the next Human turn.
            next_human_idx = None
            for j in range(i + 1, len(matches)):
                if matches[j].group(1) == "Human":
                    next_human_idx = j
                    break

            next_prompt = None

            if next_human_idx is not None:
                human_match = matches[next_human_idx]

                # Find the next Assistant marker after this Human turn.
                next_assistant_idx = None
                for j in range(next_human_idx + 1, len(matches)):
                    if matches[j].group(1) == "Assistant":
                        next_assistant_idx = j
                        break

                if next_assistant_idx is not None:
                    assistant_match = matches[next_assistant_idx]

                    # Include Human: and its message, then Assistant:
                    next_prompt = (
                        "\n\n"
                        + conversation[human_match.start():assistant_match.end()].strip()
                    )
                else:
                    # Human turn exists but has no following Assistant marker.
                    next_prompt = (
                        "\n\n" + conversation[human_match.start():].strip()
                        + "\n\nAssistant:"
                    )

            samples.append([prefix, next_prompt])

    return samples

def tokenize_prompt_with_completion_mask(prompts, tokenizer, device, max_length):
    """
    completion_mask:
        0 = human text / padding
        1 = assistant text, including Assistant: markers

    Truncation can remove earlier conversation history.
    """
    tokenizer.padding_side = "left"

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"]

    completion_mask = torch.zeros_like(input_ids)

    for b, text in enumerate(prompts):
        # Each assistant span includes "Assistant:" and its response,
        # up to (but excluding) the next Human: marker.
        matches = list(re.finditer(
            r"(?:^|\n\n)(Human|Assistant):",
            text
        ))

        spans = []
        for i, match in enumerate(matches):
            if match.group(1) != "Assistant":
                continue

            start = match.start(1)
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )
            spans.append((start, end))

        for j, (start, end) in enumerate(offsets[b].tolist()):
            if attention_mask[b, j].item() == 0 or start == end:
                continue

            if any(start < span_end and end > span_start
                   for span_start, span_end in spans):
                completion_mask[b, j] = 1

    return input_ids, attention_mask, completion_mask


def raw_token_logprobs(model, input_ids, attention_mask, response_start):
    """
    Calculate raw model log-probabilities for response tokens.
    response_start is the first response-token position in input_ids.

    Returns log P(token_t | tokens_before_t) for each response token.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    # Logits at position t-1 predict token at position t.
    response_ids = input_ids[:, response_start:]
    response_logits = outputs.logits[
        :, response_start - 1 : -1, :
    ]

    log_probs = F.log_softmax(response_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(
        dim=-1, index=response_ids.unsqueeze(-1)
    ).squeeze(-1)

    return token_logprobs


def generate_response(
    prompts, model, tokenizer, device,
    max_prompt_length=1024, max_new_tokens=128
):
    """
    Returns:
        prompt_ids, prompt_attention, prompt_completion_mask,
        response_ids, raw_response_logprobs, response_mask

    response_mask includes EOS, excludes tokens after EOS.
    """
    model.eval()
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids, prompt_attention, prompt_completion_mask = (
        tokenize_prompt_with_completion_mask(
            prompts, tokenizer, device, max_prompt_length
        )
    )

    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        response_ids = generated[:, prompt_len:]
        B, T = response_ids.shape

        # Include first EOS; exclude all later generated padding/tokens.
        response_mask = torch.ones(
            (B, T), dtype=torch.bool, device=device
        )

        if tokenizer.eos_token_id is not None and T > 0:
            eos_hits = response_ids.eq(tokenizer.eos_token_id)
            has_eos = eos_hits.any(dim=1)
            first_eos = eos_hits.int().argmax(dim=1)
            positions = torch.arange(T, device=device)[None, :]

            response_mask = (
                ~has_eos[:, None]
            ) | (positions <= first_eos[:, None])

        # Attention over prompt + generated response.
        full_attention = torch.cat(
            [prompt_attention, response_mask.long()], dim=1
        )

        # Recompute log-probabilities from raw forward-pass logits.
        full_ids = generated
        old_logprobs = raw_token_logprobs(
            model, full_ids, full_attention, prompt_len
        )

        old_logprobs = old_logprobs.masked_fill(
            ~response_mask, 0.0
        )

    # The generated response tokens are PPO actions.
    action_mask = response_mask.long()

    # Full-sequence completion mask:
    # assistant tokens in the prompt/history + generated response.
    full_completion_mask = torch.cat(
        [prompt_completion_mask, action_mask], dim=1
    )

    return {
        "prompt_ids": prompt_ids,
        "prompt_attention_mask": prompt_attention,
        "prompt_completion_mask": prompt_completion_mask,
        "response_ids": response_ids,
        "old_logprobs": old_logprobs,
        "response_mask": response_mask.long(),
        "full_ids": full_ids,
        "full_attention_mask": full_attention,
        "completion_mask": full_completion_mask,
        "action_mask": action_mask,
    }

def decode_responses(rollout, tokenizer):
    responses = []
    for ids, mask in zip(
        rollout["response_ids"], rollout["response_mask"]
    ):
        responses.append(
            tokenizer.decode(
                ids[mask.bool()], skip_special_tokens=False
            )
        )
    return responses


def make_completed_texts(prompts, rollout, tokenizer):
    responses = decode_responses(rollout, tokenizer)
    return [
        prompt + response
        for prompt, response in zip(prompts, responses)
    ]
#-----------------------------------prob ,value,reward models------------------------------------------------------

@torch.no_grad()
def get_reference_logprobs(rollout, ref_model):
    """
    Reference log-probabilities for the exact sampled tokens.
    Uses raw forward-pass logits, not generate() scores.
    """
    return raw_token_logprobs(
        ref_model,
        rollout["full_ids"],
        rollout["full_attention_mask"],
        rollout["prompt_ids"].shape[1],
    ).masked_fill(
        ~rollout["response_mask"].bool(), 0.0
    )


@torch.no_grad()
def get_sequence_rewards(texts, reward_model, tokenizer, device,
                         max_length):
    """
    One scalar reward per completed conversation.
    Right padding is used because the provided RewardModel
    computes last_indices = attention_mask.sum(1) - 1.
    """
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    ).to(device)

    scores = reward_model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    ).reshape(-1)

    tokenizer.padding_side = old_padding_side
    return scores






@torch.no_grad()
def get_token_values(value_model, full_ids, attention_mask, prompt_len):
    """
    Expected ValueModel interface:
      value_model.backbone(...) returns last hidden states
      value_model.value_head(hidden_states) returns [B, L, 1]
    """
    outputs = value_model.backbone(
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    hidden = outputs.last_hidden_state
    token_values = value_model.value_head(hidden).squeeze(-1)

    # Value at each response token is the value of the state
    # immediately before predicting that token.
    return token_values[:, prompt_len - 1 : -1]




class LinearHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class RewardModel(nn.Module):
    def __init__(self, model, hidden_size):
        super().__init__()
        self.backbone = model
        self.reward_head = LinearHead(hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Final-layer representation of every token
        hidden_states = outputs.hidden_states[-1]

        # Index of the last non-padding token in each sequence
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            input_ids.size(0), device=input_ids.device
        )

        last_hidden = hidden_states[batch_indices, last_indices]
        scores = self.reward_head(last_hidden)

        return scores

class ValueModel(nn.Module):
    def __init__(self, model, hidden_size):
        super().__init__()
        self.backbone = model
        self.value_head = LinearHead(hidden_size) # You can reuse your LinearHead class

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states) # Returns scalar value per token
        return values

