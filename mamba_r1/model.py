# !pip install mamba_ssm einops torch loguru

import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from mamba_ssm import Mamba
from safetensors.torch import load_model
from transformers import AutoTokenizer
from zeta.nn import OutputHead

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_position_embeddings).type_as(
            self.inv_freq
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :]
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :]
        )


class SwitchFeedForward(nn.Module):
    """
    Implementation of Switch Transformer's Mixture of Experts layer.
    Based on "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        expert_dim: int = 2048,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True,
        min_expert_capacity: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.expert_dim = expert_dim
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.min_expert_capacity = min_expert_capacity

        # Router
        self.route = nn.Linear(dim, num_experts)

        # Expert feed forward networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, expert_dim),
                    nn.GELU(),
                    nn.Linear(expert_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )

        logger.info(
            f"Initialized Switch FFN with {num_experts} experts"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Calculate routing probabilities
        route_logits = self.route(x)
        routing_weights = F.softmax(route_logits, dim=-1)

        # Find top expert for each token
        selected_expert = routing_weights.argmax(dim=-1)

        # Calculate capacity
        capacity = int(
            self.capacity_factor * seq_len / self.num_experts
        )
        capacity = max(capacity, self.min_expert_capacity)

        # Dispatch tokens to experts
        final_output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            expert_mask = selected_expert == expert_idx
            tokens = x[expert_mask]

            if tokens.shape[0] > 0:
                expert_output = self.experts[expert_idx](tokens)
                final_output[expert_mask] = expert_output

        return final_output


def sample(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(
        dim=-1
    )


@torch.inference_mode()
def generate(
    model: any,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(
        model.max_seq_len, max_new_tokens + max(prompt_lens)
    )
    tokens = torch.full(
        (len(prompt_tokens), total_len),
        -1,
        dtype=torch.long,
        device="cuda",
    )
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(
            t, dtype=torch.long, device="cuda"
        )
    prev_pos = 0
    finished = torch.tensor(
        [False] * len(prompt_tokens), device="cuda"
    )
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(
            prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(
            ~prompt_mask[:, cur_pos], next_token == eos_id
        )
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def load_and_generate(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    *args,
    **kwargs,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    with torch.device("cuda"):
        model = MambaMoE(*args, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1"
    )

    tokenizer.decode(
        generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.0)[0]
    )

    load_model(
        model,
        os.path.join(
            ckpt_path, f"model{rank}-mp{world_size}.safetensors"
        ),
    )

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            completion_tokens = generate(
                model,
                [prompt_tokens],
                max_new_tokens,
                tokenizer.eos_token_id,
                temperature,
            )
            completion = tokenizer.decode(
                completion_tokens[0], skip_special_tokens=True
            )
            print(completion)
            messages.append(
                {"role": "assistant", "content": completion}
            )
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size
        prompt_tokens = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        completion_tokens = generate(
            model,
            prompt_tokens,
            max_new_tokens,
            tokenizer.eos_token_id,
            temperature,
        )
        completions = tokenizer.batch_decode(
            completion_tokens, skip_special_tokens=True
        )
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


class MambaMoE(nn.Module):
    """
    Mamba architecture with Switch Transformer style Mixture of Experts for feedforward networks
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_experts: int = 8,
        expert_dim: int = 2048,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.rotary_emb = RotaryEmbedding(
            dim, max_position_embeddings=max_seq_len
        )
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))

        # Layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Mamba(
                            d_model=dim,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        ),
                        SwitchFeedForward(
                            dim=dim,
                            num_experts=num_experts,
                            expert_dim=expert_dim,
                        ),
                    ]
                )
            )

        logger.info(
            f"Initialized MambaMoE with {depth} layers and {num_experts} experts per layer"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_loss: bool = False,
        correct_answer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, d = x.shape

        # Add positional embeddings
        x = x + self.pos_emb[:, :n]

        # Process through layers
        for mamba, moe in self.layers:
            # Mamba block
            if mask is not None:
                x = x.masked_fill(~mask[..., None], 0)
            mamba_out = mamba(x)
            x = mamba_out + x

            # MoE block
            moe_out = moe(x)
            x = moe_out + x

        if return_logits:
            return x
        elif return_loss:
            return torch.nn.CrossEntropyLoss()(x, correct_answer)
        else:
            return OutputHead(self.dim, vocab_size=self.vocab_size)(x)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> str:
        prompt_tokens = self.tokenizer.encode(prompt)
        completion_tokens = generate(
            self,
            [prompt_tokens],
            max_new_tokens,
            self.tokenizer.eos_token_id,
            temperature,
        )
        return self.tokenizer.decode(
            completion_tokens[0], skip_special_tokens=True
        )


def create_mamba_moe(
    dim: int = 512,
    depth: int = 12,
    d_state: int = 64,
    d_conv: int = 4,
    expand: int = 2,
    num_experts: int = 8,
    expert_dim: int = 2048,
    max_seq_len: int = 2048,
) -> MambaMoE:
    """
    Create a MambaMoE model with the specified parameters.

    Args:
        dim: Model dimension
        depth: Number of layers
        d_state: SSM state expansion factor
        d_conv: Local convolution width
        expand: Block expansion factor
        num_experts: Number of experts in each MoE layer
        expert_dim: Hidden dimension of each expert
        max_seq_len: Maximum sequence length

    Returns:
        MambaMoE: Initialized model
    """
    model = MambaMoE(
        dim=dim,
        depth=depth,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_experts=num_experts,
        expert_dim=expert_dim,
        max_seq_len=max_seq_len,
    )
    return model.to("cuda" if torch.cuda.is_available() else "cpu")
