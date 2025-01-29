# Mamba R1

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


## Installation

```bash
pip install mamba-r1
```

## Usage

```python
from loguru import logger
import torch

from mamba_r1.model import create_mamba_moe


def main():
    # Set up logging
    logger.add("mamba_moe.log", rotation="500 MB")

    # Model parameters
    batch_size = 4
    seq_length = 512
    dim = 256  # Smaller dimension for example purposes

    # Create model with smaller parameters for testing
    logger.info("Initializing MambaMoE model...")
    model = create_mamba_moe(
        dim=dim,
        depth=6,  # 6 layers for this example
        d_state=16,  # Smaller state dimension
        d_conv=2,  # Smaller conv dimension
        expand=1,  # No expansion
        num_experts=4,  # 4 experts per layer
        expert_dim=512,  # Smaller expert dimension
        max_seq_len=2048,
    )

    device = torch.device("cuda:0")
    logger.info(f"Model device: {device}")

    # Create example input (batch_size, seq_length, dim)
    logger.info("Creating example input...")
    x = torch.randn(batch_size, seq_length, dim).to(device)

    # Optional attention mask (1 = attend, 0 = ignore)
    mask = torch.ones(batch_size, seq_length).bool().to(device)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    logger.info("Running forward pass...")
    with torch.no_grad():
        output = model(x, mask=mask)

    # Print output shape and statistics
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output mean: {output.mean().item():.4f}")
    logger.info(f"Output std: {output.std().item():.4f}")

    return output


if __name__ == "__main__":
    try:
        output = main()
        logger.success("Successfully ran MambaMoE forward pass!")
    except Exception as e:
        logger.exception(f"Error running MambaMoE: {str(e)}")
        raise

```

# License
MIT
