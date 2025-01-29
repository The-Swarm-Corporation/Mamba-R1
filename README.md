# Mamba R1

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)



Mamba R1 represents a novel architecture that combines the efficiency of Mamba's state space models with the scalability of Mixture of Experts (MoE). This implementation achieves superior performance through reduced memory usage and increased token processing capabilities compared to traditional transformer architectures, while maintaining model quality through selective expert routing.

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

## Architecture Overview

The architecture consists of alternating Mamba blocks and Mixture of Experts feed-forward networks (MoE FFN), allowing for efficient sequence processing while maintaining model capacity through specialized expert networks.

### Key Components

1. **Mamba Blocks**
   - Utilizes selective state space modeling for efficient sequence processing
   - Features local convolution with width d_conv
   - Implements state space models with d_state dimensionality
   - Linear projections for dimensional control

2. **Mixture of Experts (MoE)**
   - Implements Switch Transformer style routing
   - Supports dynamic expert selection
   - Features capacity factor for load balancing
   - Maintains minimum expert capacity for stability

3. **Positional Encoding**
   - Rotary embeddings for position-aware processing
   - Learnable positional parameters
   - Maximum sequence length of 2048 tokens

## Advantages Over Traditional Transformers

### Memory Efficiency
- Linear memory scaling with sequence length (O(n)) vs quadratic scaling in transformers (O(n²))
- Reduced attention computation overhead
- Efficient state-based processing without storing full attention matrices

### Processing Speed
- Faster token processing due to linear complexity
- Reduced computational bottlenecks
- Parallel expert processing for increased throughput

### Scalability
- Linear scaling of computational resources
- Efficient handling of long sequences
- Dynamic expert routing for specialized processing

## Technical Specifications

```python
Architecture Parameters:
- Dimension (dim): 512
- Depth: 12
- State Dimension (d_state): 64
- Convolution Width (d_conv): 4
- Expansion Factor: 2
- Number of Experts: 8
- Expert Dimension: 2048
- Maximum Sequence Length: 2048
```

## Implementation Details

### State Space Model Integration
The Mamba component implements selective state space modeling through:
1. Local convolution processing
2. State space transformation
3. Linear projections
4. Residual connections

### Expert Routing Mechanism
The MoE component features:
1. Token-based routing
2. Load balancing through capacity factor
3. Minimal expert capacity guarantee
4. Efficient expert selection

## Performance Characteristics

### Memory Usage
- Linear memory scaling with sequence length
- Efficient state maintenance
- Reduced attention matrix storage

### Computational Efficiency
- O(n) complexity for sequence processing
- Parallel expert computation
- Efficient state updates

### Scaling Properties
- Linear scaling with sequence length
- Efficient multi-GPU distribution
- Dynamic load balancing


## Citation

```bibtex
@article{mamba_moe2024,
    title={MambaMoE: Efficient State Space Models with Mixture of Experts},
    year={2024},
    author={[Kye Gomez]},
    journal={[Swarms]},
    volume={[Volume]},
    pages={[Pages]}
}
```

# License
MIT
