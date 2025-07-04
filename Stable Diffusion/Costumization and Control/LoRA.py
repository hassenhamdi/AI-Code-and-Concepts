import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """A LoRA layer that wraps a standard nn.Linear layer."""
    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear_layer = linear_layer  # The original, frozen layer

        self.rank = rank
        self.alpha = alpha

        # Create the low-rank matrices A and B
        self.lora_A = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear_layer.out_features, bias=False)
        
        # Initialize B with zeros, so initially LoRA has no effect
        nn.init.zeros_(self.lora_B.weight)
        
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        # Original, frozen layer output
        original_output = self.linear_layer(x)
        
        # LoRA adaptation
        lora_adaptation = self.lora_B(self.lora_A(x)) * self.scaling
        
        return original_output + lora_adaptation

def apply_lora_to_model(model: nn.Module, rank: int, alpha: float):
    """Recursively finds all nn.Linear layers in a model and replaces them with LoRALayer."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            print(f"Applying LoRA to: {name}")
            setattr(model, name, LoRALayer(module, rank, alpha))
        else:
            apply_lora_to_model(module, rank, alpha)

def freeze_original_weights(model: nn.Module):
    """Freezes all parameters except for the LoRA matrices 'lora_A' and 'lora_B'."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

def print_trainable_parameters(model: nn.Module):
    """Prints the number of trainable parameters in a model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} | All params: {all_params} | Trainable %: {100 * trainable_params / all_params:.2f}%")

# --- Demonstration ---
if __name__ == "__main__":
    # 1. Create a simple example model
    simple_model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 1024), nn.ReLU(),
        nn.Linear(1024, 768)
    )
    print("--- Original Model ---")
    print_trainable_parameters(simple_model)

    # 2. Apply LoRA to the model
    apply_lora_to_model(simple_model, rank=8, alpha=16)
    
    # 3. Freeze the original weights
    freeze_original_weights(simple_model)

    print("\n--- Model with LoRA Applied ---")
    print(simple_model)
    print_trainable_parameters(simple_model)