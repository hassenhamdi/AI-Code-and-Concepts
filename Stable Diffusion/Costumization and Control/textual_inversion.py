import torch
import torch.nn.functional as F
from typing import List

# Placeholder dummy models for demonstration.
class DummyUNet(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x, timestep, context): return self.conv(x)

class DummyVAE(torch.nn.Module):
    def encode(self, x): return x
    def decode(self, x): return x

class DummyTokenizer:
    def __init__(self, vocab_size=49408):
        self.vocab = {f"<word_{i}>": i for i in range(vocab_size)}
        self.vocab_size = vocab_size
    def add_tokens(self, new_token: str):
        if new_token not in self.vocab:
            new_id = self.vocab_size
            self.vocab[new_token] = new_id
            self.vocab_size += 1
            print(f"Added token '{new_token}' with ID {new_id}")
    def __call__(self, text: str):
        return torch.tensor([[self.vocab.get(t, 0) for t in text.split()]], dtype=torch.long)

class DummyTextEncoder(torch.nn.Module):
    def __init__(self, vocab_size=49408, embed_dim=768):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
    def forward(self, x): return self.token_embedding(x)

def train_textual_inversion(images: List[torch.Tensor], placeholder_token: str, unet, vae, text_encoder, tokenizer, learning_rate=5e-4, num_steps=1000):
    """Trains a new concept embedding via Textual Inversion."""
    print(f"--- Starting Textual Inversion Training for '{placeholder_token}' ---")
    
    # 1. Add the new placeholder token and resize the embedding matrix
    tokenizer.add_tokens(placeholder_token)
    placeholder_id = tokenizer.vocab[placeholder_token]
    old_embeddings = text_encoder.token_embedding.weight.data
    text_encoder.token_embedding = torch.nn.Embedding(tokenizer.vocab_size, old_embeddings.shape[1])
    text_encoder.token_embedding.weight.data[:old_embeddings.shape[0]] = old_embeddings

    # 2. Freeze all model parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 3. Unfreeze ONLY the new token's embedding for optimization
    params_to_optimize = [text_encoder.token_embedding.weight]
    text_encoder.token_embedding.weight.requires_grad = True
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    # --- The Training Loop ---
    for step in range(num_steps):
        image = images[torch.randint(0, len(images), (1,)).item()]
        
        with torch.no_grad():
            latents = vae.encode(image.unsqueeze(0))
            timestep = torch.randint(0, 1000, (1,)).long()
            noise = torch.randn_like(latents)
            noisy_latents = latents + noise # Simplified noise schedule

        prompt = f"a photo of {placeholder_token}"
        text_ids = tokenizer(prompt)
        context = text_encoder(text_ids)
        
        predicted_noise = unet(noisy_latents, timestep, context)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()

        # Zero out gradients for all embeddings except our new one
        with torch.no_grad():
            grad_mask = torch.zeros_like(text_encoder.token_embedding.weight.grad)
            grad_mask[placeholder_id] = 1
            text_encoder.token_embedding.weight.grad *= grad_mask

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")
            
    learned_embedding = text_encoder.token_embedding.weight[placeholder_id].detach().clone()
    print("--- Training Finished ---")
    return learned_embedding

if __name__ == "__main__":
    unet_model, vae_model, text_encoder_model, tokenizer_model = DummyUNet(), DummyVAE(), DummyTextEncoder(), DummyTokenizer()
    concept_images = [torch.randn(3, 64, 64) for _ in range(5)]
    concept_token = "<my-unique-cat>"
    learned_vector = train_textual_inversion(concept_images, concept_token, unet_model, vae_model, text_encoder_model, tokenizer_model)
    print(f"\nSuccessfully learned an embedding for '{concept_token}' with shape: {learned_vector.shape}")