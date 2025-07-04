import torch
import torch.nn.functional as F

# Placeholder dummy models for demonstration.
class DummyUNet(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x, timestep, context): return self.conv(x) + 0.1 * context.mean() # Simulate context influence

class DummyVAE(torch.nn.Module):
    def decode(self, x): return torch.randn(1, 3, 256, 256) # Return a dummy image

class DummyTextEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn(1, 77, embed_dim))
    def forward(self, x): return self.embedding.expand(x.shape[0], -1, -1)

class DummyScheduler:
    def __init__(self, num_steps=20):
        self.timesteps = torch.arange(999, 0, -(1000 // num_steps))
    def step(self, model_output, timestep, sample):
        # Simplified DDIM-like step
        alpha_prod_t = 1.0 - (timestep / 1000.0) ** 2
        alpha_prod_t_prev = 1.0 - ((timestep - 50) / 1000.0) ** 2
        pred_original_sample = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        return alpha_prod_t_prev ** 0.5 * pred_original_sample + (1 - alpha_prod_t_prev) ** 0.5 * model_output

def sample_with_cfg(
    unet, vae, text_encoder, scheduler,
    positive_prompt_ids, negative_prompt_ids,
    guidance_scale=7.5,
    latent_shape=(1, 4, 32, 32)
):
    """
    Generates an image using a diffusion model with Classifier-Free Guidance.
    """
    print(f"--- Starting Sampling with CFG (Guidance Scale: {guidance_scale}) ---")
    
    # 1. Prepare initial noisy latent and context embeddings
    latents = torch.randn(latent_shape)
    with torch.no_grad():
        conditional_context = text_encoder(positive_prompt_ids)
        unconditional_context = text_encoder(negative_prompt_ids)
    
    # 2. The Denoising Loop
    for timestep in scheduler.timesteps:
        print(f"Denoising at timestep {timestep.item()}...")
        
        # For CFG, we run the U-Net twice: once with text-conditioning and once without.
        # This is done efficiently in a single batch.
        latent_model_input = torch.cat([latents] * 2)
        context_input = torch.cat([unconditional_context, conditional_context])
        
        with torch.no_grad():
            noise_pred_batch = unet(latent_model_input, timestep, context_input)
            
        # Split the predictions
        unconditional_noise, conditional_noise = noise_pred_batch.chunk(2)
        
        # 3. Apply the Classifier-Free Guidance formula
        # This "guides" the denoising process towards the text prompt.
        guided_noise = unconditional_noise + guidance_scale * (conditional_noise - unconditional_noise)
        
        # 4. Update the latents using the scheduler's step function
        latents = scheduler.step(guided_noise, timestep, latents)

    print("--- Denoising Complete ---")
    
    # 5. Decode the final latent into an image
    with torch.no_grad():
        image = vae.decode(latents)
        
    return image

if __name__ == "__main__":
    unet_model, vae_model, text_encoder_model = DummyUNet(), DummyVAE(), DummyTextEncoder()
    scheduler = DummyScheduler(num_steps=20)
    
    # Simulate tokenized prompts
    pos_prompt_ids = torch.ones((1, 77), dtype=torch.long)
    neg_prompt_ids = torch.zeros((1, 77), dtype=torch.long)
    
    # Run the sampler
    final_image = sample_with_cfg(unet_model, vae_model, text_encoder_model, scheduler, pos_prompt_ids, neg_prompt_ids)
    print(f"\nSuccessfully generated a final image tensor of shape: {final_image.shape}")