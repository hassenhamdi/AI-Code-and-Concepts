import torch
from tqdm import trange

# --- Helper Functions ---

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma.view(-1, 1, 1, 1)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level for an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

# --- Sampler Implementations ---

@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, eta=1.0):
    """Ancestral sampling with Euler method steps."""
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
            
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        
        if sigmas[i + 1] > 0:
            x = x + torch.randn_like(x) * sigma_up
            
    return x

@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
            
        dt = sigmas[i + 1] - sigma_hat
        
        if sigmas[i + 1] == 0: # Last step is Euler
            x = x + d * dt
        else: # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
            
    return x

@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None):
    """DPM-Solver++(2M)."""
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    
    def sigma_fn(t): return t.neg().exp()
    def t_fn(sigma): return sigma.log().neg()
    
    old_denoised = None

    for i in trange(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
            
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            
        old_denoised = denoised
        
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, eta=1.0):
    """DPM-Solver++(2M) SDE."""
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
        
        if sigmas[i + 1] == 0:
            x = denoised # Denoising step
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)

            if eta > 0:
                x = x + torch.randn_like(x) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt()

            h_last = h
        old_denoised = denoised
        
    return x