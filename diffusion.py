import torch

## diffusion code inspired by https://github.com/pabloppp/pytorch-tools/blob/master/torchtools/utils/diffusion.py and Würstchen (ICLR 2024, oral) https://github.com/dome272/Wuerstchen/tree/main

# Samplers --------------------------------------------------------------------
class SimpleSampler():
    def __init__(self, diffuzz):
        self.current_step = -1
        self.diffuzz = diffuzz

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape):
        return torch.randn(*shape, device=self.diffuzz.device)

    def step(self, x, t, t_prev, noise):
        raise NotImplementedError("You should override the 'apply' function.")

class DDPMSampler(SimpleSampler):
    def step(self, x, t, t_prev, noise):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        mu = (1.0 / alpha).sqrt() * (x - (1-alpha) * noise / (1-alpha_cumprod).sqrt())
        std = ((1-alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * torch.randn_like(mu)
        return mu + std * (t_prev != 0).float().view(t_prev.size(0), *[1 for _ in x.shape[1:]])

class DDIMSampler(SimpleSampler):
    def step(self, x, t, t_prev, noise):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])

        x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / (alpha_cumprod).sqrt()
        dp_xt = (1 - alpha_cumprod_prev).sqrt()
        return (alpha_cumprod_prev).sqrt() * x0 + dp_xt * noise


sampler_dict = {
    'ddpm': DDPMSampler,
    'ddim': DDIMSampler,
}

# Custom simplified forward/backward diffusion (cosine schedule)
class Diffuzz():
    def __init__(self, s=0.008, device="cpu", cache_steps=None, scaler=1, clamp_range=(0.0001, 0.9999)):
        self.device = device
        self.s = torch.tensor([s]).to(device)
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2
        self.scaler = scaler
        self.cached_steps = None
        self.clamp_range = clamp_range
        if cache_steps is not None:
            self.cached_steps = self._alpha_cumprod(torch.linspace(0, 1, cache_steps, device=device))

    def _alpha_cumprod(self, t):
        if self.cached_steps is None:
            if self.scaler > 1:
                t = 1 - (1-t) ** self.scaler
            elif self.scaler < 1:
                t = t ** self.scaler
            alpha_cumprod = torch.cos((t + self.s) / (1 + self.s) * torch.pi * 0.5).clamp(0, 1) ** 2 / self._init_alpha_cumprod
            return alpha_cumprod.clamp(self.clamp_range[0], self.clamp_range[1])
        else:
            return self.cached_steps[t.mul(len(self.cached_steps)-1).long()]

    def diffuse(self, x, t, noise=None): # t -> [0, 1]
        if noise is None:
            noise = torch.randn_like(x)
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        return alpha_cumprod.sqrt() * x + (1-alpha_cumprod).sqrt() * noise, noise

    def undiffuse(self, x, t, t_prev, noise, sampler=None):
        if sampler is None:
            sampler = DDPMSampler(self)
        return sampler(x, t, t_prev, noise)

    def sample(self, model, model_inputs, shape, mask=None, t_start=1.0, t_end=0.0, timesteps=20, x_init=None, sampler='ddpm', steps="linear"):
        if steps=="linear":
            r_range = torch.linspace(t_start, t_end, timesteps+1)[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)
        elif steps=="ays":
            ## implementation of "Align Your Steps: Optimizing Sampling Schedules in Diffusion Models"
            #assert sampler == "ddim"
            i = torch.arange(timesteps+1)
            r_range = torch.tan((1 - i/timesteps)*math.atan(t_start) + (i/timesteps) * math.atan(t_end))[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)
        else:
            raise NotImplementedError(f"{steps=} not implemented")
        
        if isinstance(sampler, str):
            if sampler in sampler_dict:
                sampler = sampler_dict[sampler](self)
            else:
                raise ValueError(f"If sampler is a string it must be one of the supported samplers: {list(sampler_dict.keys())}")
        elif issubclass(sampler, SimpleSampler):
            sampler =  sampler(self)
        else:
            raise ValueError("Sampler should be either a string or a SimpleSampler object.")
        preds = []
        x = sampler.init_x(shape) if x_init is None or mask is not None else x_init.clone()
     
        for i in range(0, timesteps):
            if mask is not None and x_init is not None:
                x_renoised, _ = self.diffuse(x_init, r_range[i])
                x = x * mask + x_renoised * (1-mask)

            pred_noise = model(x, r_range[i], **model_inputs)

            x = self.undiffuse(x, r_range[i], r_range[i+1], pred_noise, sampler=sampler)
            preds.append(x)
        return preds

    def p2_weight(self, t, k=1.0, gamma=1.0):
        alpha_cumprod = self._alpha_cumprod(t)
        return (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma

