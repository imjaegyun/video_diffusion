import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    num_frames = 5,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

videos = torch.randn(1, 3, 5, 32, 32) # video (batch, channels, frames, height, width) - normalized from -1 to +1
loss = diffusion(videos)
loss.backward()
# after a lot of training

sampled_videos = diffusion.sample(batch_size = 4)
sampled_videos.shape # (4, 3, 5, 32, 32)