import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(dim=64, dim_mults=(1, 2, 4, 8))

diffusion = GaussianDiffusion(
    model,
    image_size=64,  # 이미지 크기
    num_frames=10,  # 프레임 수
    timesteps=1000,  # 디퓨전 단계
    loss_type='l1'   # 손실 함수
)

trainer = Trainer(
    diffusion,
    '/home/im_jg/video-diffusion-pytorch/output_video.gif',  # .gif 데이터 폴더 경로
    train_batch_size=32,
    train_lr=1e-4,
    train_num_steps=700000  # 훈련 단계 수
)

trainer.train()  # 모델 학습 시작
