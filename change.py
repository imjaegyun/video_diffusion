from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(mp4_path, gif_path):
    # VideoFileClip 객체를 생성하여 mp4 파일 로드
    clip = VideoFileClip(mp4_path)
    
    # GIF 파일로 변환 (fps는 프레임 속도, 필요에 따라 조절 가능)
    clip.write_gif(gif_path, fps=15)

# 사용 예시
mp4_file = '/home/im_jg/video-diffusion-pytorch/bb_1_140702_two-wheeled-vehicle_216_24920.mp4'  # MP4 파일 경로
gif_file = 'output_video.gif'  # 저장할 GIF 파일 경로

convert_mp4_to_gif(mp4_file, gif_file)
