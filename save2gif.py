import imageio
import os

# 이미지 파일 패턴 설정
image_files = [f'./data/graspnet/scenes/scene_0000/realsense/rgb/{i:04d}.png' for i in range(256)]
images = []

# 이미지 파일을 불러와서 리스트에 추가
for image_file in image_files:
    images.append(imageio.imread(image_file))

# GIF 파일로 저장
output_gif = 'rgb_animation.gif'
imageio.mimsave(output_gif, images, duration=0.5)  # duration은 각 프레임 간의 시간 (초)입니다.

print(f"Saved {output_gif}")


