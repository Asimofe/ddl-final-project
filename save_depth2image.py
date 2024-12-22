import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 저장할 디렉토리 생성
output_dir = './depth_image'
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 패턴 설정
image_files = [f'./data/graspnet/scenes/scene_0000/realsense/depth/{i:04d}.png' for i in range(256)]

# 이미지 파일들을 순서대로 로드하고 시각화 및 저장
for image_file in image_files:
    # PNG 파일 로드 (16비트 정수로 읽어옴)
    depth_image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    # 깊이 이미지가 16비트일 경우, 시각화를 위해 정규화
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalized = np.uint8(depth_image_normalized)

    # 이미지를 시각화
    #plt.imshow(depth_image_normalized, cmap='gray')
    #plt.title(f'Depth Image - {image_file}')
    #plt.colorbar()
    #plt.show()

    # 정규화된 이미지를 depth_image 디렉토리에 저장
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, depth_image_normalized)

    print(f"Saved {output_path}")
