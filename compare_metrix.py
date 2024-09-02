# 对比两个目录之间的图像生成指标 PSNR SSIM
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tqdm
# 读取两个文件夹中的图片
folder1 = "/mnt/sda/Datasets/chuxinning/PIR_IR_0730_crop/test_A"   #这里替换成你的文件夹路径
folder2 = "/mnt/sda/Datasets/chuxinning/PIR_IR_0730_crop/test_B"          #这里替换成你的文件夹路径
 
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)
 
psnr_values = []
ssim_values = []
 
for idx, (file1, file2) in tqdm.tqdm(enumerate(zip(files1, files2), start=1)):
    img1 = cv2.imread(os.path.join(folder1, file1))
    img2 = cv2.imread(os.path.join(folder2, file2))
 
    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    # 计算 PSNR
    psnr_val = psnr(gray1, gray2)
    psnr_values.append(psnr_val)
 
    # 计算 SSIM
    ssim_val, _ = ssim(gray1, gray2, full=True)
    ssim_values.append(ssim_val)
 
    print(f"Processing pair {idx}: {file1} and {file2}")
 
# 计算平均值
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)
 
print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")