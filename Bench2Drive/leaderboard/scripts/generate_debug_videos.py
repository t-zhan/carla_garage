#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm  # 进度条库

def natural_sort_key(s):
    """自然排序键函数"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def main():
    parser = argparse.ArgumentParser(description='使用OpenCV从图片生成视频')
    parser.add_argument('--input', default='Bench2Drive/leaderboard/data/eval_bench2drive220_max_traj', 
                        help='输入目录路径')
    parser.add_argument('--output', default='combined_video.mp4', 
                        help='输出视频路径')
    parser.add_argument('--fps', type=int, default=20, help='输出视频帧率')
    args = parser.parse_args()

    # 处理输入路径
    input_dir = Path(args.input)
    output_file = input_dir / args.output
    
    print(f"处理目录: {input_dir}")
    print(f"输出视频: {output_file}")
    
    # 扫描所有符合条件的目录
    scenario_dirs = []
    for dir_path in input_dir.iterdir():
        if dir_path.is_dir() and 'RouteScenario' in dir_path.name:
            # 提取目录名中的数字用于排序
            match = re.search(r'RouteScenario_(\d+)', dir_path.name)
            if match:
                sort_key = int(match.group(1))
                scenario_dirs.append((sort_key, dir_path))
    
    # 按数字排序目录
    scenario_dirs.sort(key=lambda x: x[0])
    # scenario_dirs = scenario_dirs[:10]  # 只处理前10个目录
    
    # 收集所有图片文件
    all_images = []
    for idx, (sort_key, dir_path) in enumerate(scenario_dirs):
        print(f"[{idx+1}/{len(scenario_dirs)}] 扫描目录: {dir_path.name}")
        
        # 获取目录下所有PNG文件并按自然顺序排序
        image_files = sorted(dir_path.glob('*.png'), key=natural_sort_key)
        all_images.extend(image_files)
    
    # 验证图片数量
    if not all_images:
        print("错误: 未找到图片文件")
        return
    
    print(f"找到图片数量: {len(all_images)}")
    
    # 从第一张图片获取视频分辨率
    first_image = cv2.imread(str(all_images[0]))
    if first_image is None:
        print(f"无法读取首张图片: {all_images[0]}")
        return
    
    height, width, _ = first_image.shape
    print(f"视频分辨率: {width}x{height}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    video = cv2.VideoWriter(str(output_file), fourcc, args.fps, (width, height))
    
    if not video.isOpened():
        print("无法创建视频文件")
        return
    
    # 处理每张图片
    print("生成视频...")
    for img_path in tqdm(all_images, desc="处理图片"):
        img = cv2.imread(str(img_path))
        if img is not None:
            # 确保所有图片分辨率一致
            if img.shape != (height, width, 3):
                img = cv2.resize(img, (width, height))
            video.write(img)
    
    # 释放资源
    video.release()
    print(f"视频生成完成: {output_file}")

if __name__ == '__main__':
    main()