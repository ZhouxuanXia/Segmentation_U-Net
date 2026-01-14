#!/usr/bin/env python3
"""
查看 PyTorch checkpoint 文件内容的脚本
用法: python view_checkpoint.py checkpoints/checkpoint_epoch1.pth
"""

import argparse
import torch


def view_checkpoint(checkpoint_path):
    """加载并显示 checkpoint 内容"""
    print(f"\n{'='*60}")
    print(f"📂 加载模型: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 显示所有 keys
    print(f"📋 模型包含 {len(checkpoint)} 个参数层:\n")
    
    total_params = 0
    for key, value in checkpoint.items():
        if hasattr(value, 'shape'):
            params = value.numel()
            total_params += params
            print(f"  {key}")
            print(f"    └─ 形状: {list(value.shape)}, 参数量: {params:,}")
        else:
            print(f"  {key}")
            print(f"    └─ 类型: {type(value).__name__}, 值: {value}")
    
    print(f"\n{'='*60}")
    print(f"📊 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='查看 PyTorch checkpoint 文件内容')
    parser.add_argument('checkpoint', type=str, help='checkpoint 文件路径')
    args = parser.parse_args()
    
    view_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()
