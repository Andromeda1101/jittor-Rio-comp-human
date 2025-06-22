import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
from jittor import optim
from jittor.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import gc
import sys
import argparse
import time

from dataset.dataset import get_dataloader
from dataset.format import parents
from dataset.sampler import SamplerMix
from models.metrics import J2J
from models.skeleton import create_model
from dataset.exporter import Exporter

# Set Jittor flags
jt.flags.use_cuda = 1

def train(args):
    # 在训练开始前清理GPU内存
    jt.gc()
    jt.sync_all()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # 创建模型
    model = create_model(model_name=args.model_name, output_channels=66)
    
    # 加载预训练模型
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # 创建优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                            lr=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)
    
    # 创建学习率调度器
    lr_scheduler = CosineAnnealingLR(optimizer, 
                                    T_max=args.epochs//4,
                                    eta_min=args.learning_rate/100)

    # 损失函数
    criterion = nn.MSELoss()
    
    # 获取数据加载器
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024),
        transform=None
    )

    val_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.val_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=SamplerMix(num_samples=1024),
        transform=None
    ) if args.val_data_list else None
    
    def compute_loss(pred, target):
        # 基础MSE损失
        mse_loss = criterion(pred, target)
        
        # 如果是pct2模型，添加骨骼约束损失
        if args.model_name == 'pct2':
            B = pred.shape[0]
            pred_reshaped = pred.reshape(B, -1, 3)  # [B, num_joints, 3]
            constraint_loss = model.bone_constraints.compute_constraint_loss(pred_reshaped)
            # 合并损失，constraint_weight在参数中设置
            total_loss = mse_loss + args.constraint_weight * constraint_loss
            return total_loss, {
                'mse_loss': mse_loss.item(),
                'constraint_loss': constraint_loss.item()
            }
        else:
            return mse_loss, {'mse_loss': mse_loss.item()}

    # 训练循环
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_constraint_loss = 0.0
        
        start_time = time.time()
        for batch_idx, (vertices, joints, _) in enumerate(train_loader):
            # 每50个batch清理一次GPU内存
            if batch_idx > 0 and batch_idx % 50 == 0:
                jt.sync_all()
                jt.gc()
                gc.collect()
                
            # 前向传播
            pred = model(vertices)
            
            # 计算损失
            loss, loss_dict = compute_loss(pred, joints)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新学习率
            lr_scheduler.step()
            
            # 记录损失
            train_loss += loss.item()
            train_mse_loss += loss_dict['mse_loss']
            if 'constraint_loss' in loss_dict:
                train_constraint_loss += loss_dict['constraint_loss']
            
            # 打印进度
            if (batch_idx + 1) % args.print_freq == 0:
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")
        
        # 计算平均损失
        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        if args.model_name == 'pct2':
            train_constraint_loss /= len(train_loader)
        
        # 验证阶段
        if val_loader:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            
            for vertices, joints, _ in val_loader:
                with jt.no_grad():
                    pred = model(vertices)
                    loss, _ = compute_loss(pred, joints)
                    val_loss += loss.item()
                    
                    # 计算J2J损失
                    B = pred.shape[0]
                    pred_reshaped = pred.reshape(B, -1, 3)
                    joints_reshaped = joints.reshape(B, -1, 3)
                    for i in range(B):
                        J2J_loss += J2J(pred_reshaped[i], joints_reshaped[i]).item()
            
            val_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {val_loss:.4f} to {model_path}")
        
        # 打印epoch统计信息
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f} "
                   f"Train MSE Loss: {train_mse_loss:.4f} "
                   + (f"Train Constraint Loss: {train_constraint_loss:.4f} " if args.model_name == 'pct2' else "")
                   + (f"Val Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}" if val_loader else ""))
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Skeleton Model')
    parser.add_argument('--model_name', type=str, default='pct2',
                        choices=['pct', 'pct2', 'jspct', 'enhanced', 'unified'],
                        help='Model architecture to use')
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to training data list')
    parser.add_argument('--val_data_list', type=str,
                        help='Path to validation data list')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--constraint_weight', type=float, default=1.0,
                        help='Weight for skeleton constraint loss')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tmp', 'skeleton'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tmp', 'skin'), exist_ok=True)
    
    train(args)

if __name__ == '__main__':
    main()