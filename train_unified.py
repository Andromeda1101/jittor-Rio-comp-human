import jittor as jt
import numpy as np
import os
import argparse
import time
import random
import wandb

from jittor import nn
from jittor import optim
from jittor.lr_scheduler import CosineAnnealingLR

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name, parents
from dataset.sampler import SamplerMix
from models.metrics import J2J
from models.unified import UnifiedModel
from dataset.exporter import Exporter

# Set Jittor flags
jt.flags.use_cuda = 1

def train(args):
    # 初始化wandb
    # wandb.init(project="unified-model", config=vars(args))
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    log_message(f"Starting unified training with parameters: {args}")
    
    # 创建模型
    model = UnifiedModel(
        feat_dim=args.feat_dim,
        num_joints=args.num_joints
    )
    
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
    
    lr_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs//4,  # 每1/4训练周期重启一次
        eta_min=args.learning_rate/100
    )

    # 创建损失函数
    criterion_joint = nn.MSELoss()
    criterion_skin_mse = nn.MSELoss()
    criterion_skin_l1 = nn.L1Loss()

    # 蒙皮损失：KL散度 + 正则化项
    def SkinLoss(pred, target, vertices, joints):
        # KL散度损失
        criterion_kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        kl_loss = criterion_kl_loss(nn.log_softmax(pred, dim=-1), nn.log_softmax(target, dim=-1))
        
        # 空间连续性正则化
        diff = pred[:, 1:, :] - pred[:, :-1, :]
        spatial_reg = jt.mean(jt.abs(diff))
        
        # 调整vertices形状为(B, N, 3)以进行距离计算
        vertices = vertices.permute(0, 2, 1)  # 从(B, 3, N)变为(B, N, 3)
        
        # 关节-顶点距离计算 - 使用广播机制
        # vertices: (B, N, 1, 3)
        # joints: (B, 1, J, 3)
        vertices_expanded = vertices.unsqueeze(2)  # (B, N, 1, 3)
        joints_expanded = joints.unsqueeze(1)      # (B, 1, J, 3)
        
        # 计算每个顶点到每个关节的距离
        dist_pred = jt.norm(vertices_expanded - joints_expanded, dim=-1)  # (B, N, J)
        weight_dist = jt.exp(-dist_pred * 0.5)
        dist_consistency = nn.mse_loss(pred, weight_dist)
        
        return kl_loss + 0.1 * spatial_reg + 0.05 * dist_consistency
    
    # 创建数据加载器
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=512),
        transform=transform,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=1024, vertex_samples=512),
            transform=transform,
        )
    else:
        val_loader = None
    
    # 训练循环
    best_loss = float('inf')
    no_improve_epochs = 0
    stop1 = 0  # 用于早停

    for epoch in range(args.epochs):
        model.train()
        train_joint_loss = 0.0
        train_skin_l1loss = 0.0
        train_skin_mseloss = 0.0
        train_skin_klloss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']
            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var

            # 确保vertices的形状为(B, 3, N)
            if vertices.shape[1] != 3:
                vertices = vertices.permute(0, 2, 1)
            
            # 前向传播
            joint_pred, skin_pred = model(vertices)  # 输出 (B, 22, 3), (B, N, 22)
            
            # 调整joints格式以匹配预测结果
            joints = joints.reshape(joint_pred.shape)  # 确保为 (B, 22, 3)

            # 计算损失
            joint_loss = criterion_joint(joint_pred, joints)
            skin_mseloss = criterion_skin_mse(skin_pred, skin)
            skin_l1loss = criterion_skin_l1(skin_pred, skin)
            skin_klloss = SkinLoss(skin_pred, skin, vertices, joint_pred)
            total_loss = joint_loss + args.skin_weight * skin_klloss
            total_loss /= args.accum_steps
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(total_loss)
            optimizer.step()
            
            if (batch_idx + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()  # 更新学习率
            
            # 记录损失
            train_joint_loss += joint_loss.item()
            train_skin_l1loss += skin_l1loss.item()
            train_skin_mseloss += skin_mseloss.item()
            train_skin_klloss += skin_klloss.item()
            
            # 打印进度
            if (batch_idx + 1) % args.print_freq == 0:
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Joint Loss: {joint_loss.item():.4f} Skin MSE Loss: {skin_mseloss.item():.4f} Skin L1 Loss: {skin_l1loss.item():.4f} Skin KL Loss: {skin_klloss.item():.4f}")
        

        # 计算epoch统计
        train_joint_loss /= len(train_loader)
        train_skin_mseloss /= len(train_loader)
        train_skin_l1loss /= len(train_loader)
        train_skin_klloss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Joint Loss: {train_joint_loss:.4f} "
                   f"Train Skin MSE Loss: {train_skin_mseloss:.4f} "
                   f"Train Skin L1 Loss: {train_skin_l1loss:.4f} "
                   f"Train Skin KL Loss: {train_skin_klloss:.4f} "
                   f"Time: {epoch_time:.2f}s")
        
        # 验证阶段
        if val_loader and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_joint_loss = 0.0
            J2J_loss = 0.0
            l1_loss = 0.0
            mse_loss = 0.0

            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                
                # 调整输入形状
                vertices = vertices.permute(0, 2, 1)  # (B, 3, N) - 输入要求
                
                # 前向传播
                with jt.no_grad():
                    joint_pred, skin_pred = model(vertices)  # 输出 (B, 22, 3), (B, N, 22)
                
                joints = joints.reshape(joint_pred.shape)  # 确保为 (B, 22, 3)

                # 计算损失
                joint_loss = criterion_joint(joint_pred, joints)
                skin_l1loss = criterion_skin_l1(skin_pred, skin)
                skin_mseloss = criterion_skin_mse(skin_pred, skin)

                # 展示部分调整
                if batch_idx == show_id:
                    exporter = Exporter()
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_ref.png", 
                                           joints=joints[0].numpy(), parents=parents)  # 已经是(22, 3)
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_pred.png", 
                                           joints=joint_pred[0].numpy(), parents=parents)
                    exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/vertices.png", 
                                      vertices=vertices[0].permute(1, 0).numpy())  # 转回(N, 3)用于显示
                    for i in id_to_name:
                        name = id_to_name[i]
                        # export every joint's corresponding skinning
                        vertices_vis = vertices[0].permute(1, 0)  # 先转换为 (N, 3)
                        exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_ref.png",
                                           vertices=vertices_vis.numpy(), 
                                           skin=skin.numpy()[0, :, i], 
                                           joint=joints[0, i])
                        exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_pred.png",
                                           vertices=vertices_vis.numpy(), 
                                           skin=skin_pred.numpy()[0, :, i], 
                                           joint=joints[0, i])
                
                val_joint_loss += joint_loss.item()
                l1_loss += skin_l1loss.item()
                mse_loss += skin_mseloss.item()
                for i in range(joint_pred.shape[0]):
                    J2J_loss += J2J(joint_pred[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / joint_pred.shape[0] 
                
            
            # 计算验证损失
            val_joint_loss /= len(val_loader)
            l1_loss /= len(val_loader)
            mse_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            total_val_loss = val_joint_loss + args.skin_weight * (l1_loss + mse_loss)
            
            log_message(f"Validation J2J Loss: {J2J_loss:.4f} "
                      f"Skin MSE Loss: {mse_loss:.4f} "
                      f"Skin L1 Loss: {l1_loss:.4f} "
                      )
            
            if stop1 == 0:
                # 保存最佳模型
                if l1_loss < best_loss:
                    best_loss = l1_loss
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with L1 loss {l1_loss:.4f} to {model_path}")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= args.patience // 2:
                        log_message(f"Early stopping 1 triggered after {epoch+1} epochs")
                        stop1 = 1
                        no_improve_epochs = 0
                        best_loss = J2J_loss + l1_loss
            else:
                if J2J_loss + l1_loss < best_loss:
                    best_loss = J2J_loss + l1_loss
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with J2J loss {J2J_loss:.4f} and L1 loss {l1_loss:.4f} to {model_path}")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= args.patience:
                        log_message(f"Early stopping 2 triggered after {epoch+1} epochs")
                        break
        
        # 保存检查点
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")
    
    # wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Unified Joint and Skin Training')
    
    # 数据集参数
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to training data list')
    parser.add_argument('--val_data_list', type=str, 
                        help='Path to validation data list')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for data')
    
    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension size')
    parser.add_argument('--num_joints', type=int, default=22,
                        help='Number of joints')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam','adamw'], help='Optimizer type')
    parser.add_argument('--skin_weight', type=float, default=2.0,
                        help='Weight for skin loss')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--skin_temperature', type=float, default=0.1,
                        help='Temperature for skinning softmax')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    parser.add_argument('--visual_freq', type=int, default=5,
                        help='Visualization frequency')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # 创建可视化目录
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    
    # 开始训练
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()