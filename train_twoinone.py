import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn, optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from models.twoinone import JointSkinModel

jt.flags.use_cuda = 0  # 训练建议用1

def train(args):
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)

    log_message(f"Starting training with parameters: {args}")

    # 初始化采样器
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)

    # 创建模型
    model = JointSkinModel(feat_dim=args.feat_dim, num_joints=args.num_joints)
    if jt.flags.use_cuda:
        model.cuda()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_joint = nn.MSELoss()
    criterion_skin = nn.MSELoss()

    # 数据加载
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
    )
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            transform=transform,
        )
    else:
        val_loader = None

    best_loss = 1e10
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_joint = 0.0
        train_loss_skin = 0.0
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            try:
                vertices = data['vertices']
                gt_joints = data['joints']
                gt_skin = data['skin']
                # [B, N, 3] -> [B, N, 3]，如需 [B, 3, N] 可 permute
                # vertices = vertices.permute(0, 2, 1)  # 若模型需要

                if jt.flags.use_cuda:
                    vertices = vertices.cuda()
                    gt_joints = gt_joints.cuda()
                    gt_skin = gt_skin.cuda()

                pred_joints, pred_skin = model(vertices)
                loss_joint = criterion_joint(pred_joints, gt_joints)
                loss_skin = criterion_skin(pred_skin, gt_skin)
                loss = loss_joint + loss_skin

                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()

                train_loss_joint += loss_joint.item()
                train_loss_skin += loss_skin.item()

                if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                    log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                                f"Loss joint: {loss_joint.item():.4f} Loss skin: {loss_skin.item():.4f}")
            except Exception as e:
                log_message(f"Batch {batch_idx+1} error: {e}")
                continue

        train_loss_joint /= len(train_loader)
        train_loss_skin /= len(train_loader)
        epoch_time = time.time() - start_time
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss joint: {train_loss_joint:.4f} "
                    f"Train Loss skin: {train_loss_skin:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}")

        # 验证
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_joint = 0.0
            val_loss_skin = 0.0
            for batch_idx, data in enumerate(val_loader):
                try:
                    vertices = data['vertices']
                    gt_joints = data['joints']
                    gt_skin = data['skin']
                    if jt.flags.use_cuda:
                        vertices = vertices.cuda()
                        gt_joints = gt_joints.cuda()
                        gt_skin = gt_skin.cuda()
                    pred_joints, pred_skin = model(vertices)
                    loss_joint = criterion_joint(pred_joints, gt_joints)
                    loss_skin = criterion_skin(pred_skin, gt_skin)
                    val_loss_joint += loss_joint.item()
                    val_loss_skin += loss_skin.item()
                except Exception as e:
                    log_message(f"Val batch {batch_idx+1} error: {e}")
                    continue
            val_loss_joint /= len(val_loader)
            val_loss_skin /= len(val_loader)
            log_message(f"Validation Loss: joint: {val_loss_joint:.4f} skin: {val_loss_skin:.4f}")

            # 保存最优模型
            if val_loss_joint + val_loss_skin < best_loss:
                best_loss = val_loss_joint + val_loss_skin
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.patience:
                    log_message(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # 保存checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train JointSkinModel')
    parser.add_argument('--train_data_list', type=str, default='data/train_list.txt')
    parser.add_argument('--val_data_list', type=str, default='data/val_list.txt')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='twoinone')
    parser.add_argument('--output_dir', type=str, default='output/twoinone')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--feat_dim', type=int, default=384)
    parser.add_argument('--num_joints', type=int, default=22)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()