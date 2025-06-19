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

# Set Jittor flags
jt.flags.use_cuda = 1

def chamfer_distance(pred, gt):
    # pred, gt: [B, J, 3]
    B, J, _ = pred.shape
    pred_expand = pred.unsqueeze(2)  # [B, J, 1, 3]
    gt_expand = gt.unsqueeze(1)      # [B, 1, J, 3]
    dist = ((pred_expand - gt_expand) ** 2).sum(-1)  # [B, J, J]
    min_pred_gt = dist.min(dim=2)[0]  # [B, J]
    min_gt_pred = dist.min(dim=1)[0]  # [B, J]
    cd = min_pred_gt.mean() + min_gt_pred.mean()

    return cd

def train(args):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    log_message(f"Starting training with parameters: {args}")

    # Create model
    model = JointSkinModel(feat_dim=args.feat_dim, num_joints=args.num_joints)
    
    if jt.flags.use_cuda:
        model.cuda()

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Create loss functions
    criterion_skin_mse = nn.MSELoss()
    criterion_skin_l1 = nn.L1Loss()
    criterion_joint = nn.MSELoss()  
    # Create dataloaders
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
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
        train_loss_skin_mse = 0.0
        train_loss_skin_l1 = 0.0
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            vertices = data['vertices']
            gt_joints = data['joints']
            gt_skin = data['skin']
            
            vertices: jt.Var
            gt_joints: jt.Var
            gt_skin: jt.Var
            
            if jt.flags.use_cuda:
                vertices = vertices.cuda()
                gt_joints = gt_joints.cuda()
                gt_skin = gt_skin.cuda()

            pred_joints, pred_skin = model(vertices)
            # joints loss: MSE
            loss_joint = chamfer_distance(pred_joints, gt_joints)
            # skin loss: MSE + L1
            loss_skin_mse = criterion_skin_mse(pred_skin, gt_skin)
            loss_skin_l1 = criterion_skin_l1(pred_skin, gt_skin)
            # 可选骨长损失
            bone_length_pred = jt.norm(pred_joints[:, 1:] - pred_joints[:, :-1], dim=-1)
            bone_length_gt = jt.norm(gt_joints[:, 1:] - gt_joints[:, :-1], dim=-1)
            loss_bone = nn.L1Loss()(bone_length_pred, bone_length_gt)

            # log_message(f"size of loss_joint: {loss_joint.shape}, loss_skin_mse: {loss_skin_mse.shape}, loss_skin_l1: {loss_skin_l1.shape}, loss_bone: {loss_bone.shape}")
            # 总损失
            
            loss = loss_joint + loss_skin_mse + loss_skin_l1 + 0.1 * loss_bone
            log_message(111)
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            log_message(222)
            train_loss_joint += loss_joint.item()
            log_message(333)
            train_loss_skin_mse += loss_skin_mse.item()
            train_loss_skin_l1 += loss_skin_l1.item()
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss joint(CD): {loss_joint.item():.4f} "
                            f"Loss skin(MSE): {loss_skin_mse.item():.4f} "
                            f"Loss skin(L1): {loss_skin_l1.item():.4f}")

        train_loss_joint /= len(train_loader)
        train_loss_skin_mse /= len(train_loader)
        train_loss_skin_l1 /= len(train_loader)
        epoch_time = time.time() - start_time
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss joint(CD): {train_loss_joint:.4f} "
                    f"Train Loss skin(MSE): {train_loss_skin_mse:.4f} "
                    f"Train Loss skin(L1): {train_loss_skin_l1:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}")

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_joint = 0.0
            val_loss_skin_mse = 0.0
            val_loss_skin_l1 = 0.0
            for batch_idx, data in enumerate(val_loader):
                vertices = data['vertices']
                gt_joints = data['joints']
                gt_skin = data['skin']
                if jt.flags.use_cuda:
                    vertices = vertices.cuda()
                    gt_joints = gt_joints.cuda()
                    gt_skin = gt_skin.cuda()
                    
                pred_joints, pred_skin = model(vertices)
                loss_joint = chamfer_distance(pred_joints, gt_joints)
                loss_skin_mse = criterion_skin_mse(pred_skin, gt_skin)
                loss_skin_l1 = criterion_skin_l1(pred_skin, gt_skin)
                
                val_loss_joint += loss_joint.item()
                val_loss_skin_mse += loss_skin_mse.item()
                val_loss_skin_l1 += loss_skin_l1.item()
                
            val_loss_joint /= len(val_loader)
            val_loss_skin_mse /= len(val_loader)
            val_loss_skin_l1 /= len(val_loader)
            log_message(f"Validation Loss: joint(CD): {val_loss_joint:.4f} "
                        f"skin(MSE): {val_loss_skin_mse:.4f} "
                        f"skin(L1): {val_loss_skin_l1:.4f}")

            # Save best model
            total_val_loss = val_loss_joint + val_loss_skin_mse + val_loss_skin_l1
            if total_val_loss < best_loss:
                best_loss = total_val_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.patience:
                    log_message(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train JointSkinModel')
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, default='data/train_list.txt')
    parser.add_argument('--val_data_list', type=str, default='data/val_list.txt')
    parser.add_argument('--data_root', type=str, default='data')
    # Model parameters
    parser.add_argument('--feat_dim', type=int, default=384)
    parser.add_argument('--num_joints', type=int, default=22)
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='output/twoinone')
    args = parser.parse_args()
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()