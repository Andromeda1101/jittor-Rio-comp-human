import jittor as jt
import numpy as np
import os
import argparse
from tqdm import tqdm

from models.twoinone import JointSkinModel
from dataset.dataset import get_dataloader

jt.flags.use_cuda = 1

def predict(args):
    model = JointSkinModel(feat_dim=args.feat_dim, num_joints=args.num_joints)
    model.load_parameters(args.model_path)
    model.eval()
    loader = get_dataloader(split='test', batch_size=1, shuffle=False)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("start predicting...")
    for batch_idx, batch in tqdm(enumerate(loader)):
        vertices = batch['vertices'].cuda()  # [1, N, 3]
        cls = batch.get('cls', [f'class_{batch_idx}'])
        id_ = batch.get('id', [batch_idx])

        pred_joints, pred_skin = model(vertices)
        pred_joints = pred_joints.numpy()[0]  # [J, 3]
        pred_skin = pred_skin.numpy()[0]      # [N, J]

        # 保存结果
        save_path = os.path.join(output_dir, str(cls[0]), str(id_[0]))
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "predict_joints.npy"), pred_joints)
        np.save(os.path.join(save_path, "predict_skin.npy"), pred_skin)

    print("finished")

def main():
    parser = argparse.ArgumentParser(description='Predict with JointSkinModel')
    parser.add_argument('--feat_dim', type=int, default=256)
    parser.add_argument('--num_joints', type=int, default=22)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/twoinone_pred')
    args = parser.parse_args()
    predict(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()