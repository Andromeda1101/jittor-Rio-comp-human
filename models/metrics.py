import jittor as jt
from jittor import nn

def J2J(
    joints_a: jt.Var,
    joints_b: jt.Var,
) -> jt.Var:
    '''
    calculate J2J loss in [-1, 1]^3 cube
    
    joints_a: (J1, 3) joint

    joints_b: (J2, 3) joint
    '''
    assert isinstance(joints_a, jt.Var)
    assert isinstance(joints_b, jt.Var)
    assert joints_a.ndim == 2, "joints_a should be shape (J1, 3)"
    assert joints_b.ndim == 2, "joints_b should be shape (J2, 3)"
    dis1 = ((joints_a.unsqueeze(0) - joints_b.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss1 = dis1.min(dim=-1)
    dis2 = ((joints_b.unsqueeze(0) - joints_a.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss2 = dis2.min(dim=-1)
    return (loss1.mean() + loss2.mean()) / 2 / 2

def symmetric_bone_length_constraint(pred_joints, symmetric_bones):
    """
    约束对称骨骼长度相等
    pred_joints: [B, J, 3] 预测的关节位置
    symmetric_bones: 对称骨骼对列表
    """
    loss = 0.0
    num_constraints = 0
    
    for bone_pair in symmetric_bones:
        left_start, left_end, right_start, right_end = bone_pair
        
        # 计算左侧骨骼长度
        left_vec = pred_joints[:, left_end] - pred_joints[:, left_start]
        left_length = jt.norm(left_vec, dim=1, keepdim=True)
        
        # 计算右侧骨骼长度
        right_vec = pred_joints[:, right_end] - pred_joints[:, right_start]
        right_length = jt.norm(right_vec, dim=1, keepdim=True)
        
        # 计算长度差异损失
        length_diff = left_length - right_length
        loss += jt.mean(length_diff ** 2)
        num_constraints += 1
    
    if num_constraints == 0:
        return 0.0
        
    return loss / num_constraints

def adaptive_joint_angle_constraint(pred_joints, angle_constraints):
    """
    自适应关节角度约束
    pred_joints: [B, J, 3] 预测的关节位置
    angle_constraints: 约束列表
    """
    loss = 0.0
    num_constraints = 0
    
    for constraint in angle_constraints:
        if len(constraint) < 4:
            continue
            
        child, parent, grandparent, max_angle = constraint
        
        # 计算从祖父关节到父关节的向量
        v1 = pred_joints[:, parent] - pred_joints[:, grandparent]
        # 计算从父关节到子关节的向量
        v2 = pred_joints[:, child] - pred_joints[:, parent]
        
        # 计算角度
        dot_product = (v1 * v2).sum(dim=1)
        norm_product = jt.norm(v1, dim=1) * jt.norm(v2, dim=1)
        cos_theta = dot_product / (norm_product + 1e-7)
        cos_theta = jt.clamp(cos_theta, -1.0, 1.0)
        angle = jt.acos(cos_theta)
        
        # 只惩罚超过最大角度的部分
        angle_violation = nn.relu(angle - max_angle)
        loss += jt.mean(angle_violation ** 2)
        num_constraints += 1
    
    if num_constraints == 0:
        return 0.0
        
    return loss / num_constraints

def skin_symmetry_constraint(pred_skin, symmetry_map, symmetric_joints):
    """
    约束对称顶点的皮肤权重对称
    pred_skin: [B, V, J] 预测的皮肤权重
    symmetry_map: [V] 每个顶点的对称顶点索引
    symmetric_joints: 对称关节对字典 {左关节: 右关节}
    """
    loss = 0.0
    num_pairs = 0
    B, V, J = pred_skin.shape
    
    # 遍历所有顶点
    for v in range(V):
        sym_v = symmetry_map.get(v, -1)
        if sym_v == -1 or sym_v >= V:
            continue
            
        # 遍历所有对称关节对
        for left_joint, right_joint in symmetric_joints.items():
            if left_joint >= J or right_joint >= J:
                continue
                
            # 获取左侧顶点在左侧关节的权重
            left_weight = pred_skin[:, v, left_joint]
            # 获取右侧顶点在右侧关节的权重
            right_weight = pred_skin[:, sym_v, right_joint]
            
            # 计算权重差异损失
            weight_diff = left_weight - right_weight
            loss += jt.mean(weight_diff ** 2)
            num_pairs += 1
    
    if num_pairs == 0:
        return 0.0
        
    return loss / num_pairs