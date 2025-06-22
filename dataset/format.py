id_to_name = {
  0: 'hips',
  1: 'spine',
  2: 'chest',
  3: 'upper_chest',
  4: 'neck',
  5: 'head',
  6: 'l_shoulder',
  7: 'l_upper_arm',
  8: 'l_lower_arm',
  9: 'l_hand',
  10: 'r_shoulder',
  11: 'r_upper_arm',
  12: 'r_lower_arm',
  13: 'r_hand',
  14: 'l_upper_leg',
  15: 'l_lower_leg',
  16: 'l_foot',
  17: 'l_toe_base',
  18: 'r_upper_leg',
  19: 'r_lower_leg',
  20: 'r_foot',
  21: 'r_toe_base',
}

parents = [None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20,]

# 骨骼对称映射
symmetry_map = {
    # 左侧关节 -> 右侧关节
    6: 10, 7: 11, 8: 12, 9: 13,  # 手臂
    14: 18, 15: 19, 16: 20, 17: 21  # 腿部
}

# 对称关节对 (用于皮肤权重对称约束)
symmetric_joints = {
    # 左侧关节: 右侧关节
    6: 10, 7: 11, 8: 12, 9: 13,
    14: 18, 15: 19, 16: 20, 17: 21
}

# 骨骼长度对称对
symmetric_bones = [
    (6, 7, 10, 11),   # 上臂: l_shoulder->l_upper_arm vs r_shoulder->r_upper_arm
    (7, 8, 11, 12),   # 前臂: l_upper_arm->l_lower_arm vs r_upper_arm->r_lower_arm
    (8, 9, 12, 13),   # 手部: l_lower_arm->l_hand vs r_lower_arm->r_hand
    (14, 15, 18, 19), # 大腿: l_upper_leg->l_lower_leg vs r_upper_leg->r_lower_leg
    (15, 16, 19, 20), # 小腿: l_lower_leg->l_foot vs r_lower_leg->r_foot
    (16, 17, 20, 21)  # 脚部: l_foot->l_toe_base vs r_foot->r_toe_base
]

# 关节角度约束
angle_constraints = [
    # (子关节, 父关节, 祖父关节, 最大角度(弧度))
    (7, 6, 3, 2.0),    # 左肩关节
    (11, 10, 3, 2.0),   # 右肩关节
    (8, 7, 6, 2.5),     # 左肘关节
    (12, 11, 10, 2.5),  # 右肘关节
    (15, 14, 0, 2.2),   # 左髋关节
    (19, 18, 0, 2.2),   # 右髋关节
    (16, 15, 14, 2.5),  # 左膝关节
    (20, 19, 18, 2.5)   # 右膝关节
]