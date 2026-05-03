import torch
from pytorch_kinematics import transforms


def calc_jacobian(serial_chain, th, tool=None, frame="world"):
    """
    Return robot geometric Jacobian J (N,6,DOF) where dot{x} = J dot{q}.
    The first 3 rows relate the translational velocities and the
    last 3 rows relate the angular velocities.

    tool is the transformation wrt the end effector; default is identity. If specified, will have to
    specify for each of the N inputs.

    frame controls how the twist is expressed:
        - "world": base/world frame
        - "local": end-effector/tool frame
    """
    if frame not in ("world", "local"):
        raise ValueError(f"Unsupported Jacobian frame '{frame}'. Expected 'world' or 'local'.")

    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=serial_chain.dtype, device=serial_chain.device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    ndof = th.shape[1]

    if tool is None:
        tool_matrix = transforms.Transform3d(
            device=serial_chain.device, dtype=serial_chain.dtype
        ).get_matrix().repeat(N, 1, 1)
    else:
        if tool.dtype != serial_chain.dtype or tool.device != serial_chain.device:
            tool = tool.to(device=serial_chain.device, copy=True, dtype=serial_chain.dtype)
        tool_matrix = tool.get_matrix()

    ee_pose = serial_chain.forward_kinematics(th).get_matrix()
    ee_pos = ee_pose[:, :3, 3]
    ee_rot = ee_pose[:, :3, :3]
    tool_offset = torch.matmul(ee_rot, tool_matrix[:, :3, 3].unsqueeze(-1)).squeeze(-1)
    target_pos = ee_pos + tool_offset

    J = torch.zeros((N, 6, ndof), dtype=serial_chain.dtype, device=serial_chain.device)

    world = transforms.Transform3d(
        device=serial_chain.device, dtype=serial_chain.dtype
    ).get_matrix().repeat(N, 1, 1)

    joint_idx = 0
    for f in serial_chain._serial_frames:
        if f.joint.joint_type == "fixed":
            theta = torch.zeros((N, 1), dtype=serial_chain.dtype, device=serial_chain.device)
        else:
            theta = th[:, joint_idx].view(N, 1)

        joint_tf = world @ f.get_transform(theta).get_matrix()
        joint_origin = joint_tf[:, :3, 3]

        axis_local = f.joint.axis.view(1, 3, 1).expand(N, -1, -1)
        axis_world = torch.matmul(joint_tf[:, :3, :3], axis_local).squeeze(-1)

        if f.joint.joint_type == "revolute":
            J[:, :3, joint_idx] = torch.cross(axis_world, target_pos - joint_origin, dim=1)
            J[:, 3:, joint_idx] = axis_world
            joint_idx += 1
        elif f.joint.joint_type == "prismatic":
            J[:, :3, joint_idx] = axis_world
            joint_idx += 1

        world = joint_tf

    if frame == "world":
        return J

    target_rot = ee_rot @ tool_matrix[:, :3, :3]
    rot_t = target_rot.transpose(1, 2)
    J_local = J.clone()
    J_local[:, :3, :] = torch.matmul(rot_t, J[:, :3, :])
    J_local[:, 3:, :] = torch.matmul(rot_t, J[:, 3:, :])
    return J_local
