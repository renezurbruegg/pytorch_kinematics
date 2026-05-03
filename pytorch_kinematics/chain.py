"""Kinematic chain classes and utilities for forward/inverse kinematics computation."""

from typing import Dict, List, Optional, Tuple, Union
import torch
from . import jacobian
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev


def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.
    
    The skew-symmetric matrix [v]_x is a 3x3 matrix such that [v]_x @ u = v × u 
    for any vector u. This representation is useful for efficient computation of 
    cross products in robotics and rigid body dynamics.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

    Raises:
        ValueError: If input tensor is not of shape (..., 3).
        
    Examples:
        >>> vec = torch.tensor([1.0, 2.0, 3.0])
        >>> skew = skew_symmetric_matrix(vec)
        >>> skew.shape
        torch.Size([1, 3, 3])
    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    shape = vec.shape[:-1]
    length = torch.prod(torch.tensor(shape)).item()
    # create a skew-symmetric matrix
    skew_sym_mat = torch.zeros(length, 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[..., 0, 1] = -vec[..., 2]
    skew_sym_mat[..., 0, 2] = vec[..., 1]
    skew_sym_mat[..., 1, 2] = -vec[..., 0]
    skew_sym_mat[..., 1, 0] = vec[..., 2]
    skew_sym_mat[..., 2, 0] = -vec[..., 1]
    skew_sym_mat[..., 2, 1] = vec[..., 0]
    skew_sym_mat = skew_sym_mat.view(*shape, 3, 3)

    return skew_sym_mat


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) == 0:
        N = 1
        th = th.view(1, 1)
    elif len(th.shape) == 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    return th, N


class Chain(object):
    """A kinematic chain representing a robot or articulated mechanism.
    
    A Chain represents a kinematic structure defined by a tree of links and joints.
    It provides methods for forward kinematics, Jacobian computation, and other
    kinematic operations.
    
    Attributes:
        dtype: PyTorch dtype for all tensors in the chain (default: torch.float32).
        device: PyTorch device for computation (default: "cpu").
    """
    
    def __init__(self, root_frame, dtype: torch.dtype = torch.float32, 
                 device: Union[str, torch.device] = "cpu") -> None:
        """Initialize a Chain from a root frame.
        
        Args:
            root_frame: The root frame of the kinematic chain.
            dtype: Data type for all tensors (default: torch.float32).
            device: Device for computation (default: "cpu").
        """
        self._root = root_frame
        self.dtype = dtype
        self.device = device
        self._frame_names: List[str] = []

        def _load_frames_recursive(frame) -> None:
            self._frame_names.append(frame.link.name)
            for child in frame.children:
                _load_frames_recursive(child)

        _load_frames_recursive(self._root)

    def to(self, dtype: Optional[torch.dtype] = None, 
            device: Optional[Union[str, torch.device]] = None) -> "Chain":
        """Move chain to specified device and/or dtype.
        
        Args:
            dtype: Target dtype (e.g., torch.float32). If None, dtype is unchanged.
            device: Target device (e.g., "cpu" or "cuda:0"). If None, device is unchanged.
            
        Returns:
            Self for method chaining.
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)
        return self

    @property
    def tails(self) -> List:
        """Get all end-effector (leaf) frames in the chain.
        
        Returns:
            List of tail frames (frames with no children).
        """
        root = self._root
        tails = []
        children = [*root.children]
        while len(children) > 0:
            child = children.pop()
            if child.children is None or len(child.children) == 0:
                tails.append(child)
            else:
                children.extend(child.children)
        return tails

    def __str__(self) -> str:
        """Return string representation of the chain."""
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name: str, frame) -> Optional['Frame']:
        """Recursively search for a frame by name.
        
        Args:
            name: Frame name to search for.
            frame: Current frame to search from.
            
        Returns:
            The frame object if found, None otherwise.
        """
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name: str):
        """Find a frame by name in the chain.
        
        Args:
            name: Name of the frame to find.
            
        Returns:
            The frame object if found, None otherwise.
        """
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name: str, frame):
        """Recursively search for a link by name.
        
        Args:
            name: Link name to search for.
            frame: Current frame to search from.
            
        Returns:
            The link object if found, None otherwise.
        """
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name: str):
        """Find a link by name in the chain.
        
        Args:
            name: Name of the link to find.
            
        Returns:
            The link object if found, None otherwise.
        """
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    @staticmethod
    def _get_joint_parameter_names(frame, exclude_fixed: bool = True) -> List[str]:
        """Recursively get joint parameter names.
        
        Args:
            frame: Current frame in the chain.
            exclude_fixed: If True, exclude fixed joints from the list.
            
        Returns:
            List of joint parameter names.
        """
        joint_names: List[str] = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.joint.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parameter_names(child, exclude_fixed))
        return joint_names

    def get_joint_parameter_names(self, exclude_fixed: bool = True) -> List[str]:
        """Get all joint parameter names in the chain.
        
        Args:
            exclude_fixed: If True, exclude fixed joints from the list (default: True).
            
        Returns:
            Sorted list of unique joint parameter names.
        """
        names = self._get_joint_parameter_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    def add_frame(self, frame, parent_name: str) -> None:
        """Add a new frame to the chain.
        
        Args:
            frame: Frame to add.
            parent_name: Name of the parent frame.
        """
        frame = self.find_frame(parent_name)
        if not frame is None:
            frame.add_child(frame)

    @staticmethod
    def _forward_kinematics(root, th_dict: Dict[str, torch.Tensor], 
                           world: tf.Transform3d = None, 
                           parent: str = "") -> Dict[str, tf.Transform3d]:
        """Compute forward kinematics recursively for all links.
        
        Args:
            root: Root frame of the kinematic chain.
            th_dict: Dictionary mapping joint names to joint angles (batch_size, 1).
            world: World/parent frame transformation.
            parent: Name of parent frame (for bookkeeping).
            
        Returns:
            Dictionary mapping link names to their world frame transformations.
        """
        if world is None:
            world = tf.Transform3d()
        link_transforms: Dict[str, tf.Transform3d] = {}

        th, N = ensure_2d_tensor(th_dict.get(root.joint.name, 0.0), world.dtype, world.device)
        trans = world.compose(root.get_transform(th.view(N, 1)))
        link_transforms[root.link.name] = trans.compose(root.link.offset)

        for child in root.children:
            link_transforms.update(Chain._forward_kinematics(child, th_dict, trans, root.name))
        return link_transforms

    def forward_kinematics(self, th: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                          world: tf.Transform3d = None) -> Dict[str, tf.Transform3d]:
        """Compute forward kinematics for all links.
        
        Args:
            th: Joint angles as tensor (batch_size, num_joints) or dict mapping joint names to angles.
            world: World frame transformation (default: identity).
            
        Returns:
            Dictionary mapping link names to their world frame Transform3d objects.
            
        Raises:
            ValueError: If the number of joint angles doesn't match expected number.
        """
        if world is None:
            world = tf.Transform3d()
        if not isinstance(th, dict):
            jn = self.get_joint_parameter_names()
            if len(jn) != th.shape[1]:
                raise ValueError("Invalid number of joint parameters.", "Expected %d, got %d." % (len(jn), th.shape[1]))
            assert len(jn) == th.shape[1]
            th_dict = dict((j, th[:, i]) for i, j in enumerate(jn))
        else:
            th_dict = th
        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        return self._forward_kinematics(self._root, th_dict, world, "WORLD")

    @staticmethod
    def _fk_vectorized(root, theta: torch.Tensor, theta_idx: int, 
                      world: tf.Transform3d) -> Tuple[List[torch.Tensor], int]:
        """Compute vectorized forward kinematics recursively.
        
        Args:
            root: Root frame of the kinematic chain.
            theta: Joint angles tensor with shape (batch_size, num_joints).
            theta_idx: Current index in the joint angles array.
            world: World frame transformation.
            
        Returns:
            Tuple of:
                - List of link poses as (pos, quat) pairs for each frame
                - Updated theta index
        """
        batch_size = theta.shape[0]

        if root.joint.joint_type != "fixed":
            link_theta = theta[..., theta_idx].view(-1, 1)
            theta_idx += 1
        else:
            link_theta = torch.zeros(1, dtype=theta.dtype, device=theta.device).view(-1, 1)

        trans = world.compose(root.get_transform(link_theta))

        pose = trans.compose(root.link.offset).get_matrix()
        q_wxyz = tf.matrix_to_quaternion(pose[:, :3, :3])
        pose = torch.cat([pose[:, :3, -1], q_wxyz], dim=1)
        pose = pose.expand(batch_size, -1)

        frame_poses = [pose]

        if root.children is not None:
            for child in root.children:
                poses, theta_idx = Chain._fk_vectorized(child, theta, theta_idx, trans)
                frame_poses += poses

        return frame_poses, theta_idx

    @staticmethod
    def _fk_vectorized_jac(root, theta: torch.Tensor, theta_idx: int, 
                          world: tf.Transform3d, 
                          parent_jacobian: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """Compute Jacobian for each frame using vectorized operations.
        
        Args:
            root: Root frame of the kinematic chain.
            theta: Joint angles tensor with shape (batch_size, num_joints).
            theta_idx: Current index in the joint angles array.
            world: World frame transformation.
            parent_jacobian: Jacobian from parent frame (batch_size, 6, num_joints).
            
        Returns:
            Tuple of:
                - List of Jacobians for each frame
                - Updated theta index
        """
        child_jacobian = parent_jacobian.clone()
        if root.joint.joint_type != "fixed":
            link_theta = theta[..., theta_idx].view(-1, 1)
            theta_idx += 1
            # axis = (roma.unitquat_to_rotmat(world.get_matrix()[:, :3, :3]) @ axis.unsqueeze(0).unsqueeze(-1)).squeeze(-1)s
        else:
            link_theta = torch.zeros(1, dtype=theta.dtype, device=theta.device).view(-1, 1)

        current_link_tf_local = root.get_transform(link_theta)
        current_link_tf_world = world.compose(current_link_tf_local)
        offset_world = current_link_tf_local.transform_normals(root.link.offset.get_matrix()[..., :3, -1]).squeeze(1)

        current_link_offset = world.transform_normals(current_link_tf_local.get_matrix()[..., :3, -1].unsqueeze(1)).squeeze(1)
        if current_link_offset.ndim == 2:
            current_link_offset = current_link_offset.unsqueeze(-1)

        child_jacobian[..., :3, :] = parent_jacobian[..., :3, :] + parent_jacobian[..., 3:, :].cross(current_link_offset, dim=-2)

        if root.joint.joint_type != "fixed":
            # convert into current link frame
            axis = current_link_tf_world.transform_normals(root.joint.axis.unsqueeze(0)).squeeze(1)
            if root.joint.joint_type == "revolute":
                child_jacobian[..., :3, theta_idx - 1] += axis.cross(offset_world, dim=-1)
                child_jacobian[..., 3:, theta_idx - 1] += axis
            elif root.joint.joint_type == "prismatic":
                child_jacobian[..., :3, theta_idx - 1] += axis

        jacobians = [child_jacobian]
        if root.children is not None:
            for child in root.children:
                child_jacs, theta_idx = Chain._fk_vectorized_jac(child, theta, theta_idx, current_link_tf_world, parent_jacobian=child_jacobian)
                jacobians += child_jacs

        return jacobians, theta_idx

    def jac_vectorized(self, th: torch.Tensor) -> torch.Tensor:
        """Compute vectorized Jacobian matrix for the chain.
        
        Returns the Jacobian (6 x num_joints) relating end-effector velocities to joint velocities.
        Top 3 rows are linear velocity jacobian, bottom 3 rows are angular velocity jacobian.

        Args:
            th: Joint angles tensor with shape (batch_size, num_joints) or (num_joints,).

        Returns:
            Jacobian tensor with shape (batch_size, 6, num_joints) or (6, num_joints).
        """
        squeeze = False
        if th.ndim == 1:
            th = th.unsqueeze(0)
            squeeze = True
        jacobian_mat = torch.zeros(th.shape[0], 6, th.shape[-1], dtype=th.dtype, device=th.device)
        data = self._fk_vectorized_jac(self._root, th, theta_idx=0, world=tf.Transform3d(device=self.device), parent_jacobian=jacobian_mat)[0]
        data = torch.stack(data, dim=1)
        if squeeze:
            data = data.squeeze(0)
        return data

    def fk_vectorized(self, th: torch.Tensor) -> torch.Tensor:
        """Compute vectorized forward kinematics for all frames.
        
        Args:
            th: Joint angles tensor with shape (batch_size, num_joints) or (num_joints,).

        Returns:
            Pose tensor with shape (batch_size, num_frames, 7) or (num_frames, 7),
            where each pose is [x, y, z, qw, qx, qy, qz].
        """
        squeeze = False
        if th.ndim == 1:
            th = th.unsqueeze(0)
            squeeze = True
        data = self._fk_vectorized(self._root, th, theta_idx=0, world=tf.Transform3d(device=self.device))[0]
        data = torch.stack(data, dim=1)
        if squeeze:
            data = data.squeeze(0)
        return data

    def jacobian(self, th: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matrix.
        
        Args:
            th: Joint angles.

        Returns:
            Jacobian matrix.
        """
        jac = self.jac_vectorized(th)
        return jac

    def analytical_jacobian(self, th: torch.Tensor, analytical: bool = False) -> torch.Tensor:
        """Compute analytical or geometric Jacobian using automatic differentiation.
        
        Args:
            th: Joint angles tensor with shape (batch_size, num_joints) or (num_joints,).
            analytical: If True, return analytical Jacobian. If False, convert to geometric Jacobian
                       relating end-effector twist to joint velocities.

        Returns:
            Jacobian tensor with shape (batch_size, 6, num_joints) or (6, num_joints).
        """
        squeeze = False
        if th.ndim == 1:
            th = th.unsqueeze(0)
            squeeze = True

        J = vmap(jacrev(self.fk_vectorized))(th + 1e-4)
        if analytical:
            return J if not squeeze else J.squeeze(0)

        fk = self.fk_vectorized(th)
        quat_wxyz = fk[..., 3:]
        # convert angular quaternion to angular velocity at the end effector
        quat_wxyz_dot = J[..., 3:, :]
        q0, q1, q2, q3 = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
        H_X_EE = torch.stack(
            [
                torch.stack([-q1, q0, -q3, q2], dim=-1),
                torch.stack([-q2, q3, q0, -q1], dim=-1),
                torch.stack([-q3, -q2, q1, q0], dim=-1),
            ],
            dim=-2,
        )

        ang_vel = 2 * (H_X_EE @ quat_wxyz_dot)

        J_geom = torch.cat([J[..., :3, :], ang_vel], dim=-2)
        # Replace NaN with zeros
        J_geom[torch.isnan(J_geom)] = 0.0
        J = J_geom
        if squeeze:
            J = J.squeeze(0)
        return J


class SerialChain(Chain):
    """A serial kinematic chain (tree-like structure with a single end-effector).
    
    A SerialChain is a special case of Chain where there is a unique path from
    the root to the end-effector frame. This is common in robot manipulators.
    """
    
    def __init__(self, chain: Chain, end_frame_name: str, 
                 root_frame_name: str = "", **kwargs) -> None:
        """Initialize a SerialChain from a Chain object.
        
        Args:
            chain: The chain to extract the serial structure from.
            end_frame_name: Name of the end-effector frame.
            root_frame_name: Name of the root frame (default: "" uses the chain's root).
            **kwargs: Additional arguments passed to the parent Chain class.
            
        Raises:
            ValueError: If root_frame_name or end_frame_name is not valid.
        """
        if root_frame_name == "":
            super(SerialChain, self).__init__(chain._root, **kwargs)
        else:
            super(SerialChain, self).__init__(chain.find_frame(root_frame_name), **kwargs)
            if self._root is None:
                raise ValueError("Invalid root frame name %s." % root_frame_name)
        self._serial_frames = self._generate_serial_chain_recurse(self._root, end_frame_name)
        if self._serial_frames is None:
            raise ValueError("Invalid end frame name %s." % end_frame_name)

    @staticmethod
    def _generate_serial_chain_recurse(root_frame, end_frame_name):
        for child in root_frame.children:
            if child.name == end_frame_name:
                return [child]
            else:
                frames = SerialChain._generate_serial_chain_recurse(child, end_frame_name)
                if not frames is None:
                    return [child] + frames
        return None

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = []
        for f in self._serial_frames:
            if exclude_fixed and f.joint.joint_type == "fixed":
                continue
            names.append(f.joint.name)
        return names

    def forward_kinematics(self, th: torch.Tensor, 
                          world: tf.Transform3d = None,
                          end_only: bool = True) -> Union[tf.Transform3d, Dict[str, tf.Transform3d]]:
        """Compute forward kinematics for the serial chain.
        
        Args:
            th: Joint angles tensor with shape (batch_size, num_joints) or (num_joints,).
            world: World frame transformation (default: identity).
            end_only: If True, return only end-effector transform. If False, return transforms for all links.
            
        Returns:
            If end_only=True: Transform3d object for end-effector.
            If end_only=False: Dictionary mapping link names to Transform3d objects.
        """
        if world is None:
            world = tf.Transform3d()
        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        th, N = ensure_2d_tensor(th, self.dtype, self.device)

        cnt = 0
        link_transforms: Dict[str, tf.Transform3d] = {}
        trans = tf.Transform3d(matrix=world.get_matrix().repeat(N, 1, 1))
        for f in self._serial_frames:
            angle = th[:, cnt] if f.joint.joint_type != "fixed" else torch.zeros(N, dtype=self.dtype, device=self.device)
            trans = trans.compose(f.get_transform(angle.view(N, 1)))
            link_transforms[f.link.name] = trans.compose(f.link.offset)
            if f.joint.joint_type != "fixed":
                cnt += 1
        return link_transforms[self._serial_frames[-1].link.name] if end_only else link_transforms

    def jacobian(
        self,
        th: torch.Tensor,
        locations: Optional[torch.Tensor] = None,
        frame: str = "world",
    ) -> torch.Tensor:
        """Compute Jacobian matrix.
        
        Args:
            th: Joint angles.
            locations: Optional transformation for the tool frame.
            frame: Twist expression frame. "world" returns the Jacobian in the
                base/world frame, while "local" returns it in the end-effector/tool frame.

        Returns:
            Jacobian matrix.
        """
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian(self, th, tool=locations, frame=frame)
