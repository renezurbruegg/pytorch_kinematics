"""Batched forward kinematics for a `Chain`.

WHY THIS EXISTS. `Chain.forward_kinematics` used to walk the chain in
Python, one frame at a time, building a `Transform3d` per joint and
composing a 4x4 into it. On a 60-frame humanoid upper body that is 60
frames each issuing a handful of tiny CUDA kernels, and it measured the
same 16 ms whether the batch was 1 pose or 512 -- the GPU sat idle and
the time was Python plus kernel-launch latency, so no amount of batch
bought it back. Grasp synthesis calls FK several times per optimizer
step, which made it the largest single cost of a fit.

WHY THIS EXISTS. `Chain.forward_kinematics` walks the chain in Python,
one frame at a time, building a `Transform3d` per joint and composing a
4x4 into it. On the G1 upper body that is 49 frames each issuing a
handful of tiny CUDA kernels, and it costs the same 17 ms whether the
batch is 1 grasp or 512 -- measured. That flatness is the whole
diagnosis: the GPU is idle and the time is Python plus kernel-launch
latency, so no amount of batch buys it back. The anneal loop calls FK
three times per step, which made FK the single largest line in a fit.

WHAT IS DIFFERENT HERE. The same maths, issued as O(depth) kernels
instead of O(frames):

  * every frame's local transform is built in ONE batched Rodrigues
    over (B, F, 3, 3) -- fixed and prismatic frames fall out of it for
    free, since their rotation angle is zero and sin(0) = 0
  * the tree is then composed one DEPTH LEVEL at a time, all frames at
    that level in a single (B, n, 4, 4) matmul. The G1 upper body is 12
    levels deep, so 12 matmuls replace 49 sequential composes.

It is differentiable (index_put and matmul both are), returns the same
`{link_name: Transform3d}` dict the chain returns, and `verify()` below
checks it against the chain frame by frame, gradients included.
"""

from __future__ import annotations

from typing import Dict, List


class BatchedFK:
    """`chain.forward_kinematics`, issued as O(tree depth) kernels.

    Topology, joint axes and the static joint/link offsets are read off
    the chain ONCE at construction; only the joint angles move per call.
    """

    def __init__(self, chain):
        import torch

        self.chain = chain
        dev, dt = chain.device, chain.dtype
        self.device, self.dtype = dev, dt
        frames: List = []
        parent: List[int] = []
        depth: List[int] = []

        def walk(f, par, d):
            i = len(frames)
            frames.append(f)
            parent.append(par)
            depth.append(d)
            for c in f.children:
                walk(c, i, d + 1)

        walk(chain._root, -1, 0)
        self.frames = frames
        self.link_names = [f.link.name for f in frames]
        self.rows = {n: i for i, n in enumerate(self.link_names)}
        if len(set(self.link_names)) != len(self.link_names):
            raise ValueError("chain has duplicate link names, so a dict "
                             "keyed on them cannot describe it")
        F = len(frames)
        self.n_frames = F

        # --- statics: the offsets and axes that never move -------------
        joint_off = torch.stack(
            [f.joint.offset.get_matrix()[0] for f in frames])
        link_off = torch.stack([f.link.offset.get_matrix()[0] for f in frames])
        axis = torch.stack([f.joint.axis.to(dtype=dt) for f in frames])
        self.joint_off = joint_off.to(device=dev, dtype=dt)
        self.link_off = link_off.to(device=dev, dtype=dt)
        self.axis = axis.to(device=dev)

        # skew(axis) and its square, so Rodrigues is two multiply-adds
        K = torch.zeros(F, 3, 3, device=dev, dtype=dt)
        K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
        K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
        self.K = K
        self.K2 = K @ K
        self.eye3 = torch.eye(3, device=dev, dtype=dt).expand(F, 3, 3)
        bottom = torch.zeros(F, 1, 4, device=dev, dtype=dt)
        bottom[:, 0, 3] = 1.0
        self.bottom = bottom

        # --- which column of `th` drives which frame ------------------
        names = chain.get_joint_parameter_names()
        col = {n: i for i, n in enumerate(names)}
        rev_f, rev_c, pri_f, pri_c = [], [], [], []
        for i, f in enumerate(frames):
            j = col.get(f.joint.name)
            if j is None or f.joint.joint_type == "fixed":
                continue
            if f.joint.joint_type == "revolute":
                rev_f.append(i); rev_c.append(j)
            elif f.joint.joint_type == "prismatic":
                pri_f.append(i); pri_c.append(j)
            else:
                raise ValueError(f"unsupported joint {f.joint.joint_type}")
        L = torch.long
        self.rev_f = torch.tensor(rev_f, dtype=L, device=dev)
        self.rev_c = torch.tensor(rev_c, dtype=L, device=dev)
        self.pri_f = torch.tensor(pri_f, dtype=L, device=dev)
        self.pri_c = torch.tensor(pri_c, dtype=L, device=dev)
        self.n_joints = len(names)

        self.depth = max(depth) + 1

        # --- ancestor tables, for composing the tree by DOUBLING ------
        # Composing one tree level at a time costs a gather, a matmul and
        # a scatter per level -- 16 levels of that is ~45 autograd nodes,
        # and the loop is launch-bound in both directions. Pointer
        # doubling composes 2^k locals per round instead, so the same
        # product takes ceil(log2(depth)) rounds: 4 here, not 16.
        #
        # Row F is a virtual identity: a path that runs past the root
        # lands on it, which is what makes every node's product the same
        # length without a per-node branch.
        anc = [parent[i] if parent[i] >= 0 else F for i in range(F)] + [F]
        self.anc = []
        cur = anc
        span = 1
        while span < self.depth:
            self.anc.append(torch.tensor(cur, dtype=L, device=dev))
            cur = [cur[i] for i in cur]
            span *= 2
        self.eye4 = torch.eye(4, device=dev, dtype=dt).view(1, 1, 4, 4)

    def local(self, th):
        """Every frame's joint-local transform, in one batched op."""
        import torch

        B = th.shape[0]
        F = self.n_frames
        ang = th.new_zeros(B, F)
        if self.rev_f.numel():
            ang = ang.index_copy(1, self.rev_f,
                                 th.index_select(1, self.rev_c))
        slide = th.new_zeros(B, F)
        if self.pri_f.numel():
            slide = slide.index_copy(1, self.pri_f,
                                     th.index_select(1, self.pri_c))
        s = torch.sin(ang)[..., None, None]
        c = torch.cos(ang)[..., None, None]
        R = self.eye3 + s * self.K + (1.0 - c) * self.K2       # (B,F,3,3)
        p = slide[..., None] * self.axis                        # (B,F,3)
        M = torch.cat([torch.cat([R, p[..., None]], dim=-1),
                       self.bottom.expand(B, F, 1, 4)], dim=-2)
        return self.joint_off @ M                               # (B,F,4,4)

    def world(self, th):
        """(B, F, 4, 4) link transforms, composed by pointer doubling.

        After round k every row holds the product of the 2^(k+1) local
        transforms ending at that frame, so ceil(log2(depth)) rounds
        leave each row holding the whole root-to-frame product. Matrix
        multiplication is associative, so this is the same value the
        level-by-level walk produced, in log rounds instead of linear
        ones -- which is what matters when each round is a kernel launch
        and an autograd node rather than arithmetic.
        """
        import torch

        L = self.local(th)
        P = torch.cat([L, self.eye4.expand(L.shape[0], 1, 4, 4)], dim=1)
        for anc in self.anc:
            P = P.index_select(1, anc) @ P
        return P[:, :self.n_frames] @ self.link_off

    def __call__(self, th) -> Dict:
        import pytorch_kinematics.transforms as tf

        if th.shape[1] != self.n_joints:
            raise ValueError(f"expected {self.n_joints} joint angles, "
                             f"got {th.shape[1]}")
        W = self.world(th)
        out = LinkTransforms((n, tf.Transform3d(matrix=W[:, i]))
                             for i, n in enumerate(self.link_names))
        out.matrices = W
        out.rows = self.rows
        return out


class LinkTransforms(dict):
    """The FK result: a `{link: Transform3d}` dict, plus the raw stack.

    Every transform in the dict is a view of one `(B, F, 4, 4)` tensor,
    so a consumer that wants MANY links at once -- the hand model stacks
    60 of them on every distance query -- can gather from `matrices`
    with `rows` instead of cloning each `Transform3d`'s matrix. Ordinary
    dict consumers are unaffected and need not know it is here.
    """

    __slots__ = ("matrices", "rows")


def verify(chain, batch: int = 4, seed: int = 0, tol: float = 2e-5) -> dict:
    """Check the batched FK against the chain it replaces.

    Poses AND gradients, because the fit backpropagates through FK: a
    forward-only check would pass on an implementation whose gradient is
    silently wrong, which is exactly the failure that would show up as a
    fit that no longer converges.

    The reference is the RECURSIVE walk, `Chain._forward_kinematics`,
    reached directly. Never `chain.forward_kinematics`: that is the
    batched implementation, so comparing the two would compare this code
    with itself and pass on anything.
    """
    import torch

    def reference(th):
        jn = chain.get_joint_parameter_names()
        th_dict = {j: th[:, i] for i, j in enumerate(jn)}
        return chain._forward_kinematics(chain._root, th_dict,
                                         chain._fk_world(), "WORLD")

    fk = BatchedFK(chain)
    g = torch.Generator(device="cpu").manual_seed(seed)
    th = (torch.rand(batch, fk.n_joints, generator=g) * 2 - 1).to(
        device=chain.device, dtype=chain.dtype)

    ref = reference(th)
    got = fk(th)
    if set(ref) - set(got):
        raise AssertionError(f"missing links: {sorted(set(ref) - set(got))}")
    worst, worst_link = 0.0, ""
    for k in ref:
        d = float((ref[k].get_matrix() - got[k].get_matrix()).abs().max())
        if d > worst:
            worst, worst_link = d, k

    def grad_of(fn):
        t = th.clone().requires_grad_(True)
        st = fn(t)
        loss = sum(v.get_matrix()[:, :3, 3].pow(2).sum() for v in st.values())
        loss.backward()
        return t.grad

    gr = grad_of(reference)
    gg = grad_of(fk)
    gdiff = float((gr - gg).abs().max())
    out = {"links": len(ref), "pose_max_abs": worst, "worst_link": worst_link,
           "grad_max_abs": gdiff, "depth": fk.depth, "frames": fk.n_frames}
    if worst > tol or gdiff > 1e-3:
        raise AssertionError(f"batched FK disagrees with the chain: {out}")
    return out
