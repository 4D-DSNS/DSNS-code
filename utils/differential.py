# The code is taken from Neural Surface Maps paper. We thank the authors for making the code available.


import torch
from torch.nn import functional as F
from torch import autograd as Grad

class DifferentialMixin:

    # ================================================================== #
    # =================== Compute the gradient ========================= #
    def gradient(self, out, wrt):

        N = out.size(0)
        R = out.size(-1)
        C = wrt.size(-1)

        gradients = []
        for dim in range(R):
            out_p = out[..., dim].flatten()

            select = torch.ones(out_p.size(), dtype=torch.float32).to(out.device)
            # same as select[..., dim] = 1 # compute gradient for x

            gradient = Grad.grad(outputs=out_p, inputs=wrt, grad_outputs=select, create_graph=True)[0]
            gradients.append(gradient)

        J_f_uv = torch.cat(gradients, dim=1).view(N, R, C)
        return J_f_uv

    def backprop(self, out, wrt):

        select = torch.ones(out.size(), dtype=torch.float32).to(out.device)

        J = Grad.grad(outputs=out, inputs=wrt, grad_outputs=select, create_graph=True)[0]
        J = J.view(wrt.size())
        return J

    # ================================================================== #
    # ================ Compute normals using gradient ================== #
    def compute_normals(self, jacobian=None, out=None, wrt=None, return_grad=False):

        if jacobian is None:
            jacobian = self.gradient(out=out, wrt=wrt)

        cross_prod = torch.linalg.cross(jacobian[..., 0], jacobian[..., 1], dim=1)

        # set small normals to zero, happens only when vectors are orthogonal
        idx_small = cross_prod.pow(2).sum(-1) < 10.0**-7
        # normals
        normals = F.normalize(cross_prod, p=2, dim=1, eps=1e-6)  # (N, 3)
        normals[idx_small] = cross_prod[idx_small]

        if return_grad:
            return normals, jacobian
        return normals

