import numpy as np
import scipy.io as sio
import torch
from torch.nn.functional import grid_sample

from .helpers import generate_grid, uv_sphere


def cartesian_to_spherical_single_value(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(x, y)

    if phi < 0:
        phi += 2 * np.pi

    return theta, phi, r

def obtain_rotation(rotation, sphere_level, Theta, Phi):
    output = np.zeros((sphere_level, sphere_level, 3))
    Thetan, Phin = np.zeros((sphere_level, sphere_level)), np.zeros((sphere_level, sphere_level))
    for i in range(sphere_level):
        for j in range(sphere_level):
            x = spherical_to_cartesian_numpy_single(Theta[i, j], Phi[i, j])

            output[i, j] = np.matmul(rotation, x.T)
            Thetan[i, j], Phin[i, j], _ = cartesian_to_spherical_single_value(output[i, j, 0], output[i, j, 1],
                                                                              output[i, j, 2])
            if Phin[i, j] < 0:
                Phin[i, j] = Phin[i, j] + 2 * np.pi
            Thetan = np.clip(Thetan, 0, np.pi)
            Phin = np.clip(Phin, 0, 2 * np.pi)

    gamnew = np.zeros((Thetan.shape[0], Thetan.shape[1], 2))

    Theta_min, Theta_max = Thetan.min(), Thetan.max()
    Phi_min, Phi_max = Phin.min(), Phin.max()
    Theta_norm = 2 * (Thetan - Theta_min) / (Theta_max - Theta_min) - 1
    Phi_norm = 2 * (Phin - Phi_min) / (Phi_max - Phi_min) - 1

    gamnew[:, :, 0] = Theta_norm
    gamnew[:, :, 1] = Phi_norm

    return gamnew


def load_rotation_perturbation(path, sphere_level):
    rotation = sio.loadmat(f"{path}rotation.mat")["tmpoptrot"]
    perturbation = sio.loadmat(f"{path}reparam.mat")["gamcum"]

    Theta, Phi = generate_grid((sphere_level, sphere_level))
    rotation = obtain_rotation(rotation, sphere_level, Theta, Phi)
    rotation = torch.FloatTensor(rotation).expand(3, -1, -1, -1).reshape(-1, sphere_level, sphere_level, 2).cuda()

    perturbation = torch.FloatTensor(perturbation).permute(2, 0, 1).cuda()
    perturbation[0, :, :] = 2 * perturbation[0, :, :] / torch.pi - 1
    perturbation[1, :, :] = 2 * perturbation[1, :, :] / (2 * torch.pi) - 1
    perturbation = torch.nn.functional.interpolate(perturbation.unsqueeze(0), size=(sphere_level, sphere_level),
                                                   mode='bicubic', align_corners=True).squeeze().permute(1, 2,
                                                                                                         0).expand(3,
                                                                                                                   -1,
                                                                                                                   -1,
                                                                                                                   -1).reshape(
        -1, sphere_level, sphere_level, 2)

    return rotation, perturbation

def apply_perturbations_on_sphere(rotation, perturbation, sphere_level):
    sv, f = uv_sphere((sphere_level, sphere_level))
    sv, f = torch.FloatTensor(sv).cuda(), torch.LongTensor(f).cuda()
    sv = apply_gamma_surf_closed_across_all_shapes(sv.unsqueeze(0).reshape(-1, sphere_level, sphere_level, 3).permute(3, 0, 1, 2), rotation)
    sv = apply_gamma_surf_closed_across_all_shapes(sv, perturbation)
    sv = sv.permute(1, 2, 3, 0).reshape(-1, 3)
    return sv, f


def apply_gamma_surf_closed_across_all_shapes(F, gam, size=(32, 32)):
    # Perform interpolation using grid_sample
    return grid_sample(F, gam, mode='bicubic', padding_mode='reflection', align_corners=True)


def spherical_to_cartesian_numpy_single(theta, phi, radius=1.0):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.stack([y, x, z])


def project_surface_to_low_dim(S, Mu, eigenVects):
    n, dim_1, dim_2, _ = S.shape
    resolution = [dim_1, dim_2]
    n_modes = eigenVects.shape[1]

    M = torch.zeros((n_modes, n), device=Mu.device)

    for i in range(n):
        M[:, i], _ = project_to_pca_basis(S[i, :, :, :], Mu, eigenVects, resolution)

    return M


def project_to_pca_basis(S, Mu, eigenVectors, resolution=[256, 256]):
    n_modes = eigenVectors.shape[1]

    # Center the surface data and project to PCA basis
    Cn = torch.matmul(eigenVectors[:, :n_modes].T, (S.T.reshape(-1) - Mu).unsqueeze(1))

    # Reconstruct the surface from the projection
    Sn = reconstruct_surface(Cn, Mu, eigenVectors, resolution)

    return Cn.squeeze(), Sn


def reconstruct_surface(Cn, Mu, eigenVectors, resolution):
    n_modes = len(Cn)
    Sn = Mu.clone()  # Create a clone to avoid modifying the original Mu
    for i in range(n_modes):
        Sn = Sn + Cn[i, :] * eigenVectors[:, i]

    Sn = Sn.reshape(3, resolution[1], resolution[0]).T  # Reshape and transpose
    return Sn


def reconstruct_expression(M, Mu, eigenvects, resolution):
    n = M.shape[1]
    S = torch.zeros((n, resolution[0], resolution[1], 3), device=M.device)

    for i in range(n):
        S[i, :, :, :] = reconstruct_surface(torch.unsqueeze(M[:, i], dim=1), Mu, eigenvects, resolution)

    return S


def compute_gradient(y, dx):
    N = len(y)
    grad = torch.zeros((N), device=y.device)
    grad[0] = (y[1] - y[0]) / dx
    grad[-1] = (y[-1] - y[-2]) / dx
    for i in range(1, N - 1):
        grad[i] = (y[i + 1] - y[i - 1]) / (2 * dx)
    return grad

def inner_prod_q_diff(q1, q2):
    n, T = q1.shape
    dt = 1 / (T - 1)  # Step size for the integral approximation
    val = torch.sum(torch.sum(q1 * q2, dim=0)) * dt
    return val

def curve_to_srvf(p, to_normalize=True, eps=1e-8):
    n, N = p.shape
    v = torch.zeros((n, N), device=p.device)  # Initialize v

    # Compute gradient of each row of p
    for i in range(n):
        v[i, :] = compute_gradient(p[i, :], 1 / N)

    q = torch.zeros((n, N), device=p.device)  # Initialize q
    L = torch.zeros(N, device=p.device)  # Initialize L

    # Compute L and normalize v
    for i in range(N):
        v_matrix = v[:, i].reshape(n, 1)  # Reshape to (n, 1)

        # Compute the Frobenius norm
        L[i] = torch.sqrt(torch.linalg.norm(v_matrix, 'fro'))
        q[:, i] = v[:, i] / (L[i] + eps)

    if to_normalize:
        # Normalize q using the inner product function
        norm_factor = torch.sqrt(inner_prod_q_diff(q, q))
        torch.div(q, norm_factor + eps)

    return q


def cumtrapz(x, y=None, dim=None):
    if y is None:
        y = x.clone()
        x = torch.arange(len(y), dtype=y.dtype, device=y.device)  # Default x to unit spacing if not provided

    if torch.is_tensor(x) and x.numel() == 1:
        x = torch.tensor([0, x.item()], dtype=y.dtype)
        y = torch.cat([torch.zeros(1, y.shape[1], dtype=y.dtype), y], dim=0)
        return x[1] * torch.cumsum((y[:-1, :] + y[1:, :]) / 2, dim=0)

    x = torch.as_tensor(x, dtype=y.dtype, device=x.device)
    y = torch.as_tensor(y, dtype=y.dtype, device=x.device)

    if len(x) != y.shape[0]:
        raise ValueError("Length of x must match the first dimension of y.")

    if dim is None:
        if y.ndim == 1:
            dim = 0
        else:
            dim = torch.argmax(torch.tensor(y.shape)).item()

    if dim >= y.ndim:
        raise ValueError("Dimension exceeds the number of dimensions of y.")

    if x.ndim > 1:
        raise ValueError("x must be a 1D tensor or scalar.")

    dt = torch.diff(x, dim=0) / 2
    if y.ndim == 1:
        integral = torch.cat(
            [torch.tensor([0], dtype=y.dtype, device=x.device), torch.cumsum(dt * (y[:-1] + y[1:]), dim=0)])
    else:
        y = torch.moveaxis(y, dim, 0)
        integral = torch.zeros_like(y)
        for i in range(y.shape[0]):
            integral[i] = torch.cat(
                [torch.tensor([0], dtype=y.dtype), torch.cumsum(dt * (y[i, :-1] + y[i, 1:]), dim=0)])
        integral = torch.moveaxis(integral, 0, dim)

    return integral

def srvf_to_curve(q, trans_vect=None):
    if trans_vect is None:
        trans_vect = torch.zeros((q.shape[0], 1), device=q.device)

    n, T = q.shape
    # Compute the norm of each column in q
    qnorm = torch.linalg.norm(q, dim=0)

    # Initialize p
    p = torch.zeros((n, T), device=q.device)

    # Compute the cumulative trapezoidal integral for each row
    for i in range(n):
        p[i, :] = cumtrapz((q[i, :] * qnorm) / T)

    # Add the translation vector
    p += trans_vect.expand(-1, T)

    return p


def compute_geodesic(M1, M2, do_normalize_scale=False, n_samples=7):
    """
    Compute the geodesic between M1 and M2, assuming q1 and q2 are already registered.

    Parameters:
    M1, M2: Input matrices
    do_normalize_scale: Whether to normalize scale (default is True)
    n_samples: Number of samples to take along the geodesic (default is 7)

    Returns:
    geod: The geodesic curve in the original space, shape (n_samples, size(q1, 0), size(q1, 1))
    """

    q1 = curve_to_srvf(M1, do_normalize_scale)
    q2 = curve_to_srvf(M2, do_normalize_scale)

    geod = torch.zeros((n_samples, q1.shape[0], q1.shape[1])).cuda()

    for i in range(n_samples):
        t = (i) / (n_samples - 1)
        q = (1 - t) * q1 + t * q2

        M = (1 - t) * M1[:, 0] + t * M2[:, 0]  # Geodesic between the first surfaces of both curves

        geod[i, :, :] = srvf_to_curve(q, M)

    return geod

def load_pca_basis(path):
    pca_basis = sio.loadmat(path)
    pca_mean, pca_eigen_vects = torch.tensor(pca_basis["Mu"]).cuda().squeeze(), torch.tensor(pca_basis["eigenVects"]).cuda()
    return pca_mean, pca_eigen_vects