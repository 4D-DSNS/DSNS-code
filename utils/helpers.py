import numpy as np

def generate_grid(res, mysmall=np.finfo(float).eps):
    theta = np.linspace(mysmall, np.pi - mysmall, res[1])
    phi = np.linspace(mysmall, 2 * np.pi - mysmall, res[0])

    Theta = np.tile(theta, (res[0], 1))
    Phi = np.tile(phi[:, np.newaxis], (1, res[1]))

    return Theta, Phi

def spherical_to_cartesian_numpy(theta, phi, radius=1.0):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.stack([x, y, z], axis=2)

def generate_faces(res):
    faces = []
    for i in range(res[0] - 1):
        for j in range(res[1] - 1):
            # Define the indices of the four vertices of the quadrilateral
            v1 = i * res[1] + j
            v2 = i * res[1] + (j + 1)
            v3 = (i + 1) * res[1] + (j + 1)
            v4 = (i + 1) * res[1] + j

            # Each quadrilateral is split into two triangles
            faces.append([v1, v2, v4])
            faces.append([v2, v3, v4])

    return np.array(faces)

def uv_sphere(res):
    theta, phi = generate_grid(res)
    sv = spherical_to_cartesian_numpy(theta, phi)
    f = generate_faces(res)

    return sv.reshape(-1,3), f