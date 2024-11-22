import fnmatch
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import os
import trimesh
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors

from utils.helpers import uv_sphere


# Function to load meshes from .obj files using Trimesh
def load_mesh(file_path):
    # Load the mesh using Trimesh
    mesh = trimesh.load_mesh(file_path)
    # Extract the vertices and faces from the mesh
    vertices = mesh.vertices
    faces = mesh.faces
    return vertices, faces

def create_mesh(v, f, color = None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()
    if color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    else:
        mesh.paint_uniform_color(np.array([1.0, 0.706, 0.0], dtype=np.float64))

    return mesh

def apply_rotation(mesh, rotation):
    R = mesh.get_rotation_matrix_from_xyz((0, np.deg2rad(rotation), 0))
    mesh.rotate(R, center=mesh.get_center())


def create_video_with_error(mesh_data, output_path, fps=30, duration=5, spacing=0.65):
    translation = np.array([spacing, 0, 0], dtype=np.float64)

    gt_vert, n_vert = np.array(mesh_data['gt_vertices'][0], dtype=np.float64), np.array(mesh_data['neural_vertices'][0],
                                                                                        dtype=np.float64) + translation
    gt_f, n_f = np.array(mesh_data['gt_faces'][0], dtype=np.int32), np.array(mesh_data['neural_faces'][0],
                                                                             dtype=np.int32)

    norm = plt.Normalize(vmin=0, vmax=0.1)
    colors = plt.cm.coolwarm(np.array(mesh_data['error'][0], dtype=np.float64))[:, :3]

    gt_mesh = create_mesh(gt_vert, gt_f)
    n_mesh = create_mesh(n_vert, n_f, color=np.array(colors, dtype=np.float64))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=1920, height=1080)

    vis.add_geometry(gt_mesh)
    vis.add_geometry(n_mesh)

    # Setup view controls
    ctr = vis.get_view_control()
    ctr.set_zoom(np.float64(0.8))
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    fx, fy = 1, 1
    cx, cy = 1920 / 2, 1080 / 2
    camera_params.intrinsic.set_intrinsics(width=1920, height=1080, fx=fx, fy=fy, cx=cx, cy=cy)
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    ctr.change_field_of_view(step=-90)


    # Calculate number of frames
    n_frames = fps * duration
    rotation_step = 4.7 / n_frames

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1061))

    rotation = 0
    # Render frames
    for _ in range(n_frames):
        rotation += rotation_step
        # Rotate the sphere
        apply_rotation(gt_mesh, rotation)
        apply_rotation(n_mesh, rotation)

        # Update geometry
        vis.update_geometry(gt_mesh)
        vis.update_geometry(n_mesh)
        vis.poll_events()
        vis.update_renderer()

        # Capture frame
        image = vis.capture_screen_float_buffer()
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        video.write(image_bgr)

    # Render frames
    for idx in range(len(mesh_data['gt_vertices'])):
        # Remove previous geometries
        vis.remove_geometry(gt_mesh, reset_bounding_box=False)
        vis.remove_geometry(n_mesh, reset_bounding_box=False)

        gt_mesh = create_mesh(np.array(mesh_data['gt_vertices'][idx], dtype=np.float64),
                              np.array(mesh_data['gt_faces'][idx], dtype=np.int32))
        n_mesh = create_mesh(np.array(mesh_data['neural_vertices'][idx], dtype=np.float64) + translation,
                             np.array(mesh_data['neural_faces'][idx], dtype=np.int32),
                             color=plt.cm.coolwarm(norm(np.array(mesh_data['error'][idx], dtype=np.float64)))[:, :3])

        # apply_rotation(gt_mesh, rotation)
        apply_rotation(n_mesh, rotation)

        # Update geometry
        vis.add_geometry(gt_mesh, reset_bounding_box=False)
        vis.add_geometry(n_mesh, reset_bounding_box=False)

        ctr = vis.get_view_control()
        ctr.set_zoom(np.float64(0.8))
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        ctr.change_field_of_view(step=-90)

        vis.poll_events()
        vis.update_renderer()

        # Capture frame
        image = vis.capture_screen_float_buffer()
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        video.write(image_bgr)

    # Clean up
    video.release()
    vis.destroy_window()


def create_video(mesh_data, output_path, registered=False, fps=30, duration=2, spacing=1, repeat=1):
    translation = np.array([spacing, 0, 0], dtype=np.float64)

    if registered:
        source_vert, target_vert = np.array(mesh_data['source'][0], dtype=np.float64), np.array(
            mesh_data['target_reg'][0], dtype=np.float64) + translation
    else:
        source_vert, target_vert = np.array(mesh_data['source'][0], dtype=np.float64), np.array(
            mesh_data['target_unreg'][0], dtype=np.float64) + translation
    f = np.array(mesh_data['faces'][0], dtype=np.int32)

    source_mesh = create_mesh(source_vert, f)
    target_mesh = create_mesh(target_vert, f)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=1920, height=1080)
    vis.add_geometry(source_mesh)
    vis.add_geometry(target_mesh)

    # Set the camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(np.float64(0.8))
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    fx, fy = 1, 1
    cx, cy = 1920 / 2, 1080 / 2
    camera_params.intrinsic.set_intrinsics(width=1920, height=1080, fx=fx, fy=fy, cx=cx, cy=cy)
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    ctr.change_field_of_view(step=-90)

    camera_params = ctr.convert_to_pinhole_camera_parameters()

    render_options = vis.get_render_option()
    render_options.background_color = np.asarray([0.9, 0.9, 0.9])

    # Calculate number of frames
    n_frames = fps * duration
    rotation_step = 1 / n_frames

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1061))

    rotation = 0
    total_rotation = 0
    # Render frames
    for _ in range(n_frames):
        rotation += rotation_step
        total_rotation += rotation
        # Rotate the sphere
        # if i%fps==0:
        apply_rotation(source_mesh, rotation)
        apply_rotation(target_mesh, rotation)

        # Update geometry
        vis.update_geometry(source_mesh)
        vis.update_geometry(target_mesh)
        vis.poll_events()
        vis.update_renderer()

        # Capture frame
        image = vis.capture_screen_float_buffer()
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        video.write(image_bgr)

    # Render frames
    index = 0
    for idx in range((mesh_data['source'].shape[0] - 1) * repeat):
        # Remove previous geometries
        vis.remove_geometry(source_mesh, reset_bounding_box=False)
        vis.remove_geometry(target_mesh, reset_bounding_box=False)

        if idx % repeat == 0:
            source_mesh = create_mesh(np.array(mesh_data['source'][index], dtype=np.float64),
                                      np.array(mesh_data['faces'][index], dtype=np.int32))
            if registered:
                target_mesh = create_mesh(np.array(mesh_data['target_reg'][index], dtype=np.float64) + translation,
                                          np.array(mesh_data['faces'][index], dtype=np.int32))
            else:
                target_mesh = create_mesh(np.array(mesh_data['target_unreg'][index], dtype=np.float64) + translation,
                                          np.array(mesh_data['faces'][index], dtype=np.int32))
            index += 1

        apply_rotation(source_mesh, total_rotation)
        apply_rotation(target_mesh, total_rotation)

        # Update geometry
        vis.add_geometry(source_mesh, reset_bounding_box=False)
        vis.add_geometry(target_mesh, reset_bounding_box=False)

        # Update controls
        ctr = vis.get_view_control()
        ctr.set_zoom(np.float64(0.8))
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        ctr.change_field_of_view(step=-90)

        vis.poll_events()
        vis.update_renderer()

        # Capture frame
        image = vis.capture_screen_float_buffer()
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        video.write(image_bgr)

    # Clean up
    video.release()
    vis.destroy_window()


# Function to compute the error (distance) between two surfaces (meshes)
def compute_surface_error(surface1, surface2):
    # Use NearestNeighbors to find the closest points between the two surfaces
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(surface2)
    distances, _ = nbrs.kneighbors(surface1)
    return distances.flatten()  # Return the distances as a 1D array

def count_files(directory, pattern="orig_*.obj"):
    count = 0
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if filename matches the pattern
            if fnmatch.fnmatch(filename, pattern):
                count += 1
    return count

def visualize_heatmap(exp_path, total_frames):

    mesh_data = {
        "gt_vertices": [],
        "gt_faces": [],
        "neural_vertices": [],
        "neural_faces": [],
        "error": []
    }
    indexes = np.linspace(0, total_frames - 1, total_frames - 1, dtype=np.uint)
    for idx in indexes:
        vertices1, faces1 = load_mesh(f'{exp_path}//orig//orig_{idx}.obj')
        vertices2, faces2 = load_mesh(f'{exp_path}//dsns//dsns_{idx}.obj')
        mesh_data["gt_vertices"].append(vertices1)
        mesh_data["gt_faces"].append(faces1)
        mesh_data["neural_vertices"].append(vertices2)
        mesh_data["neural_faces"].append(faces2)
        mesh_data["error"].append(compute_surface_error(vertices2, vertices1))

    # Generate the video
    output_path = os.path.join(exp_path, f"dsns_quality.mp4")
    create_video_with_error(mesh_data, output_path)
    print(f"Video saved to {output_path}")

def visualize_spatiotemporal(exp_path, registered, total_frames, resolution=256):
    mesh_data = {
        "source": sio.loadmat(f"{exp_path}source.mat")["S"].reshape(total_frames, -1, 3),
        "target_unreg": sio.loadmat(f"{exp_path}target_unreg.mat")["S"].reshape(total_frames, -1, 3),
        "target_reg": sio.loadmat(f"{exp_path}target_reg.mat")["S"].reshape(total_frames, -1, 3),
        "faces": []
    }

    indexes = np.linspace(0, total_frames - 1, total_frames - 1, dtype=np.uint)
    _, f = uv_sphere((resolution, resolution))
    f = f[:, [2, 1, 0]]

    for idx in indexes:
        mesh_data["faces"].append(f)

    # Generate the video
    if registered:
        output_path = os.path.join(exp_path, "registered.mp4")
        create_video(mesh_data, output_path, registered=True)
        print(f"Video saved to {output_path}")
    else:
        output_path = os.path.join(exp_path, "unregistered.mp4")
        create_video(mesh_data, output_path, registered=False)
        print(f"Video saved to {output_path}")


