import open3d as o3d
import torch
import os   
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras, PointsRasterizationSettings, PointsRenderer, PointsRasterizer, AlphaCompositor
from tqdm import tqdm


def normalize_verts(verts):
    centroid = np.mean(verts, axis=0)
    verts = verts - centroid
    max_length = np.max(np.sqrt(np.sum(verts ** 2, axis=1)))
    verts_normalized = verts / max_length
    return verts_normalized

# Load ply file using Open3D
def load_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)  # Load the point cloud from .ply file
    verts = np.asarray(pcd.points)  # Extract the points (Nx3)
    colors = np.asarray(pcd.colors)  # Extract the colors (Nx3)

    # Normalize vertices
    verts_normalized = normalize_verts(verts)
    verts_tensor = torch.Tensor(verts_normalized).to(device)
    colors_tensor = torch.Tensor(colors).to(device)

    if torch.sum(colors_tensor > 1) >= 1:
        colors_tensor = colors_tensor / 255
    
    pointcloud = Pointclouds(points=[verts_tensor], features=[colors_tensor])
    return pointcloud

# Define the directories (modify these paths as needed)
data_dir = './database/M-PCCD/distortion/'
output_texture_dir = './database/M-PCCD/proj_6view_512_texture/'
output_depth_dir = './database/M-PCCD/proj_6view_512_depth/'
output_mask_dir = './database/M-PCCD/proj_6view_512_mask/'

os.makedirs(output_texture_dir, exist_ok=True)
os.makedirs(output_depth_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

device = torch.device("cuda:0")

# Views to render
render_view_list = [(0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0)]

# List of .ply files
ply_files = [f for f in os.listdir(data_dir) if f.endswith('.ply')]

for i, ply_file in tqdm(enumerate(ply_files), total=len(ply_files), smoothing=0.9, leave=False):
    obj_file_path = os.path.join(data_dir, ply_file)
    
    # Load point cloud using Open3D
    pointcloud = load_ply(obj_file_path)
    for k, render_view in enumerate(render_view_list):
        R, T = look_at_view_transform(1, render_view[0], render_view[1])
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.003,
            points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        # Generate fragments
        fragments = rasterizer(pointcloud)
        depth = fragments[1].cpu()
        depth = depth[:,:,:,0]

        # Create mask map
        binary_mask = (depth != -1).float()
        binary_mask_data = binary_mask.squeeze().cpu().numpy()
        binary_mask_image = Image.fromarray((binary_mask_data * 255).astype(np.uint8))
        mask_savename = os.path.join(output_mask_dir, f"{ply_file.split('.')[0]}_view_{k}.png")
        binary_mask_image.save(mask_savename)

        # Create depth map
        depth = torch.where(depth == -1, torch.tensor(0.0), depth)
        filtered_depth = depth[depth != 0.0]
        min_depth = torch.min(filtered_depth).cpu().numpy()
        max_depth = torch.max(filtered_depth).cpu().numpy()
        normalized_depth = (depth.squeeze().cpu().numpy() / max_depth * 255).astype(np.uint8)
        depth_image = Image.fromarray(normalized_depth)
        depth_savename = os.path.join(output_depth_dir, f"{ply_file.split('.')[0]}_view_{k}.png")
        depth_image.save(depth_savename)

        # Create texture map
        texture = renderer(pointcloud)
        texture_data = texture[0, ..., :3].cpu().numpy()
        texture_data = (texture_data * 255).astype(np.uint8)
        texture_image = Image.fromarray(texture_data, mode="RGB")
        texture_savename = os.path.join(output_texture_dir, f"{ply_file.split('.')[0]}_view_{k}.png")
        texture_image.save(texture_savename)

