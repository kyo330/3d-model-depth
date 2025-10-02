# Streamlit Depth → Point Cloud (GLPN) app
# -------------------------------------------------------------
# Features
# - Upload a single RGB image
# - Estimate metric-free depth with GLPN (vinvino02/glpn-nyu)
# - Preview depth map
# - Convert depth to a point cloud (approx intrinsics via FOV)
# - Interactive 3D point cloud viewer (Plotly)
# - Download depth (PNG) and point cloud (.ply)
#
# Notes
# - This does **not** need CUDA; it will use MPS/CUDA if present.
# - Without known camera intrinsics, we assume a pinhole camera with a
#   user-selectable field-of-view (FOV). Scale is relative (not metric).
# - Poisson/ball-pivot meshing is optional (can be slow on CPU).
#
# Suggested requirements.txt (put in your repo root):
# --------------------------------------------------
# streamlit>=1.36
# torch
# transformers>=4.44
# pillow
# numpy
# open3d
# plotly
# --------------------------------------------------

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import GLPNForDepthEstimation, GLPNImageProcessor

import plotly.graph_objects as go
import open3d as o3d

st.set_page_config(page_title="RGB → Depth → Point Cloud", layout="wide")
st.title("RGB → Depth → Point Cloud (GLPN)")
st.caption("Single‑image depth estimation to interactive point cloud — built with Streamlit, Transformers, and Open3D")

# ---------- Device selection ----------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
)
st.sidebar.success(f"PyTorch device: {DEVICE}")

# ---------- Cached model/processor ----------
@st.cache_resource(show_spinner=True)
def load_depth_model():
    processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu").to(DEVICE)
    model.eval()
    return processor, model

PROCESSOR, MODEL = load_depth_model()

# ---------- Helpers ----------

def resize_to_multiple_of_32(h: int, w: int, max_h: int = 480):
    new_h = min(h, max_h)
    new_h -= (new_h % 32)
    new_w = int(new_h * w / h)
    return new_h, new_w

@torch.inference_mode()
def estimate_depth(pil_img: Image.Image, max_h: int = 480) -> np.ndarray:
    """Run GLPN and return a depth array (H, W) aligned to input size."""
    # Prepare model input (GLPN favors height <= 480 and divisible by 32)
    H, W = pil_img.height, pil_img.width
    new_h, new_w = resize_to_multiple_of_32(H, W, max_h)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    inputs = PROCESSOR(images=resized, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(DEVICE)

    outputs = MODEL(**inputs)
    # outputs.predicted_depth: (B, 1, h, w)
    pred = outputs.predicted_depth

    # Upsample back to the resized image size (should already match)
    pred_resized = F.interpolate(pred, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Now upsample to original input resolution (H, W)
    pred_full = F.interpolate(pred_resized, size=(H, W), mode="bilinear", align_corners=False)
    depth = pred_full[0, 0].detach().float().cpu().numpy()

    # Normalize for visualization convenience (0-1)
    depth_min, depth_max = float(depth.min()), float(depth.max())
    if depth_max > depth_min:
        depth_vis = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_vis = np.zeros_like(depth)

    return depth, depth_vis


def depth_to_pointcloud(depth: np.ndarray, rgb: np.ndarray, fov_deg: float = 60.0, stride: int = 2):
    """Back-project depth map to a point cloud with an approximate intrinsics.

    Args:
        depth: (H, W) float32, arbitrary units (relative scale)
        rgb: (H, W, 3) uint8
        fov_deg: horizontal field of view approximation
        stride: subsampling factor to keep point counts manageable
    Returns:
        pts: (N, 3) float32 XYZ
        cols: (N, 3) float32 RGB (0-1)
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0

    # Approximate intrinsics from FOV: fx = fy = 0.5*W / tan(FOV/2)
    fov_rad = math.radians(max(1.0, min(179.0, fov_deg)))
    fx = fy = 0.5 * W / math.tan(0.5 * fov_rad)

    # Build pixel grid (subsampled)
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xv, yv = np.meshgrid(xs, ys)

    z = depth[yv, xv].astype(np.float32) + 1e-6  # avoid zero
    x = (xv - cx) * z / fx
    y = (yv - cy) * z / fy

    pts = np.stack([x, -y, -z], axis=-1).reshape(-1, 3)  # OpenGL-ish orientation
    cols = (rgb[yv, xv] / 255.0).reshape(-1, 3).astype(np.float32)

    # Remove NaNs/Infs
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    cols = cols[mask]
    return pts, cols


def save_pointcloud_ply(pts: np.ndarray, cols: np.ndarray) -> bytes:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    with io.BytesIO() as buf:
        o3d.io.write_point_cloud("/tmp/tmp_pc.ply", pcd, write_ascii=True)
        data = Path("/tmp/tmp_pc.ply").read_bytes()
    return data


def plot_pointcloud_plotly(pts: np.ndarray, cols: np.ndarray, point_size: int = 2):
    if pts.size == 0:
        st.warning("Empty point cloud.")
        return
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Plotly wants RGB in 0-255
    rgb255 = (cols * 255).clip(0, 255).astype(np.uint8)
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb255]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=point_size, color=colors, opacity=0.9),
        )
    ])
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=640,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- UI ----------
left, right = st.columns([1, 1])

with st.sidebar:
    st.header("Settings")
    max_h = st.slider("Resize height (multiple of 32)", 128, 768, 480, 32)
    fov = st.slider("Assumed horizontal FOV (°)", 20, 120, 60)
    stride = st.slider("Point cloud stride (downsample)", 1, 10, 2)
    psize = st.slider("Viewer point size", 1, 6, 2)

    st.divider()
    do_mesh = st.checkbox("(Optional) Reconstruct mesh (Poisson) — CPU heavy", value=False)
    depth_contrast = st.checkbox("Improve depth contrast for preview", value=True)

uploaded = st.file_uploader("Upload an RGB image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to begin. A single indoor scene works best.")
    st.stop()

# Load PIL image
img = Image.open(uploaded).convert("RGB")
rgb_np = np.array(img)

with st.spinner("Estimating depth with GLPN…"):
    depth, depth_vis = estimate_depth(img, max_h=max_h)

with left:
    st.subheader("Input & Depth Preview")
    st.image(img, caption=f"Input — {img.width}×{img.height}", use_column_width=True)

    if depth_contrast:
        # Slight gamma for visibility
        vis = np.clip(depth_vis ** 0.7, 0, 1)
    else:
        vis = depth_vis
    st.image((vis * 255).astype(np.uint8), clamp=True, channels="GRAY", caption="Estimated depth (normalized)")

# Save a nice depth PNG to download
buf_depth = io.BytesIO()
Image.fromarray((depth_vis * 255).astype(np.uint8)).save(buf_depth, format="PNG")
buf_depth.seek(0)

with right:
    st.subheader("Point Cloud")
    with st.spinner("Back‑projecting to 3D and rendering…"):
        pts, cols = depth_to_pointcloud(depth, rgb_np, fov_deg=float(fov), stride=int(stride))
        plot_pointcloud_plotly(pts, cols, point_size=int(psize))

# Downloads
st.divider()
left_d, right_d = st.columns(2)
with left_d:
    st.download_button("Download depth (PNG)", data=buf_depth, file_name="depth_normalized.png", mime="image/png")

with right_d:
    ply_bytes = save_pointcloud_ply(pts, cols)
    st.download_button("Download point cloud (.ply)", data=ply_bytes, file_name="pointcloud.ply", mime="application/octet-stream")

# Optional meshing (can be slow)
if do_mesh:
    st.divider()
    st.subheader("Poisson Mesh (optional)")
    with st.spinner("Estimating normals and running Poisson reconstruction… (can take a while)"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(50)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh.compute_vertex_normals()
        # Crop to remove far-away artifacts
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

        # Convert to a small preview via Plotly (triangles)
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        if len(tris) > 0:
            fig_mesh = go.Figure(data=[go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                opacity=1.0,
            )])
            fig_mesh.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=640)
            st.plotly_chart(fig_mesh, use_container_width=True)
        else:
            st.warning("Mesh has no triangles — try different settings or skip meshing.")

        # Download mesh as PLY
        with io.BytesIO() as mesh_buf:
            o3d.io.write_triangle_mesh("/tmp/tmp_mesh.ply", mesh, write_ascii=True)
            mbytes = Path("/tmp/tmp_mesh.ply").read_bytes()
        st.download_button("Download mesh (.ply)", data=mbytes, file_name="mesh.ply", mime="application/octet-stream")

st.caption("Tip: If you deploy on Hugging Face Spaces, set **Hardware: CPU** (works) or **A10G** (faster). If Open3D fails to install on Spaces, pin an older wheel or remove the meshing option.")
