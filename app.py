import streamlit as st
import sys, os, io, contextlib, importlib.util, tempfile, traceback
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="3D Reconstruction Demo", layout="wide")

st.title("3D Reconstruction — Web App")
st.caption("Auto-wrapped around your existing `main.py`. Upload image(s) and run.")

# --- Load user's main.py dynamically ---
@st.cache_resource
def load_user_module(main_path: str = "main.py"):
    spec = importlib.util.spec_from_file_location("user_main", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_main"] = mod
    spec.loader.exec_module(mod)
    return mod

user_mod = load_user_module(str(Path(__file__).parent / "main.py"))

# Try to find an entrypoint we can call
ENTRYPOINTS = ['pipeline', 'infer', 'run', 'reconstruct', 'predict', 'main']
def pick_entrypoint(mod):
    for name in ENTRYPOINTS:
        fn = getattr(mod, name, None)
        if callable(fn):
            return name, fn
    if callable(getattr(mod, "main", None)):
        return "main", getattr(mod, "main")
    return None, None

entry_name, entry_fn = pick_entrypoint(user_mod)

with st.sidebar:
    st.header("Settings")
    st.write("• Detected entrypoint:", f"`{entry_name}`" if entry_name else ":red[None found]")
    run_btn = st.button("Run", type="primary")
    st.divider()
    st.markdown("**Tips**")
    st.markdown(
        "- If no entrypoint was detected, edit `main.py` to expose a function like\n"
        "  `def pipeline(image_path: str, output_dir: str) -> dict:`\n"
        "  returning paths to outputs (e.g., depth, point cloud, mesh).\n"
        "- You can rename the function to one of the probed names."
    )

tab_in, tab_out, tab_log = st.tabs(["Input", "Outputs", "Logs"])

with tab_in:
    mode = st.radio("Input type", ["Single image", "Multiple images (folder)"], horizontal=True)
    uploads = []
    if mode == "Single image":
        img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file:
            uploads = [img_file]
            st.image(Image.open(img_file), caption=img_file.name, use_column_width=True)
    else:
        files = st.file_uploader("Upload multiple images of the scene", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if files:
            uploads = files
            st.image([Image.open(f) for f in files[:6]], caption=[f.name for f in files[:6]], width=180)

@st.cache_data(show_spinner=False)
def save_uploads(uploads):
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="webapp_"))
    in_dir = work / "inputs"
    out_dir = work / "outputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in uploads:
        p = in_dir / f.name
        with open(p, "wb") as w:
            w.write(f.read())
        paths.append(str(p))
    return str(in_dir), str(out_dir), paths

logs_io = io.StringIO()
def run_with_logs(fn, *args, **kwargs):
    with contextlib.redirect_stdout(logs_io), contextlib.redirect_stderr(logs_io):
        try:
            result = fn(*args, **kwargs)
            return result, None
        except Exception as e:
            traceback.print_exc()
            return None, e

outputs = None
error = None
if run_btn:
    if not uploads:
        st.warning("Please upload at least one image.")
    else:
        in_dir, out_dir, paths = save_uploads(uploads)
        st.toast("Running… This may take a bit depending on model size/CPU/GPU.")
        args_tried = []
        if entry_fn:
            def try_call():
                args_tried.append("fn(image_path, output_dir=...)")
                res, err = run_with_logs(entry_fn, paths[0], output_dir=out_dir)
                if err is None: return res, None
                args_tried.append("fn([image_paths], output_dir=...)")
                res, err = run_with_logs(entry_fn, paths, output_dir=out_dir)
                if err is None: return res, None
                args_tried.append("fn(input_dir=..., output_dir=...)")
                res, err = run_with_logs(entry_fn, input_dir=in_dir, output_dir=out_dir)
                if err is None: return res, None
                args_tried.append("fn(in_dir)")
                res, err = run_with_logs(entry_fn, in_dir)
                if err is None: return res, None
                return None, err
            outputs, error = try_call()
        else:
            import subprocess, sys
            args_tried.append("subprocess: python main.py --input <dir> --output <dir>")
            cmd = [sys.executable, str(Path(__file__).parent / "main.py"), "--input", in_dir, "--output", out_dir]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, check=False)
                logs_io.write(p.stdout + "\n" + p.stderr)
                if p.returncode == 0:
                    outputs = {"output_dir": out_dir}
                else:
                    error = RuntimeError(f"Process returned code {p.returncode}")
            except Exception as e:
                error = e

        with tab_log:
            st.subheader("Execution log")
            st.code(logs_io.getvalue() or "(no logs)")

        if outputs and isinstance(outputs, dict):
            st.session_state.outputs = outputs
        else:
            st.session_state.outputs = {"output_dir": out_dir}
        if error:
            st.error(f"Run failed: {error}")
            with st.expander("Tried calling with these signatures", expanded=False):
                st.markdown("\n".join(f"- `{s}`" for s in args_tried))

with tab_out:
    st.subheader("Artifacts")
    outs = getattr(st.session_state, "outputs", None)
    if not outs:
        st.info("No outputs yet. Upload files and click **Run**.")
    else:
        out_dir = Path(outs.get("output_dir", "")) if isinstance(outs, dict) else None
        candidates = {
            "Depth Image": ["depth.png", "depth.jpg", "predicted_depth.png", "depth_map.png"],
            "Point Cloud": ["points.ply", "point_cloud.ply", "pcd.ply"],
            "Mesh": ["mesh.ply", "mesh.obj", "mesh.glb"]
        }
        if out_dir and out_dir.exists():
            files = list(out_dir.rglob("*"))
            st.write(f"Output directory: `{out_dir}` ({len(files)} files)")
            for label_name, names in candidates.items():
                shown = False
                for n in names:
                    p = out_dir / n
                    if p.exists():
                        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                            st.image(str(p), caption=f"{label_name}: {n}", use_column_width=True)
                        else:
                            with open(p, "rb") as fh:
                                st.download_button(f"Download {label_name} ({n})", fh, file_name=p.name)
                        shown = True
                        break
                if not shown:
                    st.write(f"· No {label_name} found yet.")
            with st.expander("Browse all files", expanded=False):
                for f in files:
                    st.write(f"`{f.relative_to(out_dir)}`")
        else:
            for k, v in outs.items():
                if isinstance(v, str) and os.path.exists(v) and v.lower().endswith((".png",".jpg",".jpeg")):
                    st.image(v, caption=k, use_column_width=True)
                elif isinstance(v, str) and os.path.exists(v):
                    with open(v, "rb") as fh:
                        st.download_button(f"Download {k}", fh, file_name=os.path.basename(v))
                else:
                    st.write(f"**{k}:** {v}")
