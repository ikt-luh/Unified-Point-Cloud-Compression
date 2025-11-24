import os, glob, math, subprocess, tempfile
from pathlib import Path
from typing import List, Tuple
import yaml

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import open3d as o3d

import utils
from model.model import UnifiedModel

INPUT_DIR = "./data/datasets/raw/CasualSpin/CasualSpin/ply_vox10/ply_xyz_rgb/"
OUTPUT_VIDEO = "./results/q_sweep.mp4"
OUTPUT_FPS = 30

DEVICE_ID = 0
SCALING_FACTOR = 1
BLOCK_SIZE = 1024
POINT_SIZE = 0.1

MAX_FRAMES = 1260

# Model + weights
EXPERIMENT = "Main"
BASE_PATH = "./results"
WEIGHT_PATH = os.path.join(BASE_PATH, EXPERIMENT, "weights.pt")
CONFIG_PATH = os.path.join(BASE_PATH, EXPERIMENT, "config.yaml")

# Rendering
RENDERS_DIR = os.path.join(BASE_PATH, EXPERIMENT, "renders_q_sweep")
FRAME_DIRECTION="top"
RENDER_SIZE = (1280, 720) 

def draw_slider(draw: ImageDraw.Draw, origin: Tuple[int,int], length: int, value: float, label: str, font):
    x, y = origin
    h = 16
    # rail
    draw.rounded_rectangle([x, y, x+length, y+h], radius=8, outline=(255,255,255,180), width=2, fill=(0,0,0,90))
    # knob
    knob_x = x + int(np.clip(value, 0.0, 1.0) * length)
    draw.ellipse([knob_x-4, y-6, knob_x+4, y+h+6], fill=(255,255,255,220))
    # label
    draw.text((x, y-5), f"{label}: {value:.2f}", fill=(255,255,255,220), anchor="lb", font=font)

def overlay_hud(img: Image.Image, q_a: float, q_g: float, bpp: float, frame_idx: int):
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # background panel
    pad = 16
    panel_w = min(520, w-2*pad)
    panel_h = 180
    x0 = pad
    y0 = h - panel_h - pad
    draw.rounded_rectangle([x0, y0, x0+panel_w, y0+panel_h], radius=16, fill=(0,0,0,110), outline=(255,255,255,60), width=1)

    # text
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
        bigfont = ImageFont.truetype("DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()
        bigfont = font

    draw.text((x0+22, y0+14), f"bpp: {bpp:.3f}", font=bigfont, fill=(255,255,255,230))

    # sliders
    slider_x = x0 + 18
    slider_w = panel_w - 36
    draw_slider(draw, (slider_x, y0+14+70), slider_w, q_a, "Attribute Quality", font)
    draw_slider(draw, (slider_x, y0+14+70+60), slider_w, q_g, "Geometry Quality", font)

    return img

def generate_q_schedule(n: int = 630) -> List[Tuple[float, float]]:
    """
    Fixed 630-frame quality schedule:
      A) hold 30 @ (1,1)
      B) both 1->0 over 90
      C) q_a 0->1 over 90 (q_g=0)
      D) q_a 1->0 over 90 (q_g=0)
      E) hold 30 @ (0,0)
      F) q_g 0->1 over 90 (q_a=0)
      G) q_g 1->0 over 90 (q_a=0)
      H) both 0->1 over 90
      I) hold 30 @ (1,1)
    Total = 1260 frames.
    """
    assert n == 1260, f"This schedule is defined for exactly 630 frames (got n={n})"

    def hold(a: float, g: float, L: int) -> List[Tuple[float, float]]:
        return [(a, g)] * L

    def ramp_excl(start: float, end: float, L: int) -> List[float]:
        """Return L values excluding the start, including the end."""
        if L <= 0:
            return []
        return list(np.linspace(start, end, num=L + 1, endpoint=True)[1:])

    out: List[Tuple[float, float]] = []

    # A) hold 30 @ (1,1)
    out += hold(1.0, 1.0, 60)

    # B) both 1->0 over 90
    ra = ramp_excl(1.0, 0.0, 180)
    out += list(zip(ra, ra))

    # C) q_a 0->1 over 90 (q_g=0)
    ra = ramp_excl(0.0, 1.0, 180)
    out += [(v, 0.0) for v in ra]

    # D) q_a 1->0 over 90 (q_g=0)
    ra = ramp_excl(1.0, 0.0, 180)
    out += [(v, 0.0) for v in ra]

    # E) hold 30 @ (0,0)
    out += hold(0.0, 0.0, 60)

    # F) q_g 0->1 over 90 (q_a=0)
    rg = ramp_excl(0.0, 1.0, 180)
    out += [(0.0, v) for v in rg]

    # G) q_g 1->0 over 90 (q_a=0)
    rg = ramp_excl(1.0, 0.0, 180)
    out += [(0.0, v) for v in rg]

    # H) both 0->1 over 90
    ra = ramp_excl(0.0, 1.0, 180)
    out += list(zip(ra, ra))

    # I) hold 30 @ (1,1)
    out += hold(1.0, 1.0, 60)

    assert len(out) == 1260, f"Expected 630 frames, got {len(out)}"
    return out

def build_pingpong_paths(paths: List[str], total_frames: int) -> List[str]:
    if len(paths) <= 1:
        return (paths * total_frames)[:total_frames]
    cycle = list(range(len(paths))) + list(range(len(paths)-2, 0, -1))
    seq: List[str] = []
    while len(seq) < total_frames:
        for i in cycle:
            seq.append(paths[i])
            if len(seq) >= total_frames:
                break
    return seq


def compress_frame(model, device, ply_path: str, q_a: float, q_g: float):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pts)
    pts_t = torch.from_numpy(pts)[None].float()
    cols_t = torch.from_numpy(cols)[None].float()
    data = {"src": {"points": pts_t, "colors": cols_t}}

    with torch.no_grad():
        source_pc, rec_pc, bpp, t_comp, t_decomp = utils.compress_model_ours(
            EXPERIMENT, model, data, q_a, q_g, SCALING_FACTOR, BLOCK_SIZE, device, BASE_PATH
        )
    return rec_pc, float(bpp)

def render_rec(rec_pc, out_jpg_path: str):
    utils.render_pointcloud(rec_pc, out_jpg_path, point_size=POINT_SIZE)


def encode_video_ffmpeg(frames_dir: str, out_path: str, fps: int):
    cmd = [
        "ffmpeg","-y","-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.jpg"),
        "-pix_fmt","yuv420p","-crf","18","-preset","slow",
        out_path
    ]
    subprocess.run(cmd, check=True)

def main():
    # device/model
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    device = torch.device(DEVICE_ID)
    torch.cuda.set_device(device)
    print(CONFIG_PATH)
    with open(CONFIG_PATH) as stream:
        config = yaml.safe_load(stream)
    model = UnifiedModel(config["model"])
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    model.to(device).eval()
    model.update()

    # frames
    base_ply_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.ply")))
    ply_paths = build_pingpong_paths(base_ply_paths, MAX_FRAMES)
    if not ply_paths:
        raise FileNotFoundError(f"No .ply frames in {INPUT_DIR}")

    q_schedule = generate_q_schedule(MAX_FRAMES)

    # temp dirs
    Path(RENDERS_DIR).mkdir(parents=True, exist_ok=True)
    final_frames_dir = Path(RENDERS_DIR) / "video_frames"
    final_frames_dir.mkdir(parents=True, exist_ok=True)

    for idx, (ply_path, (qa, qg)) in enumerate(zip(ply_paths, q_schedule)):
        rec_pc, bpp = compress_frame(model, device, ply_path, qa, qg)

        # raw render path (jpeg)
        raw_path = os.path.join(RENDERS_DIR, "raw_{:04d}_{}.jpg".format(idx, "{}"))
        render_rec(rec_pc, str(raw_path))

        # load rendered image (whatever size your renderer outputs)
        raw_path = str(raw_path).format(FRAME_DIRECTION)
        img = Image.open(raw_path).convert("RGB")
        # optional: enforce a fixed size for consistency
        if RENDER_SIZE is not None:
            img = img.resize(RENDER_SIZE, Image.BICUBIC)

        img = overlay_hud(img, qa, qg, bpp, idx)
        img.save(final_frames_dir / f"frame_{idx:04d}.jpg", quality=95)

        # free GPU mem between frames
        torch.cuda.empty_cache()

        if (idx+1) % 10 == 0:
            print(f"[{idx+1}/{len(ply_paths)}] q_a={qa:.2f} q_g={qg:.2f} bpp={bpp:.3f}")

    encode_video_ffmpeg(str(final_frames_dir), OUTPUT_VIDEO, OUTPUT_FPS)
    print(f"Done -> {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
