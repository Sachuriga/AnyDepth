#!/usr/bin/env python3
"""
Convert an RGB video into a depth video with Depth Anything V2 (ViT-B) + SDT.

Requires a local clone of Depth-Anything-V2 (for the DINOv2 encoder definition)
and the released checkpoint dav2_sdt_vitb.pth.

Example:
    python run_video.py --video input.mp4 --ckpt dav2_sdt_vitb.pth \
        --dav2-root ../Depth-Anything-V2 --out depth.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sdt_head import SDTHead

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH_SIZE = 14


class DAv2SDT(nn.Module):
    # Attribute names match the checkpoint keys: "pretrained.*" and "depth_head.*"
    def __init__(self, encoder="vitb", fusion_channels=128):
        super().__init__()
        from depth_anything_v2.dinov2 import DINOv2

        self.layer_idx = {"vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}[encoder]
        self.pretrained = DINOv2(model_name=encoder)
        embed_dim = self.pretrained.embed_dim
        self.depth_head = SDTHead(
            in_channels=[embed_dim] * 4,
            fusion_channels=fusion_channels,
            n_output_channels=1,
            use_cls_token=True,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(
            x, self.layer_idx, reshape=True, return_class_token=True
        )
        depth = self.depth_head(features)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        return F.relu(depth).squeeze(1)


def preprocess(frame_bgr, input_size):
    h, w = frame_bgr.shape[:2]
    # Shorter side -> input_size, both sides rounded up to a multiple of the patch size
    scale = input_size / min(h, w)
    new_h = int(np.ceil(h * scale / PATCH_SIZE) * PATCH_SIZE)
    new_w = int(np.ceil(w * scale / PATCH_SIZE) * PATCH_SIZE)
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = (image - MEAN) / STD
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


def colorize(depth, invert, grayscale):
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    if invert:
        d = 1.0 - d
    d = (d * 255.0).astype(np.uint8)
    if grayscale:
        return cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def align_to(depth, ref, n_iter=2, keep=0.9):
    # Robust least-squares fit of depth -> a * depth + b onto ref (removes the per-frame
    # scale/shift ambiguity of relative depth); the moving animal is dropped as an outlier
    d = depth[::4, ::4].ravel()
    r = ref[::4, ::4].ravel()
    mask = np.ones_like(d, dtype=bool)
    for _ in range(n_iter):
        A = np.stack([d[mask], np.ones(mask.sum())], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, r[mask], rcond=None)
        resid = np.abs(a * d + b - r)
        mask = resid <= np.quantile(resid, keep)
    return a * depth + b


def build_background(infer, cap, start, count, n_samples):
    # Median depth over frames sampled across the clip: the static scene (floor + maze)
    # without the animal, since it moves
    positions = np.linspace(start, start + count - 1, num=min(n_samples, count)).astype(int)
    stack = []
    for i, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        stack.append(infer(frame))
        print(f"\rbackground {i + 1}/{len(positions)}", end="", flush=True)
    print()
    background = np.median(np.stack(stack), axis=0)
    # Second pass: align every sample to the first estimate, then take the median again
    aligned = [align_to(d, background) for d in stack]
    background = np.median(np.stack(aligned), axis=0)
    return background, aligned


def main():
    parser = argparse.ArgumentParser(description="AnyDepth (DAv2 + SDT) video depth estimation")
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--ckpt", required=True, help="path to dav2_sdt_vitb.pth")
    parser.add_argument("--dav2-root", required=True, help="path to a clone of Depth-Anything-V2")
    parser.add_argument("--out", default=None, help="output video path (default: <input>_depth.mp4)")
    parser.add_argument("--input-size", type=int, default=518, help="shorter side fed to the network")
    parser.add_argument("--side-by-side", action="store_true", help="put the RGB frame next to the depth")
    parser.add_argument("--grayscale", action="store_true", help="write grayscale instead of the inferno colormap")
    parser.add_argument("--invert", action="store_true", help="flip near/far colors")
    parser.add_argument("--save-npy", action="store_true", help="also save raw per-frame predictions as .npy")
    parser.add_argument("--fp16", action="store_true", help="run inference in half precision (faster, CUDA only)")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame to process")
    parser.add_argument("--max-frames", type=int, default=None, help="process at most this many frames (for quick tests)")
    parser.add_argument("--height", action="store_true",
                        help="subtract a static background so only height above the maze/floor remains (fixed camera only)")
    parser.add_argument("--bg-frames", type=int, default=100, help="frames sampled to build the background (--height)")
    parser.add_argument("--min-height", type=float, default=0.02,
                        help="heights below this are set to 0, as a fraction of the scene depth range (--height)")
    parser.add_argument("--max-height", type=float, default=None,
                        help="height mapped to full color, as a fraction of the scene depth range; default: auto (--height)")
    args = parser.parse_args()

    dav2_root = os.path.abspath(args.dav2_root)
    if not os.path.isfile(os.path.join(dav2_root, "depth_anything_v2", "dinov2.py")):
        sys.exit(
            f"Depth-Anything-V2 not found at {dav2_root}\n"
            "Clone it first: git clone https://github.com/DepthAnything/Depth-Anything-V2"
        )
    sys.path.insert(0, dav2_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DAv2SDT(encoder="vitb", fusion_channels=128)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    del ckpt
    model = model.to(device).eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        n_frames = max(n_frames - args.start_frame, 0)
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)

    out_path = args.out or os.path.splitext(args.video)[0] + "_depth.mp4"
    out_w = width * 2 if args.side_by_side else width
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, height))

    npy_frames = [] if args.save_npy else None
    use_fp16 = args.fp16 and device == "cuda"

    @torch.inference_mode()
    def infer(frame):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            depth = model(preprocess(frame, args.input_size).to(device))
        depth = F.interpolate(depth[:, None].float(), size=(height, width), mode="bilinear", align_corners=True)
        return depth[0, 0].cpu().numpy()

    if args.height:
        background, samples = build_background(infer, cap, args.start_frame, n_frames, args.bg_frames)
        lo, hi = np.percentile(background, [1, 99])
        scene_range = hi - lo + 1e-8
        # The model predicts disparity-like values (larger = nearer); --invert if it predicts depth
        sign = -1.0 if args.invert else 1.0

        def to_height(depth, aligned=False):
            d = depth if aligned else align_to(depth, background)
            h = sign * (d - background) / scene_range
            h[h < args.min_height] = 0.0
            return h

        max_height = args.max_height
        if max_height is None:
            # Fixed color scale for the whole video, from the animal pixels in the sampled frames
            heights = np.concatenate([h[h > 0] for h in (to_height(d, aligned=True) for d in samples)] + [np.zeros(1)])
            max_height = float(np.percentile(heights[heights > 0], 99)) if (heights > 0).any() else 1.0
        del samples
        print(f"scene depth range {scene_range:.4g}, min height {args.min_height}, max height {max_height:.4g}")
        cv2.imwrite(os.path.splitext(out_path)[0] + "_background.png",
                    colorize(background, args.invert, args.grayscale))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    idx = 0
    while args.max_frames is None or idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        depth = infer(frame)

        if args.height:
            depth = to_height(depth)
            d = (np.clip(depth / max_height, 0, 1) * 255).astype(np.uint8)
            vis = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR) if args.grayscale else cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
        else:
            vis = colorize(depth, args.invert, args.grayscale)

        if npy_frames is not None:
            npy_frames.append(depth.astype(np.float16))

        if args.side_by_side:
            vis = np.concatenate([frame, vis], axis=1)
        writer.write(vis)

        idx += 1
        print(f"\r{idx}/{n_frames}", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved depth video to {out_path}")

    if npy_frames is not None:
        npy_path = os.path.splitext(out_path)[0] + ".npy"
        np.save(npy_path, np.stack(npy_frames))
        print(f"Saved raw predictions to {npy_path}")


if __name__ == "__main__":
    main()
