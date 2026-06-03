"""Headless driver for SUPIR via the running ComfyUI server (3-node graph:
LoadImage -> SUPIR_Upscale -> SaveImage). Tuned for a Lightning SDXL base
(low steps / low cfg) and v0F for fidelity on text-bearing subjects.

Usage:
  python supir_api.py --in <crop.png> --out <result.png> [--supir v0F|v0Q]
      [--sdxl RealVisXL_V4.0_Lightning.safetensors] [--steps 10] [--cfg 1.5]
      [--scale 1.0] [--prompt "..."] [--seed 123]
"""
import argparse
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

COMFY = Path(r"C:\Users\user\ComfyUI")
SERVER = "http://127.0.0.1:8188"

SUPIR_FILES = {
    "v0F": "SUPIR-v0F_fp16.safetensors",
    "v0Q": "SUPIR-v0Q_fp16.safetensors",
}


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--supir", default="v0F", choices=["v0F", "v0Q"])
    ap.add_argument("--sdxl", default="RealVisXL_V4.0_Lightning.safetensors")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--control", type=float, default=1.0)
    ap.add_argument("--tiled", action="store_true", help="use tiled sampling (for large whole images)")
    ap.add_argument("--prompt", default="high quality, sharp focus, fine detail, photorealistic, crisp text")
    ap.add_argument("--nprompt", default="blurry, motion blur, out of focus, low quality, noise, jpeg artifacts, deformed text")
    args = ap.parse_args()

    inp = Path(args.inp)
    # stage input into ComfyUI/input
    in_name = f"supir_in_{uuid.uuid4().hex[:8]}{inp.suffix}"
    (COMFY / "input").mkdir(parents=True, exist_ok=True)
    shutil.copy2(inp, COMFY / "input" / in_name)

    prefix = "supir_" + uuid.uuid4().hex[:8]
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": in_name}},
        "2": {"class_type": "SUPIR_Upscale", "inputs": {
            "supir_model": SUPIR_FILES[args.supir],
            "sdxl_model": args.sdxl,
            "image": ["1", 0],
            "seed": args.seed,
            "resize_method": "lanczos",
            "scale_by": args.scale,
            "steps": args.steps,
            "restoration_scale": -1.0,
            "cfg_scale": args.cfg,
            "a_prompt": args.prompt,
            "n_prompt": args.nprompt,
            "s_churn": 5,
            "s_noise": 1.003,
            "control_scale": args.control,
            "cfg_scale_start": args.cfg,
            "control_scale_start": 0.0,
            "color_fix_type": "Wavelet",
            "keep_model_loaded": True,
            "use_tiled_vae": True,
            "encoder_tile_size_pixels": 512,
            "decoder_tile_size_latent": 64,
            "sampler": "RestoreEDMSampler",
            "use_tiled_sampling": bool(args.tiled),
            "sampler_tile_size": 1024,
            "sampler_tile_stride": 512,
        }},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": prefix}},
    }

    r = post("/prompt", {"prompt": prompt, "client_id": uuid.uuid4().hex})
    pid = r["prompt_id"]
    print(f"[supir] queued prompt {pid} (supir={args.supir}, steps={args.steps}, cfg={args.cfg}, scale={args.scale})", flush=True)

    t0 = time.time()
    out_imgs = None
    while time.time() - t0 < 1200:
        time.sleep(3)
        hist = get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False and status.get("status_str") == "error":
                print("[supir] ERROR:", json.dumps(status)[:800], flush=True)
                # also dump messages
                for m in status.get("messages", []):
                    print("  ", m, flush=True)
                return
            outs = entry.get("outputs", {})
            if "3" in outs and outs["3"].get("images"):
                out_imgs = outs["3"]["images"]
                break
            if status.get("completed"):
                # completed but check for any image output
                for nid, o in outs.items():
                    if o.get("images"):
                        out_imgs = o["images"]; break
                if out_imgs:
                    break
    if not out_imgs:
        print("[supir] timed out or no output", flush=True)
        return

    img = out_imgs[0]
    src = COMFY / "output" / (img.get("subfolder") or "") / img["filename"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.out)
    print(f"[supir] done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
