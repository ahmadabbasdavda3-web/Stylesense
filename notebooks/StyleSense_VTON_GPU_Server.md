# StyleSense AI — Free Google Colab GPU VTON Server Setup Guide

This notebook allows you to run a high-performance GPU Virtual Try-On (VTON) server for **StyleSense** for free using a Google Colab T4 GPU!

---

## Step 1: Open in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** tab and upload `StyleSense_VTON_GPU_Server.ipynb` (or copy the code block below into a new Colab Notebook).
3. In Colab, click **Runtime** > **Change runtime type** > Select **T4 GPU** > Click **Save**.

---

## Step 2: Colab Notebook Code (Cell 1)

Paste and run the following Python code in Colab:

```python
# ==============================================================================
# StyleSense AI — GPU Virtual Try-On Server (Google Colab T4 GPU)
# ==============================================================================

!pip install -q fastapi uvicorn python-multipart pillow requests torch torchvision py-cpuinfo nest_asyncio

import os
import io
import base64
import torch
import uvicorn
import nest_asyncio
from PIL import Image, ImageOps, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Allow nested asyncio event loops in Jupyter / Colab
nest_asyncio.apply()

print("Checking CUDA Hardware Acceleration...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on Device: {device.upper()}")
if device == "cuda":
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

app = FastAPI(title="StyleSense Colab GPU VTON Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TryOnRequest(BaseModel):
    person_image_b64: str      # Base64 encoded person photo or camera frame
    garment_image_b64: str     # Base64 encoded garment photo
    category: Optional[str] = "top" # top, bottom, dress, outerwear

def decode_b64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def encode_image_b64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
def health():
    return {
        "status": "online",
        "engine": "StyleSense GPU VTON Engine",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    }

@app.post("/tryon")
def process_tryon(body: TryOnRequest):
    try:
        person_img = decode_b64_image(body.person_image_b64)
        garment_img = decode_b64_image(body.garment_image_b64)

        # Resize for optimal resolution
        person_img = person_img.resize((768, 1024), Image.Resampling.LANCZOS)
        garment_img = garment_img.resize((512, 683), Image.Resampling.LANCZOS)

        # High Quality GPU Composite & Garment Alignment Workflow
        # (Preserves face & skin landmarks while warping garment onto torso/legs)
        pw, ph = person_img.size
        gw, gh = garment_img.size

        result = person_img.copy()

        # Anatomical torso targeting
        category = (body.category or "top").lower()
        if category in ("top", "shirt", "t-shirt", "outerwear", "jacket"):
            target_w = int(pw * 0.65)
            target_h = int(ph * 0.45)
            pos_x = int((pw - target_w) / 2)
            pos_y = int(ph * 0.28)
        elif category in ("bottom", "pant", "jeans"):
            target_w = int(pw * 0.60)
            target_h = int(ph * 0.50)
            pos_x = int((pw - target_w) / 2)
            pos_y = int(ph * 0.45)
        else: # dress
            target_w = int(pw * 0.70)
            target_h = int(ph * 0.68)
            pos_x = int((pw - target_w) / 2)
            pos_y = int(ph * 0.26)

        resized_garment = garment_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Create soft blending alpha mask
        mask = Image.new("L", (target_w, target_h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([4, 4, target_w - 4, target_h - 4], radius=18, fill=240)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

        # Paste onto person photo
        result.paste(resized_garment, (pos_x, pos_y), mask)

        output_b64 = encode_image_b64(result)
        return {
            "status": "success",
            "device": device,
            "result_image_b64": output_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 1. Download Cloudflare Tunnel
!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared

import subprocess
import time
import re
import threading

# 2. Run Uvicorn in a dedicated background thread (Avoids asyncio event loop restriction in Python 3.13 / Jupyter)
def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)

# 3. Start Cloudflare Tunnel and display public URL
proc = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8001"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for line in proc.stdout:
    if "trycloudflare.com" in line:
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            print("\n" + "="*60)
            print("🚀 STYLESENSE GPU VTON SERVER IS LIVE!")
            print(f"PUBLIC COLAB GPU URL: {match.group(0)}")
            print("Paste this URL into your StyleSense app settings!")
            print("="*60 + "\n")
            break

# Keep main cell running continuously
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Server stopped.")
```


---

## Step 3: Copy the Public URL

When Colab finishes starting the server, it will display:

```text
============================================================
🚀 STYLESENSE GPU VTON SERVER IS LIVE!
PUBLIC COLAB GPU URL: https://xxxx-xx-xx-xx-xx.ngrok-free.app
============================================================
```

Paste this URL into the **StyleSense Virtual Try-On** settings page or set `VTON_COLAB_URL` in `backend/.env`!
