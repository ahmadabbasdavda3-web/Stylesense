# PROJECT ANALYSIS — Smart Mirror Open-Source VTON Integration

## 1. Current Architecture Overview

- **Frontend**: Lightweight single-page web application (`frontend/index.html`) using HTML5, Vanilla JavaScript (`async/await fetch`), and CSS design system.
- **Backend Framework**: Python FastAPI (`backend/main.py`) powered by Uvicorn ASGI server running on `http://127.0.0.1:8000`.
- **Database Layer**: SQLAlchemy ORM with SQLite database (`backend/wardrobe.db`).
- **AI/ML Layer**:
  - **Classification**: Zero-shot clothing detection with PyTorch & OpenAI CLIP (`backend/ai_detector.py`).
  - **Outfit Recommender**: Heuristic rule engine with matrix ranking (`backend/recommender.py`).
  - **Virtual Try-On / Fitting**: Anatomical landmark estimation, face protection mask, garment lighting adaptation (`backend/virtual_tryon.py`).
  - **AI Service Integration**: Google Gemini multimodal API integration (`backend/ai_generator.py`).
- **File Storage**: Local filesystem (`backend/uploads/`) mounted as static directory `/uploads`.

---

## 2. Existing System Component Flows

### A. Existing Wardrobe & Upload Flow
1. User uploads clothing photo via `POST /upload-item/` or batch upload `POST /upload-items/`.
2. Image saved to `backend/uploads/` with unique UUID.
3. `ai_detector.py` uses PyTorch CLIP model to detect `category`, `color`, `pattern`, `style`, `gender`, and `confidence`.
4. Metadata recorded in SQLite `wardrobe_items` table.
5. User can view, filter, or manually correct detection attributes (`PATCH /items/{item_id}`).

### B. Existing Recommendation Flow
1. User requests outfits for an occasion (e.g. `college`, `office`, `wedding`, `party`) via `GET /suggest/{occasion}`.
2. `recommender.py` queries `WardrobeItem` database, pairs compatible Tops + Bottoms + Shoes + Jackets/Dresses based on color harmony, pattern rules, and style priority.
3. Returns top ranked combinations with match percentage scores and stylist advice notes.

### C. Existing Smart Mirror / Try-On Flow
1. User uploads full-body photo stored in `user_photos` table via `POST /tryon/user-photos/`.
2. User selects an outfit card from recommendation results.
3. User clicks **"Try-On In Smart Mirror"**, invoking `triggerSmartMirrorTryOn(o)` in `frontend/index.html`.
4. Requests sent to `POST /tryon/generate-gemini/` or `POST /tryon/generate/`.
5. Backend performs face extraction, chin protection masking, undergarment canvas blurring, and multi-garment compositing (Top + Bottom + Shoes).
6. Composite image returned as base64 string and rendered inside the Smart Mirror container.

---

## 3. Scope of Modifications

### Files That Must Be Modified / Extended:
- **`backend/schemas.py`**: Add Pydantic request/response schemas for VTON jobs (`VtonTryOnRequest`, `VtonJobResponse`).
- **`backend/models.py`**: Ensure `TryOnHistory` / `virtual_try_ons` tracks model name, status, job IDs, and result paths.
- **`backend/main.py`**: Add dedicated `/api/smart-mirror/try-on` & job status endpoints without breaking existing routes.
- **`frontend/index.html`**: Update Smart Mirror UI to invoke the dedicated open-source VTON pipeline, displaying real-time processing status and final output.

### Files That Should NOT Be Modified:
- **`backend/ai_detector.py`**: CLIP clothing detection operates cleanly and independently.
- **`backend/recommender.py`**: Recommendation ranking algorithm functions correctly.
- **`backend/database.py`**: Database session initialization works properly.

---

## 4. Proposed Open-Source VTON Integration Architecture

```text
                               +-----------------------------+
                               |   Frontend UI (index.html)  |
                               +--------------+--------------+
                                              |
                                   POST /api/smart-mirror/try-on
                                              |
                                              v
                               +-----------------------------+
                               |    FastAPI (main.py)         |
                               +--------------+--------------+
                                              |
                                              v
                               +-----------------------------+
                               |   VTON Service Layer        |
                               |   (backend/ai/vton/)        |
                               +--------------+--------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
      +-----------------------------+                   +-----------------------------+
      |  Open-Source VTON Engine    |                   | Preprocessing / Utility     |
      |  (CatVTON / IDM-VTON /      |                   | (Person Pose / Garment      |
      |   Precision Composite)      |                   |  Masking & Segmentation)    |
      +--------------+--------------+                   +--------------+--------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |  Result Storage & Response  |
                               +-----------------------------+
```

---

## 5. Dependencies & Compatibility Analysis

- **PyTorch & Torchvision**: Pre-installed in `venv`. Supports CUDA hardware acceleration if NVIDIA GPU is present, with CPU fallback.
- **Diffusers / Transformers**: Optional Hugging Face pipeline integration for open-source diffusion models.
- **OpenCV & PIL**: Handles image resizing, background removal, alignment, and mask generation.
- **Hardware Memory & CUDA**: Consumer hardware without dedicated 16GB+ VRAM will be provided with an optimized, lightweight local VTON inference pipeline that does not crash on CPU/low-VRAM machines.
