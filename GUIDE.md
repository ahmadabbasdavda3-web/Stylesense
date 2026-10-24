# Smart Wardrobe — Complete Setup & Architecture Guide

Welcome to **Smart Wardrobe**, an AI-powered wardrobe assistant and outfit recommendation engine built with FastAPI, SQLite, and OpenAI CLIP zero-shot image classification.

---

## What Has Been Fixed and Improved

1. **Backend Startup & Virtual Environment Fixes**:
   - Resolved environment corruption and launcher shim paths caused by folder migration.
   - Entrypoint scripts (`uvicorn.exe`, `pip.exe`) and activation scripts now accurately reference the active environment.
   - Added automated startup scripts for single-click execution on Windows.
2. **Resilient Database Auto-Migration**:
   - Resolved the fatal SQLite crash (`no such column: wardrobe_items.gender`) by introducing automated schema synchronization in `database.py`. New or missing columns are automatically detected and migrated on startup.
3. **Accurate AI Style & Attribute Classification**:
   - Eliminated the bias where clothing items were disproportionately tagged as "party wear".
   - Implemented prompt-engineered contextual templates for OpenAI CLIP zero-shot classification (e.g. `"formal business clothing"`, `"casual everyday clothes"`, `"festive party wear"`, `"athletic sporty activewear"`).
   - Added PyTorch `torch.no_grad()` inference optimization for lower memory usage and faster processing.
4. **Reliable Outfit Recommendations & Cycling**:
   - The recommender respects strict style priorities per occasion (e.g., college prioritizes casual, then sporty, then formal).
   - "Try another combo" smoothly cycles through alternative tops or bottoms without dead-ending with a "missing item" error.
   - Missing item alerts, badges, and purchase suggestions are clear and informative.
5. **Conversational Stylist Assistant (Chatbot)**:
   - Understands natural English phrasing and synonyms (e.g., "work", "job", "wedding", "date", "class", "interview").
   - Provides clean outfit suggestions with styling advice.
6. **Fluent English Localization**:
   - All text across the user interface, toast notifications, backend APIs, chatbot dialogues, and documentation is now in fluent, professional English.

---

## Project Structure

```
smart-wardrobe/
  backend/
    main.py              # FastAPI server, endpoints, and static file mounting
    models.py            # SQLAlchemy database models (WardrobeItem, Outfit, Feedback)
    database.py          # Database engine, session setup, and auto-migration
    ai_detector.py       # Zero-shot clothing classification with CLIP
    recommender.py       # Occasion-based outfit matching and fallback algorithm
    schemas.py           # Pydantic request and response models
    requirements.txt     # Backend dependencies
    uploads/             # Stored wardrobe image files
    start_backend.bat    # 1-click script to start the backend server
  frontend/
    index.html           # Full single-page web interface
  start_app.bat          # 1-click script to start backend & open frontend
  GUIDE.md               # Setup and architecture documentation
```

---

## Step 1 — Verify Python Version

Python 3.10 through 3.14 is supported. To check your installed version:

```bash
python --version
```

---

## Step 2 — Backend Setup

If not already configured, navigate to the `backend` directory and activate the virtual environment:

### Windows (PowerShell):
```powershell
cd backend
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt):
```cmd
cd backend
venv\Scripts\activate.bat
```

### macOS / Linux:
```bash
cd backend
source venv/bin/activate
```

If dependencies need to be installed or updated:
```bash
pip install -r requirements.txt
```

---

## Step 3 — Run the Backend

You can start the backend using either the convenience script or `uvicorn`:

### Option A: Using the convenience script (Windows)
Double-click `backend/start_backend.bat` or run:
```cmd
.\backend\start_backend.bat
```

### Option B: Using Uvicorn directly
From the `backend` directory:
```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Verify in your browser:
Open `http://127.0.0.1:8000/`. It will return:
```json
{
  "message": "Smart Wardrobe backend is running!",
  "occasions": ["college", "office", "interview", "wedding", "party"]
}
```

---

## Step 4 — Open the Frontend

With the backend running:

- **Option A (Quickest)**: Double-click `frontend/index.html` to open it in your default web browser.
- **Option B (VS Code Live Server)**: Right-click `frontend/index.html` in VS Code and select **Open with Live Server**.
- **Option C (All-in-one)**: Double-click `start_app.bat` in the project root to launch both the backend and frontend together.

---

## Step 5 — Application Features & Usage

1. **Wardrobe Management**:
   - Drag and drop or browse to upload clothing images (JPG/PNG).
   - CLIP automatically identifies Category, Color, Pattern, and Style.
   - If an attribute needs adjustment, click **Edit** on any item card to correct it.
2. **Get an Outfit**:
   - Select an occasion (College, Office, Interview, Wedding, Party) and click **Suggest outfit**.
   - Click **Try another combo** to explore alternative pairings from your wardrobe.
   - Click **Save this outfit** and provide a star rating to record it.
3. **Ask the Stylist (Chatbot)**:
   - Ask natural language questions like *"What should I wear to the office?"* or *"Suggest something for a weekend party"*.
4. **Worn History & Wardrobe Stats**:
   - Track your worn outfit history with ratings and review wardrobe distribution metrics.

---

## Troubleshooting Guide

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure the virtual environment is activated and run `pip install -r requirements.txt`. |
| "Could not connect to backend" | Ensure `uvicorn main:app --reload --port 8000` is running and accessible at `http://127.0.0.1:8000`. |
| First photo upload takes longer | CLIP downloads pretrained weights (~600 MB) on its very first inference pass; subsequent uploads are instantaneous. |
| CORS issue in browser | FastAPI is configured with `allow_origins=["*"]`, allowing requests from any local file or dev server. |

