import os
import shutil
import uuid
import json
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import Base, engine, SessionLocal, init_db
from models import User, StyleProfile, WardrobeItem, Outfit, Feedback, TryOnSession
from ai_detector import detect_attributes as clip_detect_attributes
from ai_detector import CATEGORIES as CLIP_CATEGORIES
from recommender import recommend_outfit, recommend_outfits, normalize_gender, OCCASION_STYLE_PRIORITY
from schemas import (
    UserCreate, UserLogin, Token, StyleProfileSchema,
    SaveOutfitRequest, RateOutfitRequest, FeedbackRequest, ChatRequest, ItemUpdateRequest, StylistRecommendRequest,
    TryOnSessionCreate, TryOnProcessRequest
)

from services.auth_service import (
    get_password_hash, verify_password, create_access_token, get_current_user_optional
)
from services.cloudinary_service import upload_image_to_cloud
from services.gemini_service import analyze_clothing_image
from services.stylist_service import generate_stylist_recommendation
from services.tryon_service import (
    create_tryon_session, process_tryon_frame, set_colab_gpu_url, get_colab_gpu_url, detect_skin_tone_from_b64
)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

init_db()

app = FastAPI(
    title="StyleSense AI API",
    description="Personal AI Wardrobe & Dressing Room Backend Powered by Gemini AI & Remote Colab GPU VTON Engine",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def item_to_dict(item: WardrobeItem):
    """Formats WardrobeItem into rich structured dictionary for frontend consumption."""
    def _parse_json(val, default):
        if not val:
            return default
        try:
            return json.loads(val)
        except Exception:
            return default

    img_path = (item.image_path or "").replace("\\", "/")
    cloud_url = item.cloudinary_url or (f"/{img_path}" if img_path else "")

    return {
        "id": item.id,
        "user_id": item.user_id,
        "image_path": img_path,
        "image_url": cloud_url,
        "thumbnail_url": item.thumbnail_path or cloud_url,
        "cloudinary_url": item.cloudinary_url,
        "category": item.category,
        "subcategory": item.subcategory or item.category,
        "color": item.color,
        "secondary_colors": _parse_json(item.secondary_colors, []),
        "pattern": item.pattern,
        "material": item.material or "unknown",
        "fit": item.fit or "regular",
        "style": item.style,
        "season": _parse_json(item.season, ["spring", "summer", "autumn", "winter"]),
        "occasion": _parse_json(item.occasion, ["casual"]),
        "formality": item.formality or "casual",
        "gender": getattr(item, "gender", "unisex") or "unisex",
        "brand": item.brand,
        "price": item.price,
        "confidence": item.confidence,
        "needs_review": item.needs_review,
        "wear_count": item.wear_count,
        "date_added": item.date_added.isoformat() if item.date_added else None,
        "ai_generated_attributes": _parse_json(item.ai_generated_attributes, {}),
        "user_edited_attributes": _parse_json(item.user_edited_attributes, {})
    }


@app.get("/")
def home():
    return {
        "app": "StyleSense AI",
        "message": "StyleSense AI backend is running!",
        "occasions": list(OCCASION_STYLE_PRIORITY),
        "colab_gpu_url": get_colab_gpu_url(),
        "features": [
            "AI Wardrobe Management",
            "Gemini Multimodal Clothing Analysis",
            "Personal Style Profile",
            "AI Stylist Recommendations",
            "Remote Colab GPU Virtual Try-On Engine"
        ]
    }


# ==========================================
# AUTH & USER API ENDPOINTS
# ==========================================

@app.post("/api/auth/register", response_model=Token)
def register(body: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if existing:
        raise HTTPException(400, "User with this email already exists.")

    hashed_pwd = get_password_hash(body.password)
    new_user = User(
        email=body.email.lower().strip(),
        name=body.name,
        password_hash=hashed_pwd
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    profile = StyleProfile(user_id=new_user.id)
    db.add(profile)
    db.commit()

    access_token = create_access_token({"sub": new_user.id, "email": new_user.email, "name": new_user.name})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": new_user.id, "email": new_user.email, "name": new_user.name}
    }


@app.post("/api/auth/login", response_model=Token)
def login(body: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password.")

    access_token = create_access_token({"sub": user.id, "email": user.email, "name": user.name})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name}
    }


@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user_optional)):
    return current_user


# ==========================================
# STYLE PROFILE ENDPOINTS
# ==========================================

@app.get("/api/profile")
def get_profile(current_user: dict = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    profile = db.query(StyleProfile).filter(StyleProfile.user_id == current_user["id"]).first()
    if not profile:
        profile = StyleProfile(user_id=current_user["id"])
        db.add(profile)
        db.commit()
        db.refresh(profile)

    def _json(val):
        try:
            return json.loads(val) if val else []
        except Exception:
            return []

    return {
        "user_id": profile.user_id,
        "preferred_styles": _json(profile.preferred_styles),
        "preferred_colors": _json(profile.preferred_colors),
        "avoided_colors": _json(profile.avoided_colors),
        "preferred_fits": _json(profile.preferred_fits),
        "favorite_categories": _json(profile.favorite_categories),
        "lifestyle": profile.lifestyle,
        "height": profile.height,
        "body_proportions": profile.body_proportions,
    }


@app.put("/api/profile")
def update_profile(body: StyleProfileSchema, current_user: dict = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    profile = db.query(StyleProfile).filter(StyleProfile.user_id == current_user["id"]).first()
    if not profile:
        profile = StyleProfile(user_id=current_user["id"])
        db.add(profile)

    profile.preferred_styles = json.dumps(body.preferred_styles)
    profile.preferred_colors = json.dumps(body.preferred_colors)
    profile.avoided_colors = json.dumps(body.avoided_colors)
    profile.preferred_fits = json.dumps(body.preferred_fits)
    profile.favorite_categories = json.dumps(body.favorite_categories)
    profile.lifestyle = body.lifestyle
    if body.height is not None:
        profile.height = body.height
    if body.body_proportions is not None:
        profile.body_proportions = body.body_proportions

    db.commit()
    return {"message": "Style profile updated successfully!"}


# ==========================================
# WARDROBE & UPLOAD API ENDPOINTS
# ==========================================

def _process_and_save_upload(file: UploadFile, current_user: dict, db: Session) -> WardrobeItem:
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cloud_info = upload_image_to_cloud(file_path)
    cloudinary_url = cloud_info.get("secure_url") if cloud_info else None
    thumbnail_url = cloud_info.get("thumbnail_url") if cloud_info else None

    try:
        ai_res = analyze_clothing_image(file_path)
    except Exception as e:
        ai_res = {
            "category": "top", "subcategory": "shirt", "color": "unknown",
            "pattern": "solid", "style": "casual", "confidence": 0.5, "needs_review": True
        }

    category = ai_res.get("category", "top").lower().strip()
    if category not in ("top", "bottom", "dress", "shoes", "outerwear"):
        category = "top"

    # Single taxonomy of record on WRITE: store the fine label so every
    # reader (recommender, stats, export) works without coarse mapping.
    # Coarse-only rows are flagged for one-tap confirmation in the modal.
    raw_sub = (ai_res.get("subcategory", "") or "").strip()
    needs_fine_review = False
    if raw_sub in CLIP_CATEGORIES:
        fine_category = raw_sub
    else:
        fine_category = category
        needs_fine_review = True

    new_item = WardrobeItem(
        user_id=current_user["id"],
        image_path=f"uploads/{safe_name}",
        thumbnail_path=thumbnail_url,
        cloudinary_url=cloudinary_url,
        category=fine_category,
        subcategory=raw_sub if raw_sub else fine_category,
        color=ai_res.get("color", "unknown"),
        secondary_colors=json.dumps(ai_res.get("secondaryColors", [])),
        pattern=ai_res.get("pattern", "solid"),
        material=ai_res.get("material", "unknown"),
        fit=ai_res.get("fit", "regular"),
        style=ai_res.get("style", "casual"),
        season=json.dumps(ai_res.get("season", ["spring", "summer", "autumn", "winter"])),
        occasion=json.dumps(ai_res.get("occasion", ["casual"])),
        formality=ai_res.get("formality", "casual"),
        gender=ai_res.get("genderSuitability", "unisex"),
        confidence=ai_res.get("confidence", 0.9),
        needs_review=bool(ai_res.get("needs_review", False)) or needs_fine_review,
        ai_generated_attributes=json.dumps(ai_res)
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item


@app.post("/upload-item/")
def upload_item(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload a valid image file.")

    new_item = _process_and_save_upload(file, current_user, db)
    return {
        "message": "Item uploaded and AI analyzed successfully!",
        "item": item_to_dict(new_item)
    }


@app.post("/upload-items/")
def upload_items(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    results = []
    errors = []

    for file in files:
        if not (file.content_type or "").startswith("image/"):
            errors.append({"filename": file.filename, "error": "Not an image file, skipped."})
            continue

        try:
            item = _process_and_save_upload(file, current_user, db)
            results.append(item_to_dict(item))
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    return {
        "message": f"{len(results)} item(s) analyzed successfully" + (f", {len(errors)} failed" if errors else ""),
        "items": results,
        "errors": errors
    }


@app.get("/items/")
def get_items(current_user: dict = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    query = db.query(WardrobeItem).filter(
        (WardrobeItem.user_id == current_user["id"]) | (WardrobeItem.user_id == 1) | (WardrobeItem.user_id.is_(None))
    )
    items = query.order_by(WardrobeItem.date_added.desc()).all()
    return [item_to_dict(i) for i in items]


@app.patch("/items/{item_id}")
def update_item(
    item_id: int,
    update: ItemUpdateRequest,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    item = db.query(WardrobeItem).filter(
        WardrobeItem.id == item_id,
        WardrobeItem.user_id == current_user["id"]
    ).first()
    if not item:
        raise HTTPException(404, "Wardrobe item not found.")

    edits = {}
    for field in ("category", "subcategory", "color", "pattern", "style", "fit", "material", "formality", "brand", "price"):
        val = getattr(update, field, None)
        if val is not None and str(val).strip():
            setattr(item, field, str(val).strip())
            edits[field] = str(val).strip()

    if update.gender:
        g = str(update.gender).strip().lower()
        if g in ("male", "female", "unisex"):
            item.gender = g
            edits["gender"] = g

    # Single taxonomy of record: a fine subcategory always determines the
    # coarse category. (Split columns once let shirt-labeled jackets and
    # pant-labeled jeans sit in the DB, poisoning recommendations.)
    sub = (getattr(item, "subcategory", "") or "").strip()
    if sub in CLIP_CATEGORIES and item.category != sub:
        item.category = sub
        edits["category"] = sub

    item.user_edited_attributes = json.dumps(edits)
    item.needs_review = False
    item.confidence = 1.0

    db.commit()
    db.refresh(item)
    return {"message": "Item attributes updated successfully!", "item": item_to_dict(item)}


@app.delete("/items/{item_id}")
def delete_item(
    item_id: int,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    item = db.query(WardrobeItem).filter(
        WardrobeItem.id == item_id,
        WardrobeItem.user_id == current_user["id"]
    ).first()
    if not item:
        raise HTTPException(404, "Wardrobe item not found.")

    if item.image_path:
        filename = os.path.basename(item.image_path)
        disk_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(disk_path):
            try:
                os.remove(disk_path)
            except Exception:
                pass

    db.delete(item)
    db.commit()
    return {"message": "Item deleted."}


# ==========================================
# VIRTUAL TRY-ON (VTON) API ENDPOINTS
# ==========================================

@app.post("/api/tryon/session")
def api_create_tryon_session(
    body: TryOnSessionCreate,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Initializes a new Virtual Try-On session."""
    return create_tryon_session(
        user_id=current_user["id"],
        garment_ids=body.garment_ids,
        db=db
    )


@app.post("/api/tryon/process")
def api_process_tryon(
    body: TryOnProcessRequest,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Executes Virtual Try-On request using remote Colab GPU engine (if available) or lightweight local composite engine."""
    person_b64 = body.person_image_b64 or body.user_image_b64
    if not body.garment_id:
        raise HTTPException(400, "garment_id is required.")
    if not person_b64:
        raise HTTPException(400, "person_image_b64 or user_image_b64 (camera frame or user photo) is required.")

    return process_tryon_frame(
        session_token=body.session_token or "guest_session",
        garment_id=body.garment_id,
        person_image_b64=person_b64,
        db=db,
        upload_dir=UPLOAD_DIR,
        colab_url_override=body.colab_url,
        height=body.height,
        weight=body.weight,
        skin_tone=body.skin_tone,
        body_type=body.body_type,
        complexion_note=body.complexion_note
    )


@app.get("/api/tryon/config")
def get_tryon_config():
    """Returns current Virtual Try-On GPU engine configuration."""
    return {
        "colab_gpu_url": get_colab_gpu_url(),
        "is_colab_active": bool(get_colab_gpu_url()),
        "fallback_engine": "Lightweight Local Composite Engine"
    }


@app.post("/api/tryon/config")
def update_tryon_config(url: Optional[str] = Form(None)):
    """Updates the Google Colab GPU server tunnel URL on the fly."""
    set_colab_gpu_url(url or "")
    return {
        "message": "Colab GPU server URL updated successfully!",
        "colab_gpu_url": get_colab_gpu_url()
    }


@app.post("/api/detect-skin-tone")
def api_detect_skin_tone(body: Dict[str, Any]):
    """Auto-detects skin tone from uploaded person image base64."""
    person_b64 = body.get("person_image_b64") or body.get("user_image_b64") or ""
    return detect_skin_tone_from_b64(person_b64)


# ==========================================
# RECOMMENDATION & AI STYLIST ENDPOINTS
# ==========================================

@app.post("/api/stylist/recommend")
def recommend_stylist(
    body: StylistRecommendRequest,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    return generate_stylist_recommendation(
        user_id=current_user["id"],
        occasion=body.occasion,
        user_prompt=body.prompt,
        db=db,
        gender=body.gender or "all"
    )


@app.get("/recommend/{occasion}")
def recommend(
    occasion: str,
    exclude: str = "",
    gender: str = "all",
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    exclude_ids = {int(x) for x in exclude.split(",") if x.strip().isdigit()}
    items = db.query(WardrobeItem).filter(WardrobeItem.user_id == current_user["id"]).all()
    result = recommend_outfit(items, occasion, exclude_ids=exclude_ids, gender=gender)
    return result


@app.get("/suggest/{occasion}")
def suggest(
    occasion: str,
    limit: int = 6,
    offset: int = 0,
    exclude: str = "",
    exclude_pairs: str = "",
    gender: str = "all",
    skin_tone: Optional[str] = None,
    height: Optional[str] = None,
    weight: Optional[str] = None,
    body_type: Optional[str] = None,
    shuffle: bool = False,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    limit = max(1, min(limit, 12))
    offset = max(0, offset)
    exclude_ids = {int(x) for x in exclude.split(",") if x.strip().isdigit()}
    seen_pairs = set()
    for token in (exclude_pairs or "").split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        t, b = token.split(":", 1)
        if not t.strip().isdigit():
            continue
        bid = int(b) if b.strip().isdigit() else None
        seen_pairs.add((int(t), bid))

    items = db.query(WardrobeItem).filter(WardrobeItem.user_id == current_user["id"]).all()
    options, total = recommend_outfits(
        items, occasion.lower().strip(), limit=limit, offset=offset,
        exclude_ids=exclude_ids, gender=gender, shuffle=shuffle, exclude_pairs=seen_pairs,
        skin_tone=skin_tone, height=height, weight=weight, body_type=body_type
    )

    if not options and offset == 0:
        return {
            "occasion": occasion.lower().strip(),
            "gender": normalize_gender(gender),
            "total": 0,
            "options": [],
            "message": "That's every outfit combination available in your wardrobe for this occasion. Upload more clothes for fresh ideas!"
        }
    return {
        "occasion": occasion.lower().strip(),
        "gender": normalize_gender(gender),
        "total": total,
        "options": options
    }


@app.post("/save-outfit/")
def save_outfit(
    body: SaveOutfitRequest,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    outfit = Outfit(
        user_id=current_user["id"],
        name=body.name or f"{body.occasion.capitalize()} Look",
        occasion=body.occasion,
        top_id=body.top_id,
        bottom_id=body.bottom_id,
        shoe_id=body.shoe_id,
        outerwear_id=body.outerwear_id,
        reasoning=body.reasoning,
        tryon_result_url=body.tryon_result_url,
        rating=body.rating or 5,
        is_saved_look=body.is_saved_look
    )
    db.add(outfit)
    db.commit()
    db.refresh(outfit)

    for item_id in (body.top_id, body.bottom_id, body.shoe_id, body.outerwear_id):
        if item_id:
            item = db.query(WardrobeItem).filter(WardrobeItem.id == item_id).first()
            if item:
                item.wear_count += 1
    db.commit()

    return {"message": "Outfit saved successfully!", "outfit_id": outfit.id}


@app.post("/rate-outfit/{outfit_id}")
def rate_outfit(
    outfit_id: int,
    body: RateOutfitRequest,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    outfit = db.query(Outfit).filter(Outfit.id == outfit_id, Outfit.user_id == current_user["id"]).first()
    if not outfit:
        raise HTTPException(404, "Outfit not found.")
    outfit.rating = body.rating
    db.commit()
    return {"message": "Rating saved!", "outfit_id": outfit.id, "rating": body.rating}


@app.get("/calendar/")
def get_calendar(current_user: dict = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    outfits = db.query(Outfit).filter(Outfit.user_id == current_user["id"]).order_by(Outfit.date_worn.desc()).all()
    return [
        {
            "id": o.id,
            "name": o.name,
            "date": o.date_worn.strftime("%Y-%m-%d"),
            "occasion": o.occasion,
            "outfit_id": o.id,
            "top_id": o.top_id,
            "bottom_id": o.bottom_id,
            "shoe_id": o.shoe_id,
            "outerwear_id": o.outerwear_id,
            "rating": o.rating,
            "tryon_result_url": o.tryon_result_url,
            "is_saved_look": o.is_saved_look
        }
        for o in outfits
    ]


@app.get("/stats/")
def get_stats(current_user: dict = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    items = db.query(WardrobeItem).filter(WardrobeItem.user_id == current_user["id"]).all()
    total_items = len(items)

    if total_items == 0:
        return {"message": "Your wardrobe is currently empty. Upload some clothing items to view analytics."}

    outfits = db.query(Outfit).filter(Outfit.user_id == current_user["id"]).all()
    total_outfits_worn = len(outfits)

    most_worn = max(items, key=lambda i: i.wear_count)
    least_worn = min(items, key=lambda i: i.wear_count)

    category_count = {}
    style_count = {}
    color_count = {}
    pattern_count = {}

    for i in items:
        category_count[i.category] = category_count.get(i.category, 0) + 1
        style_count[i.style] = style_count.get(i.style, 0) + 1
        color_count[i.color] = color_count.get(i.color, 0) + 1
        pattern_count[i.pattern] = pattern_count.get(i.pattern, 0) + 1

    occasion_count = {}
    for o in outfits:
        occ = (o.occasion or "other").capitalize()
        occasion_count[occ] = occasion_count.get(occ, 0) + 1

    ratings_count = {str(s): 0 for s in range(1, 6)}
    rated_outfits = [o.rating for o in outfits if o.rating]
    for r in rated_outfits:
        key = str(r)
        if key in ratings_count:
            ratings_count[key] += 1
    avg_rating = round(sum(rated_outfits) / len(rated_outfits), 1) if rated_outfits else 0.0

    needs_review_count = sum(1 for i in items if i.needs_review)

    tops_count = category_count.get("top", 0) + category_count.get("shirt", 0) + category_count.get("t-shirt", 0)
    bottoms_count = category_count.get("bottom", 0) + category_count.get("pant", 0) + category_count.get("jeans", 0) + category_count.get("skirt", 0)
    shoes_count = category_count.get("shoes", 0)
    potential_combos = tops_count * bottoms_count

    versatility_score = 0
    if tops_count > 0 and bottoms_count > 0:
        ratio = min(tops_count, bottoms_count) / max(tops_count, bottoms_count)
        base = 70 if shoes_count > 0 else 50
        versatility_score = min(100, int(base + (ratio * 20) + min(len(style_count) * 2.5, 10)))

    insights = []
    if shoes_count == 0:
        insights.append("Footwear Gap: Add 1-2 pairs of shoes to complete your outfits!")
    dominant_style = max(style_count, key=style_count.get) if style_count else "casual"
    insights.append(f"Style Profile: Your wardrobe heavily features {dominant_style.capitalize()} wear.")

    return {
        "total_items": total_items,
        "total_outfits_worn": total_outfits_worn,
        "needs_review_count": needs_review_count,
        "most_worn_item": {"id": most_worn.id, "category": most_worn.category, "wear_count": most_worn.wear_count},
        "least_worn_item": {"id": least_worn.id, "category": least_worn.category, "wear_count": least_worn.wear_count},
        "category_breakdown": category_count,
        "style_breakdown": style_count,
        "color_breakdown": color_count,
        "pattern_breakdown": pattern_count,
        "occasion_breakdown": occasion_count,
        "rating_breakdown": ratings_count,
        "avg_rating": avg_rating,
        "potential_combos": potential_combos,
        "versatility_score": versatility_score,
        "stylist_advice": insights,
    }


OCCASION_SYNONYMS = {
    "college": ["college", "university", "campus", "class", "lecture", "school", "casual"],
    "office": ["office", "work", "job", "business", "meeting", "presentation"],
    "interview": ["interview", "job interview"],
    "wedding": ["wedding", "marriage", "reception", "ceremony", "festive"],
    "party": ["party", "club", "dinner", "celebration", "gathering", "night out", "date"],
    "home": ["home", "house", "sleep", "night", "bed", "relax", "lounge", "stay home", "pajama", "pyjama"],
}


@app.post("/chatbot/")
def chatbot(
    body: ChatRequest,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    message = body.message.lower()
    found_occasion = next((occ for occ in OCCASION_STYLE_PRIORITY if occ in message), None)

    if not found_occasion:
        for occ, syns in OCCASION_SYNONYMS.items():
            if any(s in message for s in syns):
                found_occasion = occ
                break

    if not found_occasion:
        found_occasion = "casual"

    stylist_res = generate_stylist_recommendation(
        user_id=current_user["id"],
        occasion=found_occasion,
        user_prompt=body.message,
        db=db
    )

    outfit = stylist_res.get("outfit")
    if not outfit or (not outfit.get("top") and not outfit.get("bottom")):
        return {"reply": "I couldn't find a suitable outfit combination in your wardrobe. Try uploading more items!"}

    reasoning = stylist_res.get("reasoning", "")
    top_desc = f"{outfit['top']['color']} {outfit['top']['subcategory']}" if outfit.get("top") else ""
    bottom_desc = f"{outfit['bottom']['color']} {outfit['bottom']['subcategory']}" if outfit.get("bottom") else ""

    reply_text = f"✨ **AI Stylist Advice for {found_occasion.capitalize()}**:\n"
    if top_desc and bottom_desc:
        reply_text += f"Try pairing your **{top_desc}** with **{bottom_desc}**.\n"
    elif top_desc:
        reply_text += f"Wear your **{top_desc}**.\n"

    if reasoning:
        reply_text += f"\n*Stylist Note*: {reasoning}"

    return {
        "reply": reply_text,
        "outfit": outfit,
        "stylist_recommendation": stylist_res
    }


@app.post("/feedback/")
def give_feedback(
    body: FeedbackRequest,
    current_user: dict = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    fb = Feedback(user_id=current_user["id"], outfit_id=body.outfit_id, liked=int(body.liked), comment=body.comment)
    db.add(fb)
    db.commit()
    return {"message": "Feedback saved, thank you!"}
