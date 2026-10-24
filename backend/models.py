from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text
from datetime import datetime, timezone
from database import Base


def utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    password_hash = Column(String)
    date_created = Column(DateTime, default=utc_now)


class StyleProfile(Base):
    __tablename__ = "style_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    preferred_styles = Column(Text, default="[]")       # JSON array of strings
    preferred_colors = Column(Text, default="[]")       # JSON array of strings
    avoided_colors = Column(Text, default="[]")         # JSON array of strings
    preferred_fits = Column(Text, default="[]")         # JSON array of strings
    favorite_categories = Column(Text, default="[]")    # JSON array of strings
    lifestyle = Column(String, default="casual")        # e.g., "office worker", "student", "freelancer"
    height = Column(Float, nullable=True)               # in cm
    body_proportions = Column(String, nullable=True)    # optional fit note
    date_updated = Column(DateTime, default=utc_now, onupdate=utc_now)


class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=1, index=True)    # Default 1 for backwards compatibility
    image_path = Column(String)                         # Relative local path: uploads/filename.jpg
    thumbnail_path = Column(String, nullable=True)
    cloudinary_url = Column(String, nullable=True)
    
    category = Column(String, default="unknown")        # top, bottom, dress, shoes, outerwear
    subcategory = Column(String, default="unknown")     # shirt, t-shirt, pant, jeans, blazer, jacket, etc.
    color = Column(String, default="unknown")           # primary color
    secondary_colors = Column(Text, default="[]")       # JSON array
    pattern = Column(String, default="solid")           # solid, striped, checked, floral, printed
    material = Column(String, default="unknown")        # cotton, denim, leather, wool, silk, polyester
    fit = Column(String, default="regular")             # slim, regular, loose, oversized
    style = Column(String, default="casual")            # formal, casual, party wear, sporty, smart casual
    season = Column(Text, default='["spring", "summer", "autumn", "winter"]') # JSON array
    occasion = Column(Text, default='["casual"]')       # JSON array
    formality = Column(String, default="casual")        # formal, semi-formal, casual
    gender = Column(String, default="unisex")           # male, female, unisex
    
    brand = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    confidence = Column(Float, default=1.0)
    needs_review = Column(Boolean, default=False)
    wear_count = Column(Integer, default=0)
    
    ai_generated_attributes = Column(Text, nullable=True) # Full structured JSON string from Gemini / CLIP
    user_edited_attributes = Column(Text, nullable=True)  # JSON string tracking manual user fixes
    date_added = Column(DateTime, default=utc_now)


class Outfit(Base):
    __tablename__ = "outfits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=1, index=True)
    name = Column(String, default="Saved Look")
    occasion = Column(String)
    top_id = Column(Integer)
    bottom_id = Column(Integer, nullable=True)
    shoe_id = Column(Integer, nullable=True)
    outerwear_id = Column(Integer, nullable=True)
    reasoning = Column(Text, nullable=True)
    tryon_result_url = Column(String, nullable=True)
    rating = Column(Integer, nullable=True)
    is_saved_look = Column(Boolean, default=False)
    date_worn = Column(DateTime, default=utc_now)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=1)
    outfit_id = Column(Integer)
    liked = Column(Integer)
    comment = Column(String, nullable=True)


class TryOnSession(Base):
    __tablename__ = "tryon_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=1, index=True)
    session_token = Column(String, unique=True, index=True)
    status = Column(String, default="IDLE")             # IDLE, REQUESTING_CAMERA, CONNECTING, LIVE, PROCESSING, ERROR
    active_garment_ids = Column(Text, default="[]")     # JSON array of selected item IDs
    result_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=utc_now)
