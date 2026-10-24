from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any


# --- Auth Schemas ---
class UserCreate(BaseModel):
    email: str
    password: str
    name: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


# --- Style Profile Schemas ---
class StyleProfileSchema(BaseModel):
    preferred_styles: List[str] = []
    preferred_colors: List[str] = []
    avoided_colors: List[str] = []
    preferred_fits: List[str] = []
    favorite_categories: List[str] = []
    lifestyle: str = "casual"
    height: Optional[float] = None
    body_proportions: Optional[str] = None


# --- Clothing Analysis Schema ---
class GeminiAnalysisResult(BaseModel):
    category: str = "top"
    subcategory: str = "shirt"
    color: str = "blue"
    secondaryColors: List[str] = []
    pattern: str = "solid"
    material: str = "cotton"
    fit: str = "regular"
    style: str = "casual"
    season: List[str] = ["spring", "summer"]
    occasion: List[str] = ["casual"]
    formality: str = "casual"
    genderSuitability: str = "unisex"
    confidence: float = 0.9
    raw_ai_attributes: Optional[Dict[str, Any]] = None


# --- Wardrobe Item Schemas ---
class ItemUpdateRequest(BaseModel):
    category: Optional[str] = None
    subcategory: Optional[str] = None
    color: Optional[str] = None
    pattern: Optional[str] = None
    style: Optional[str] = None
    fit: Optional[str] = None
    material: Optional[str] = None
    formality: Optional[str] = None
    gender: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None


# --- Outfit & Stylist Schemas ---
class SaveOutfitRequest(BaseModel):
    name: Optional[str] = "Saved Look"
    occasion: str
    top_id: int
    bottom_id: Optional[int] = None
    shoe_id: Optional[int] = None
    outerwear_id: Optional[int] = None
    reasoning: Optional[str] = None
    tryon_result_url: Optional[str] = None
    rating: Optional[int] = 5
    is_saved_look: bool = True


class RateOutfitRequest(BaseModel):
    rating: int  # 1-5


class FeedbackRequest(BaseModel):
    outfit_id: int
    liked: bool
    comment: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class StylistRecommendRequest(BaseModel):
    occasion: str = "casual"
    prompt: Optional[str] = None
    gender: Optional[str] = "all"


# --- TryOn Schemas ---
class TryOnSessionCreate(BaseModel):
    garment_ids: List[int] = []


class TryOnProcessRequest(BaseModel):
    session_token: str
    garment_id: int
    person_image_b64: Optional[str] = None
    user_image_b64: Optional[str] = None
    colab_url: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    skin_tone: Optional[str] = None
    body_type: Optional[str] = None
    complexion_note: Optional[str] = None
