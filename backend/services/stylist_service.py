import os
import json
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from models import WardrobeItem, StyleProfile
from recommender import recommend_outfit

logger = logging.getLogger(__name__)

STYLIST_SYSTEM_PROMPT = """
You are StyleSense AI, a high-end personal fashion stylist.
You recommend complete, highly coordinated outfits from the user's ACTUAL wardrobe items.

CRITICAL RULES:
1. You MUST ONLY recommend clothing items using the exact item IDs provided in the CANDIDATE WARDROBE list. DO NOT invent or hallucinate item IDs.
2. Ensure color harmony, pattern balance, and formality alignment appropriate for the occasion and user profile.
3. Output MUST be a single valid JSON object with NO markdown formatting, NO backticks, and NO conversational wrapping text:

{
  "occasion": "office",
  "outfit": {
    "top_id": 12,            // Must be an integer ID from candidates (or null)
    "bottom_id": 15,         // Must be an integer ID from candidates (or null for dress)
    "shoe_id": 8,            // Integer ID from candidates (or null if unavailable)
    "outerwear_id": null     // Integer ID from candidates (or null if unneeded)
  },
  "reasoning": "The white cotton Oxford shirt paired with navy trousers creates a crisp, professional formal look.",
  "confidence": 0.92
}
"""


def generate_stylist_recommendation(
    user_id: int,
    occasion: str,
    user_prompt: Optional[str],
    db: Session,
    gender: str = "all"
) -> Dict[str, Any]:
    """Generates an AI Stylist recommendation using candidate pre-filtering and Gemini reasoning."""
    # 1. Fetch user wardrobe items
    query = db.query(WardrobeItem).filter(WardrobeItem.user_id == user_id)
    all_items = query.all()

    if not all_items:
        return {
            "occasion": occasion,
            "outfit": None,
            "reasoning": "Your wardrobe is currently empty. Upload some clothing items to get AI styling advice!",
            "confidence": 0.0,
            "source": "empty_wardrobe"
        }

    # 2. Fetch User Style Profile
    profile = db.query(StyleProfile).filter(StyleProfile.user_id == user_id).first()
    profile_data = {}
    if profile:
        def _parse(val):
            try:
                return json.loads(val) if val else []
            except Exception:
                return []
        profile_data = {
            "preferred_styles": _parse(profile.preferred_styles),
            "preferred_colors": _parse(profile.preferred_colors),
            "avoided_colors": _parse(profile.avoided_colors),
            "lifestyle": profile.lifestyle,
            "height": profile.height
        }

    # 3. Candidate Pre-Filtering (compact representation for Gemini)
    candidates = []
    valid_ids = set()
    for item in all_items:
        valid_ids.add(item.id)
        candidates.append({
            "id": item.id,
            "category": item.category,
            "subcategory": item.subcategory or item.category,
            "color": item.color,
            "pattern": item.pattern,
            "style": item.style,
            "formality": item.formality,
            "gender": getattr(item, "gender", "unisex") or "unisex"
        })

    # 4. Try Gemini AI Stylist if API key is present
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            prompt_content = f"""
USER STYLIST REQUEST:
Occasion: {occasion}
Prompt: {user_prompt or "Suggest the best look for this occasion."}

USER STYLE PROFILE:
{json.dumps(profile_data, indent=2)}

CANDIDATE WARDROBE ITEMS (Choose top_id, bottom_id, shoe_id, outerwear_id ONLY from these IDs):
{json.dumps(candidates, indent=2)}
"""

            response = None
            for model_name in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[STYLIST_SYSTEM_PROMPT, prompt_content]
                    )
                    if response and response.text:
                        break
                except Exception as e:
                    logger.warning(f"Stylist model {model_name} failed: {e}")
                    continue

            if response and response.text:
                text = response.text.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()

                res_json = json.loads(text)
                outfit = res_json.get("outfit", {})

                # Validate returned IDs against candidate wardrobe to prevent hallucinations
                top_id = outfit.get("top_id") if outfit.get("top_id") in valid_ids else None
                bottom_id = outfit.get("bottom_id") if outfit.get("bottom_id") in valid_ids else None
                shoe_id = outfit.get("shoe_id") if outfit.get("shoe_id") in valid_ids else None
                outerwear_id = outfit.get("outerwear_id") if outfit.get("outerwear_id") in valid_ids else None

                if top_id or bottom_id:
                    # Enrich with full item objects from DB
                    items_by_id = {i.id: i for i in all_items}
                    top_obj = items_by_id.get(top_id)
                    bottom_obj = items_by_id.get(bottom_id)
                    shoe_obj = items_by_id.get(shoe_id)
                    outerwear_obj = items_by_id.get(outerwear_id)

                    return {
                        "occasion": occasion,
                        "outfit": {
                            "top": _item_to_dict(top_obj) if top_obj else None,
                            "bottom": _item_to_dict(bottom_obj) if bottom_obj else None,
                            "shoes": _item_to_dict(shoe_obj) if shoe_obj else None,
                            "outerwear": _item_to_dict(outerwear_obj) if outerwear_obj else None,
                        },
                        "reasoning": res_json.get("reasoning", "A tailored outfit selection matching your style."),
                        "confidence": res_json.get("confidence", 0.9),
                        "source": "gemini_stylist"
                    }

        except Exception as e:
            logger.warning(f"Gemini Stylist recommendation failed: {e}. Falling back to matrix recommender.")

    # 5. Fallback to heuristic matrix recommender if Gemini is absent or fails
    fallback_res = recommend_outfit(all_items, occasion, gender=gender)
    top_item = fallback_res.get("top")
    bottom_item = fallback_res.get("bottom")
    shoe_item = fallback_res.get("shoes")
    outerwear_item = fallback_res.get("jacket")

    return {
        "occasion": occasion,
        "outfit": {
            "top": top_item,
            "bottom": bottom_item,
            "shoes": shoe_item,
            "outerwear": outerwear_item
        },
        "reasoning": fallback_res.get("note", "Recommended based on style priority and color harmony."),
        "confidence": (fallback_res.get("match_score", 80) / 100.0),
        "source": "matrix_recommender"
    }


def _item_to_dict(item: Optional[WardrobeItem]) -> Optional[Dict[str, Any]]:
    if not item:
        return None
    img_path = (item.image_path or "").replace("\\", "/")
    return {
        "id": item.id,
        "category": item.category,
        "subcategory": item.subcategory or item.category,
        "color": item.color,
        "pattern": item.pattern,
        "style": item.style,
        "formality": item.formality,
        "image_path": img_path,
        "image_url": item.cloudinary_url or f"/{img_path}"
    }
