import os
import json
import logging
from typing import Dict, Any, Optional
from PIL import Image

from ai_detector import detect_attributes as clip_detect_attributes

logger = logging.getLogger(__name__)

GEMINI_PROMPT = """
You are a senior computer-vision fashion expert and fashion stylist.
Analyze this clothing image and extract precise fashion attributes in strict JSON format.

Output MUST be a single valid JSON object with NO markdown formatting, NO backticks, and NO conversational text:

{
  "category": "top",             // Options: "top", "bottom", "dress", "shoes", "outerwear"
  "subcategory": "t-shirt",      // e.g. "shirt", "t-shirt", "pant", "jeans", "blazer", "jacket", "sneakers", "boots", "dress", "hoodie"
  "color": "blue",               // Primary color: "red", "blue", "black", "white", "green", "yellow", "brown", "grey", "pink", "beige", "orange", "purple", "olive"
  "secondaryColors": [],         // Array of secondary colors detected
  "pattern": "solid",            // Options: "solid", "striped", "checked", "floral", "printed"
  "material": "cotton",          // e.g. "cotton", "denim", "leather", "wool", "silk", "polyester", "linen", "nylon", "suede"
  "fit": "regular",              // Options: "slim", "regular", "loose", "oversized"
  "style": "casual",             // Options: "formal", "casual", "party wear", "sporty", "smart casual", "nightwear"
  "season": ["spring", "summer"],// Array subset of: ["spring", "summer", "autumn", "winter"]
  "occasion": ["casual"],        // Array subset of: ["casual", "office", "interview", "wedding", "party", "home", "sport"]
  "formality": "casual",         // Options: "formal", "semi-formal", "casual"
  "genderSuitability": "unisex", // Options: "male", "female", "unisex"
  "confidence": 0.95
}
"""


def analyze_clothing_image(image_path: str) -> Dict[str, Any]:
    """Analyzes clothing image using Gemini Multimodal API with fallback to CLIP zero-shot classification."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.info("GEMINI_API_KEY not set. Using fast CLIP zero-shot detector.")
        return _clip_fallback_result(image_path)

    try:
        # Try importing official google-genai SDK
        from google import genai
        client = genai.Client(api_key=api_key)

        pil_img = Image.open(image_path).convert("RGB")

        # Try gemini-2.5-flash or gemini-1.5-flash
        response = None
        for model_name in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[pil_img, GEMINI_PROMPT]
                )
                if response and response.text:
                    break
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue

        if not response or not response.text:
            logger.warning("Gemini returned empty response. Falling back to CLIP.")
            return _clip_fallback_result(image_path)

        text = response.text.strip()

        # Strip triple backticks if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed = json.loads(text)
        parsed["source"] = "gemini"
        parsed["needs_review"] = parsed.get("confidence", 0.9) < 0.75
        return parsed

    except Exception as e:
        logger.warning(f"Gemini API analysis failed: {e}. Falling back to CLIP.")
        return _clip_fallback_result(image_path)


def _clip_fallback_result(image_path: str) -> Dict[str, Any]:
    """Executes local PyTorch CLIP detector and formats into full attribute dict."""
    clip_res = clip_detect_attributes(image_path)

    # Normalize category mapping
    raw_cat = clip_res.get("category", "shirt")
    category = "top"
    if raw_cat in ("shirt", "t-shirt"):
        category = "top"
    elif raw_cat in ("pant", "jeans"):
        category = "bottom"
    elif raw_cat == "skirt":
        category = "bottom"
    elif raw_cat == "dress":
        category = "dress"
    elif raw_cat == "shoes":
        category = "shoes"
    elif raw_cat == "jacket":
        category = "outerwear"

    return {
        "category": category,
        "subcategory": raw_cat,
        "color": clip_res.get("color", "unknown"),
        "secondaryColors": [],
        "pattern": clip_res.get("pattern", "solid"),
        "material": "unknown",
        "fit": "regular",
        "style": clip_res.get("style", "casual"),
        "season": ["spring", "summer", "autumn", "winter"],
        "occasion": ["casual"],
        "formality": "casual" if clip_res.get("style") != "formal" else "formal",
        "genderSuitability": clip_res.get("gender", "unisex"),
        "confidence": clip_res.get("confidence", 0.9),
        "needs_review": clip_res.get("needs_review", False),
        "source": "clip_fallback",
        "raw_clip": clip_res
    }
