"""
Outfit recommendation with occasion dress codes.

Why the old model suggested t-shirts at weddings:
  TOP_CATEGORIES = ["shirt", "t-shirt"] for EVERY occasion, so a t-shirt
  mislabeled by CLIP as "party wear" (id=39, green t-shirt) became a valid
  "wedding top". Category appropriateness was never enforced.

Fix: each occasion now defines ALLOWED categories + ranked styles.
   wedding/office/interview ban t-shirt and jeans entirely — no style label
   can smuggle them in. Dresses are first-class outfits. Pairs are scored
   on style tier + wear rotation + confidence + color/pattern harmony.
"""

import random as _random

try:
    from ai_detector import CATEGORIES as _FINE_CATS
except ImportError:  # pragma: no cover - ai_detector always present in prod
    _FINE_CATS = ["shirt", "t-shirt", "pant", "jeans", "dress", "jacket", "shoes", "skirt"]


def _fine_cat(item):
    """Single taxonomy of record at READ time: subcategory wins when valid,
    else the stored category. Keeps coarse Gemini labels (top/bottom/
    outerwear) combinable even if a write path stored them raw."""
    sub = (getattr(item, "subcategory", "") or "").strip()
    if sub in _FINE_CATS:
        return sub
    return getattr(item, "category", "")

OCCASION_STYLE_PRIORITY = {
    "college": ["casual", "sporty", "formal"],
    "office": ["formal", "casual"],
    "interview": ["formal"],
    "wedding": ["party wear", "formal"],
    "party": ["party wear", "casual", "formal"],
    "home": ["nightwear", "casual"],
}

# Strict dress code per occasion. Anything not listed here is BANNED for
# that occasion, regardless of its style label.
OCCASION_DRESSCODE = {
    "college": {
        "tops": ["shirt", "t-shirt"],
        "bottoms": ["pant", "jeans", "skirt"],
        "styles": ["casual", "sporty", "formal"],
        "allow_dress": False,
        "dress_styles": [],
    },
    "office": {
        "tops": ["shirt"],
        "bottoms": ["pant"],
        "styles": ["formal", "casual"],
        "allow_dress": True,
        "dress_styles": ["formal", "casual"],
    },
    "interview": {
        "tops": ["shirt"],
        "bottoms": ["pant"],
        "styles": ["formal"],
        "allow_dress": True,
        "dress_styles": ["formal"],
    },
    "wedding": {
        "tops": ["shirt"],
        "bottoms": ["pant"],
        "styles": ["party wear", "formal"],
        "allow_dress": True,
        "dress_styles": ["party wear", "formal"],
    },
    "party": {
        "tops": ["shirt", "t-shirt"],
        "bottoms": ["pant", "jeans", "skirt"],
        "styles": ["party wear", "formal", "casual"],
        "allow_dress": True,
        "dress_styles": ["party wear", "formal", "casual"],
    },
    "home": {
        # Nightwear finally has a correct home: sleep/lounge pieces are
        # banned from every going-out occasion by omission above.
        "tops": ["shirt", "t-shirt"],
        "bottoms": ["pant", "jeans", "skirt"],
        "styles": ["nightwear", "casual"],
        "allow_dress": True,
        "dress_styles": ["nightwear", "casual"],
    },
}

SHOE_CATEGORIES = ["shoes"]
JACKET_CATEGORIES = ["jacket"]

VALID_GENDERS = ("all", "male", "female")

# Gender synonyms accepted from API / chatbot ("mens" -> "male", ...).
GENDER_ALIASES = {
    "male": "male", "men": "male", "mens": "male", "man": "male", "boy": "male", "boys": "male",
    "masculine": "male", "gents": "male", "gentlemen": "male",
    "female": "female", "women": "female", "womens": "female", "woman": "female", "girl": "female",
    "girls": "female", "feminine": "female", "ladies": "female",
    "all": "all", "any": "all", "unisex": "all", "both": "all", "everyone": "all",
}


def normalize_gender(value):
    """Map free text to all/male/female. Unknown -> all (never hide items)."""
    if not value:
        return "all"
    return GENDER_ALIASES.get(str(value).strip().lower(), "all")


def _gender_ok(item_gender, want):
    """Unisex (or unknown) items are visible under every filter; only an
    explicit opposite-gender label hides an item."""
    if want == "all":
        return True
    g = (item_gender or "unisex").strip().lower()
    return g == want or g in ("unisex", "unknown", "")

NEUTRAL_COLORS = {"black", "white", "grey", "beige", "brown"}

# ---- Stylist color science -------------------------------------------------
# hue: degrees on a 12-step wheel (None = neutral, goes with everything).
# light: 0-100 perceived lightness. sat: 0-100 saturation.
# formal: True if the color reads formal (dark/muted/neutral).
COLOR_PROFILE = {
    "black":  {"hue": None, "light": 8,  "sat": 0,  "formal": True},
    "white":  {"hue": None, "light": 95, "sat": 0,  "formal": True},
    "grey":   {"hue": None, "light": 55, "sat": 0,  "formal": True},
    "beige":  {"hue": 45,   "light": 78, "sat": 25, "formal": True},
    "brown":  {"hue": 25,   "light": 32, "sat": 45, "formal": True},
    "blue":   {"hue": 220,  "light": 38, "sat": 60, "formal": True},
    "red":    {"hue": 0,    "light": 45, "sat": 75, "formal": False},
    "green":  {"hue": 140,  "light": 42, "sat": 45, "formal": False},
    "yellow": {"hue": 55,   "light": 75, "sat": 70, "formal": False},
    "orange": {"hue": 30,   "light": 55, "sat": 75, "formal": False},
    "pink":   {"hue": 340,  "light": 72, "sat": 55, "formal": False},
    "purple": {"hue": 280,  "light": 40, "sat": 55, "formal": False},
    "olive":  {"hue": 70,   "light": 42, "sat": 45, "formal": True},
}

# Traditional shoe rules: which bottom colors each shoe color serves.
SHOE_MATCH = {
    "black": {"black", "grey", "blue", "white", "olive"},
    "brown": {"beige", "brown", "blue", "green", "white", "olive"},
    "white": {"white", "beige", "blue", "grey", "pink"},
    "grey":  {"grey", "black", "blue", "white"},
    "beige": {"beige", "white", "brown", "blue"},
    "blue":  {"blue", "white", "grey"},
    "green": {"green", "beige", "brown"},
    "red":   {"red", "black", "white"},
    "pink":  {"pink", "white", "grey", "blue"},
    "purple": {"purple", "black", "white"},
    "orange": {"orange", "white", "beige"},
    "yellow": {"yellow", "white", "blue"},
    "olive": {"olive", "beige", "brown", "white", "black"},
}

SKIN_TONE_COLOR_AFFINITY = {
    "light": {
        "best": {"blue", "navy", "black", "green", "purple", "red", "grey"},
        "avoid": {"beige", "yellow"},
    },
    "fair": {
        "best": {"blue", "navy", "black", "green", "purple", "red", "grey"},
        "avoid": {"beige", "yellow"},
    },
    "medium": {
        "best": {"olive", "beige", "brown", "blue", "green", "white", "black", "yellow", "red"},
        "avoid": set(),
    },
    "wheatish": {
        "best": {"olive", "beige", "brown", "blue", "green", "white", "black", "yellow", "red"},
        "avoid": set(),
    },
    "dark": {
        "best": {"white", "yellow", "pink", "orange", "blue", "beige", "green", "gold", "red"},
        "avoid": {"black", "brown"},
    },
    "deep": {
        "best": {"white", "yellow", "pink", "orange", "blue", "beige", "green", "gold", "red"},
        "avoid": {"black", "brown"},
    }
}


def _hue_distance(h1, h2):
    if h1 is None or h2 is None:
        return None
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def _color_score(t_col, b_col, occasion):
    """Lower = better. Returns (score, notes[])."""
    notes = []
    score = 0.0
    tp = COLOR_PROFILE.get(t_col, {})
    bp = COLOR_PROFILE.get(b_col, {})
    t_neutral = t_col in NEUTRAL_COLORS
    b_neutral = b_col in NEUTRAL_COLORS

    # Both neutral = safe but flat; one neutral = ideal anchor.
    if t_neutral and b_neutral:
        if t_col == b_col and t_col not in ("black", "white"):
            score += 8.0
            notes.append("same neutral head-to-toe looks flat")
        elif t_col == "white" and b_col == "white":
            score += 6.0
            notes.append("white-on-white washes out")
        else:
            score += 0.5
    elif b_neutral:
        score -= 3.0
        notes.append("neutral bottom anchors the look")
    elif t_neutral:
        score -= 1.0
    else:
        # Two chromatic colors: use hue-wheel harmony.
        d = _hue_distance(tp.get("hue"), bp.get("hue"))
        if d is not None:
            if d < 25:
                score += 7.0
                notes.append(f"{t_col} + {b_col} are too close, looks like a uniform")
            elif d < 65:
                score -= 2.0
                notes.append("analogous colors blend smoothly")
            elif 140 <= d <= 210:
                score -= 2.5
                notes.append("complementary colors balance each other")
            elif 90 <= d < 140:
                score += 4.0
                notes.append(f"{t_col} + {b_col} clash at this distance")
        # Two loud colors fight; mute with a neutral instead.
        if tp.get("sat", 0) >= 60 and bp.get("sat", 0) >= 60:
            score += 5.0
            notes.append("two saturated brights compete")

    # Lightness contrast: classic tailoring wants top/bottom separation.
    tl, bl = tp.get("light", 50), bp.get("light", 50)
    contrast = abs(tl - bl)
    if contrast < 10 and not (t_col == "black" and b_col == "black"):
        score += 4.0
        notes.append("low contrast, pieces melt together")
    elif 25 <= contrast <= 70:
        score -= 2.0
        notes.append("good light/dark contrast")

    # Formal occasions want a formal palette (dark/muted/neutral).
    # Strict rooms (interview first) punish loud colors hard — a pink shirt
    # must never outrank white/blue for an interview.
    if occasion in ("wedding", "office", "interview"):
        informal = [c for c in (t_col, b_col)
                    if c and not COLOR_PROFILE.get(c, {}).get("formal", False)]
        if informal:
            weight = 5.0 if occasion == "interview" else 4.0
            score += weight * len(informal)
            notes.append(f"{'/'.join(informal)} reads casual for {occasion}")
    return score, notes


def _pattern_formality_score(t_pat, b_pat, occasion):
    """Small penalty for playful patterns in strict rooms (tops/bottoms only —
    dresses play by their own rules)."""
    if occasion not in ("wedding", "office", "interview"):
        return 0.0, []
    table = {"interview": {"floral": 3.0, "printed": 2.0},
             "office": {"floral": 2.0, "printed": 1.0},
             "wedding": {"floral": 2.0, "printed": 1.0}}
    weights = table[occasion]
    score, notes = 0.0, []
    for pat in ((t_pat or "").lower(), (b_pat or "").lower()):
        if pat in weights:
            score += weights[pat]
    if score:
        notes.append(f"playful pattern for {occasion}")
    return score, notes


def _pattern_score(t_pat, b_pat):
    """Lower = better. Returns (score, notes[])."""
    notes = []
    score = 0.0
    t_pat = (t_pat or "solid").lower()
    b_pat = (b_pat or "solid").lower()
    if t_pat == "solid" and b_pat == "solid":
        score += 0.0
    elif t_pat != "solid" and b_pat == "solid":
        score -= 2.0
        notes.append("patterned top balanced with solid bottom")
    elif t_pat == "solid" and b_pat != "solid":
        score += 1.0
        notes.append("patterned bottom needs a calm solid top")
    else:
        score += 10.0
        if t_pat == b_pat:
            score += 5.0
            notes.append(f"double {t_pat} clashes")
        else:
            notes.append(f"{t_pat} + {b_pat} compete; wear one pattern at a time")
    return score, notes


def _shoe_score(shoe_color, bottom_color, shoe_style, outfit_worst_tier, priority, shoe_pattern="solid"):
    """Lower = better. Color link to bottom + style-tier coherence + pattern
    discipline (loud patterned shoes, e.g. floral purple boots, sink fast)."""
    score = 0.0
    notes = []
    if shoe_color and bottom_color:
        if shoe_color == bottom_color:
            score -= 2.5
            notes.append(f"{shoe_color} shoes echo the {bottom_color} bottoms")
        elif bottom_color in SHOE_MATCH.get(shoe_color, set()) or shoe_color in NEUTRAL_COLORS:
            score -= 1.5
        else:
            score += 4.0
            notes.append(f"{shoe_color} shoes fight {bottom_color} bottoms")
    pat = (shoe_pattern or "solid").lower()
    if pat in ("floral", "printed"):
        score += 5.0
        notes.append(f"{pat} shoes steal the outfit — prefer solid footwear")
    elif pat != "solid":
        score += 3.5
        notes.append(f"{pat} shoes are loud — solid shoes match more")
    s_tier = _tier(shoe_style, priority)
    # Shoes are rarely labeled "party wear", so a one-tier gap (e.g. formal
    # shoes with a party outfit) is normal and free. Larger gaps mean the
    # shoes read clearly more casual than the clothes.
    if s_tier > outfit_worst_tier + 1:
        gap = s_tier - (outfit_worst_tier + 1)
        score += 4.0 * gap
        notes.append("shoes read more casual than the outfit")
    return score, notes


def _tier(style, priority):
    try:
        return priority.index(style)
    except ValueError:
        return 999


def _item_quality(item, tier, gender_bonus=0.0, occasion="college"):
    """Lower = better. Least-worn first, then high confidence, reviewed items.
    gender_bonus is negative (preferred) when the item's own gender label
    exactly matches the requested filter — user-confirmed items outrank
    generic unisex ones without hiding anything."""
    wear = getattr(item, "wear_count", 0) or 0
    conf = getattr(item, "confidence", 1.0) or 0.0
    review_penalty = 5.0 if getattr(item, "needs_review", False) else 0.0
    score = tier * 50.0 + wear * 3.0 - conf * 5.0 + review_penalty + gender_bonus

    # Occasion-specific style tuning (Party Wear specialization)
    if occasion == "party":
        cat = (getattr(item, "category", "") or "").lower()
        subcat = (getattr(item, "subcategory", "") or "").lower()
        style = (getattr(item, "style", "") or "").lower()
        col = (getattr(item, "color", "") or "").lower()
        pat = (getattr(item, "pattern", "") or "").lower()

        # Explicit party wear items get heavy priority
        if style == "party wear":
            score -= 30.0
        elif style == "formal" or "blazer" in subcat or "jacket" in cat:
            score -= 15.0

        # Party colors (black, red, blue, wine, dark tones) get extra appeal
        if col in ("black", "red", "blue", "purple", "olive", "grey"):
            score -= 8.0

        # Printed or stylish patterns outrank plain basic items for parties
        if pat in ("printed", "striped", "checked", "floral"):
            score -= 5.0

        # Basic plain t-shirts get penalized for party wear unless they have party style
        if cat == "t-shirt" and style == "casual" and pat == "solid":
            score += 25.0

    return score


def _gender_rank_bonus(item, want):
    if want != "all" and (getattr(item, "gender", "unisex") or "unisex").strip().lower() == want:
        return -2.5
    return 0.0


def _pair_harmony(top, bottom, occasion="college"):
    """Penalty (positive) or bonus (negative) for color/pattern combos."""
    t_col = (getattr(top, "color", "") or "").lower()
    b_col = (getattr(bottom, "color", "") or "").lower()
    t_pat = (getattr(top, "pattern", "") or "").lower()
    b_pat = (getattr(bottom, "pattern", "") or "").lower()
    c_score, _ = _color_score(t_col, b_col, occasion)
    p_score, _ = _pattern_score(t_pat, b_pat)
    f_score, _ = _pattern_formality_score(t_pat, b_pat, occasion)
    return c_score + p_score + f_score


def _explain_pair(top, bottom, occasion):
    """Human-readable stylist notes for a pair. Returns (notes[], total)."""
    t_col = (getattr(top, "color", "") or "").lower()
    b_col = (getattr(bottom, "color", "") or "").lower()
    t_pat = (getattr(top, "pattern", "") or "").lower()
    b_pat = (getattr(bottom, "pattern", "") or "").lower()
    c_score, c_notes = _color_score(t_col, b_col, occasion)
    p_score, p_notes = _pattern_score(t_pat, b_pat)
    f_score, f_notes = _pattern_formality_score(t_pat, b_pat, occasion)
    return c_notes + p_notes + f_notes, c_score + p_score + f_score


def _jacket_score(jacket, top, bottom, occasion):
    """Lower = better. A layer must earn its place: neutral (or outfit-echo)
    colors, calm patterns, coherent tier. Returns (score, notes[])."""
    notes = []
    score = 0.0
    j_col = (getattr(jacket, "color", "") or "").lower()
    j_pat = (getattr(jacket, "pattern", "") or "").lower()
    t_col = (getattr(top, "color", "") or "").lower()
    b_col = (getattr(bottom, "color", "") or "").lower()
    t_pat = (getattr(top, "pattern", "") or "").lower()
    # Jacket echoing the top reads as "two shirts" — forbid same color+pattern.
    if j_col == t_col and j_pat == (t_pat or "solid"):
        score += 12.0
        notes.append(f"{j_col} jacket doubles the {t_col} top")
    # Prefer neutral jackets; loud-on-loud layers fight the outfit.
    if j_col not in NEUTRAL_COLORS:
        j_sat = COLOR_PROFILE.get(j_col, {}).get("sat", 50)
        t_sat = COLOR_PROFILE.get(t_col, {}).get("sat", 0)
        if t_sat >= 55 and j_sat >= 55:
            score += 6.0
            notes.append(f"loud {j_col} layer fights the {t_col} top")
        else:
            score += 2.0
    else:
        score -= 1.5
    # Two patterns (top + jacket) is one too many.
    if j_pat != "solid" and (t_pat or "solid") != "solid":
        score += 5.0
        notes.append("patterned layer over a patterned top is busy")
    return score, notes


def _serialize(item):
    if item is None:
        return None
    path = (getattr(item, "image_path", "") or "").replace("\\", "/")
    return {
        "id": item.id,
        "path": path,
        "category": getattr(item, "category", ""),
        "color": getattr(item, "color", ""),
        "pattern": getattr(item, "pattern", ""),
        "style": getattr(item, "style", ""),
        "gender": getattr(item, "gender", "unisex") or "unisex",
    }


def _ranked_sets(items, tops, bottoms, priority, exclude_ids, occasion="college", want_gender="all"):
    pairs = []
    for top in tops:
        if top.id in exclude_ids:
            continue
        t_tier = _tier(getattr(top, "style", ""), priority)
        if t_tier >= 999:
            continue
        for bottom in bottoms:
            if bottom.id in exclude_ids:
                continue
            b_tier = _tier(getattr(bottom, "style", ""), priority)
            if b_tier >= 999:
                continue
            worst = max(t_tier, b_tier)
            score = (_item_quality(top, t_tier, _gender_rank_bonus(top, want_gender), occasion=occasion)
                     + _item_quality(bottom, b_tier, _gender_rank_bonus(bottom, want_gender), occasion=occasion)
                     + _pair_harmony(top, bottom, occasion))
            pairs.append((worst, score, top, bottom, t_tier, b_tier))
    pairs.sort(key=lambda p: (p[0], p[1]))
    return pairs


def _ranked_dresses(dresses, priority, exclude_ids, want_gender="all", occasion="college"):
    out = []
    for d in dresses:
        if d.id in exclude_ids:
            continue
        tier = _tier(getattr(d, "style", ""), priority)
        if tier >= 999:
            continue
        out.append((tier, _item_quality(d, tier, _gender_rank_bonus(d, want_gender), occasion=occasion), d))
    out.sort(key=lambda p: (p[0], p[1]))
    return out


def _ranked_shoes(items, priority, exclude_ids, want_gender="all"):
    exclude_ids = exclude_ids or set()
    cands = [i for i in items
             if _fine_cat(i) in SHOE_CATEGORIES
             and _tier(getattr(i, "style", ""), priority) < 999
             and i.id not in exclude_ids]
    if not cands and exclude_ids:
        cands = [i for i in items
                 if _fine_cat(i) in SHOE_CATEGORIES
                 and _tier(getattr(i, "style", ""), priority) < 999]
    if not cands:
        cands = [i for i in items if _fine_cat(i) in SHOE_CATEGORIES]
    cands.sort(key=lambda i: (_tier(getattr(i, "style", ""), priority),
                              _item_quality(i, 0, _gender_rank_bonus(i, want_gender))))
    return cands


def _best_shoe(items, priority, exclude_ids):
    ranked = _ranked_shoes(items, priority, exclude_ids)
    if not ranked:
        return None, None
    best = ranked[0]
    return best, _tier(getattr(best, "style", ""), priority)


def _item_sig(item):
    return (getattr(item, "category", ""), getattr(item, "color", ""),
            getattr(item, "pattern", ""), getattr(item, "style", ""))


def _pick_rotated_shoe(shoe_pool, shoe_use):
    """Least-used shoe in the pool (missing key counts as 0)."""
    if not shoe_pool:
        return None
    least = min(shoe_use.get(c.id, 0) for c in shoe_pool)
    for cand in shoe_pool:
        if shoe_use.get(cand.id, 0) == least:
            return cand
    return shoe_pool[0]


def _match_percent(worst_tier, harmony, shoe_adj=0.0):
    pct = (95 - worst_tier * 12
           + min(8.0, max(0.0, -harmony))
           - max(0.0, harmony) * 1.6
           - max(0.0, shoe_adj) * 1.2)
    return int(max(5, min(99, round(pct))))


def _build_set_option(occasion, priority, top, bottom, t_tier, b_tier,
                      shoe, jacket):
    worst = max(t_tier, b_tier)
    style_notes, harmony = _explain_pair(top, bottom, occasion)
    bottom_color = (getattr(bottom, "color", "") or "").lower()
    shoe_adj, shoe_notes = (0.0, [])
    if shoe is not None:
        shoe_adj, shoe_notes = _shoe_score(
            (getattr(shoe, "color", "") or "").lower(), bottom_color,
            getattr(shoe, "style", ""), worst, priority,
            getattr(shoe, "pattern", "solid"))
    notes = style_notes + shoe_notes
    is_fallback = worst > 0
    harmony_line = ""
    if harmony <= -3:
        harmony_line = " Colors and pattern work well together."
    elif harmony >= 8:
        harmony_line = " Bold pairing — keep accessories minimal."
    layer_line = ""
    if jacket is not None:
        layer_line = f" Optional layer: {jacket.color} {jacket.pattern} jacket."
        notes = notes + ["layer adds depth without doubling the top"]
    return {
        "occasion": occasion,
        "outfit_type": "set",
        "is_fallback": is_fallback,
        "rank_tier": worst,
        "match_score": _match_percent(worst, harmony, shoe_adj),
        "match_notes": notes[:4],
        "note": ("Perfect match found!"
                 if not is_fallback
                 else f"No '{priority[0]}' set available — showing closest match ({priority[worst]})."),
        "explanation": (f"{top.color} {top.category} ({top.style}) + {bottom.color} {bottom.category} ({bottom.style})."
                        f"{harmony_line}{layer_line}"),
        "top": _serialize(top),
        "bottom": _serialize(bottom),
        "dress": None,
        "shoes": _serialize(shoe),
        "jacket": _serialize(jacket),
        "purchase_suggestion": None if shoe else f"Consider adding '{priority[0]}' style shoes to complete this look!",
    }


def _build_dress_option(occasion, priority, dress, d_tier, shoe):
    shoe_adj, shoe_notes = (0.0, [])
    if shoe is not None:
        shoe_adj, shoe_notes = _shoe_score(
            (getattr(shoe, "color", "") or "").lower(),
            (getattr(dress, "color", "") or "").lower(),
            getattr(shoe, "style", ""), d_tier, priority,
            getattr(shoe, "pattern", "solid"))
    notes = [f"{dress.color} {dress.pattern} one-piece"] + shoe_notes
    is_fallback = d_tier > 0
    return {
        "occasion": occasion,
        "outfit_type": "dress",
        "is_fallback": is_fallback,
        "rank_tier": d_tier,
        "match_score": _match_percent(d_tier, 0.0, shoe_adj),
        "match_notes": notes[:4],
        "note": ("Perfect match found!"
                 if not is_fallback
                 else f"No '{priority[0]}' dress available — showing closest match ({priority[d_tier]})."),
        "explanation": (f"{dress.color} {dress.pattern} dress in {dress.style} style, "
                        f"worn {getattr(dress, 'wear_count', 0)}x."),
        "top": _serialize(dress),
        "bottom": None,
        "dress": _serialize(dress),
        "shoes": _serialize(shoe),
        "jacket": None,
        "purchase_suggestion": None if shoe else f"Consider adding '{priority[0]}' style shoes to complete this look!",
    }


def recommend_outfits(items, occasion, limit=6, offset=0, exclude_ids=None, gender="all", shuffle=False, exclude_pairs=None, skin_tone=None, height=None, weight=None, body_type=None):
    """Ranked, diversified outfit options with pagination (no repeats).

    Returns (page, total). shuffle=True re-rolls within each style tier
    (dress code never shuffles away) for on-demand fresh pairs.
    exclude_pairs is a set of (top_id, bottom_id_or_None) combos already
    shown — items stay reusable in NEW combinations, so flipping through
    never starves early (excluding items instead of pairs capped college
    at ~18 of 36).
    """
    occasion = (occasion or "").lower().strip()
    gender = normalize_gender(gender)
    code = OCCASION_DRESSCODE.get(occasion)
    if not code:
        return [], 0
    priority = code["styles"]
    active_excludes = set(exclude_ids or [])
    seen_pairs = set(exclude_pairs or [])

    tops = [i for i in items if _fine_cat(i) in code["tops"]
            and _gender_ok(getattr(i, "gender", "unisex"), gender)]
    bottoms = [i for i in items if _fine_cat(i) in code["bottoms"]
               and _gender_ok(getattr(i, "gender", "unisex"), gender)]
    dresses = ([i for i in items if _fine_cat(i) == "dress"
                and getattr(i, "style", "") in code.get("dress_styles", [])
                and _gender_ok(getattr(i, "gender", "unisex"), gender)]
               if code.get("allow_dress") else [])

    # Legacy item-skips (e.g. /recommend?exclude=) expand into pair-skips
    # for garments: same visible outcome as before, but combinable items
    # stay reusable instead of starving the pool. Shoes/jackets keep hard
    # item-skips via active_excludes in their own selectors below.
    for t in tops:
        if t.id in active_excludes:
            for b in bottoms:
                seen_pairs.add((t.id, b.id))
    for b in bottoms:
        if b.id in active_excludes:
            for t in tops:
                seen_pairs.add((t.id, b.id))
    for d in dresses:
        if d.id in active_excludes:
            seen_pairs.add((d.id, None))

    # NOTE: item excludes apply to accessories (shoes/jackets) only — tops,
    # bottoms and dresses stay reusable so new COMBINATIONS keep coming.
    # Pair-level exclusion (seen_pairs) is what prevents repeats.
    pairs = _ranked_sets(items, tops, bottoms, priority, set(), occasion, gender)
    ranked_dresses = _ranked_dresses(dresses, priority, set(), gender, occasion=occasion)

    shoes_ranked = [i for i in _ranked_shoes(items, priority, active_excludes, gender)
                    if _gender_ok(getattr(i, "gender", "unisex"), gender)]
    jackets = [i for i in items
               if _fine_cat(i) in JACKET_CATEGORIES
               and _tier(getattr(i, "style", ""), priority) < 999
               and _gender_ok(getattr(i, "gender", "unisex"), gender)
               and i.id not in active_excludes]
    jackets.sort(key=lambda i: (_tier(getattr(i, "style", ""), priority),
                                _item_quality(i, 0, _gender_rank_bonus(i, gender))))
    jacket_pool = list(jackets) if occasion in ("wedding", "office", "interview", "party") else []

    # Merge sets + dresses on one scale, then diversify greedily.
    merged = []
    for worst, score, top, bottom, t_tier, b_tier in pairs:
        merged.append((worst, score, "set", (top, bottom, t_tier, b_tier)))
    for d_tier, d_score, dress in ranked_dresses:
        # Dresses read one tier dressier for wedding/party tie-breaks.
        tie_bias = -0.01 if occasion in ("wedding", "party") else 0.0
        merged.append((d_tier, d_score + tie_bias, "dress", (dress, d_tier)))
    merged.sort(key=lambda m: (m[0], m[1]))
    if shuffle:
        # Tier-respecting shuffle: best-tier outfits stay best-tier, order
        # within each tier is re-rolled so "give another pair" feels fresh
        # without ever breaking dress-code priority.
        tiers = {}
        for m in merged:
            tiers.setdefault(m[0], []).append(m)
        merged = [m for t in sorted(tiers) for m in _random.sample(tiers[t], len(tiers[t]))]

    options = []
    top_use, bot_use, shoe_use, jacket_use = {}, {}, {}, {}
    top_sig_use, bot_sig_use = {}, {}
    shoe_pool = list(shoes_ranked)

    def _pick_jacket(top, bottom):
        """Best harmony-gated layer for THIS pair, rotated across cards.
        Returns None rather than a clashing/doubled layer — a missing layer
        beats a wrong one."""
        best, best_score = None, 4.0  # gate: must score <= 4 to attach
        for cand in jacket_pool:
            if jacket_use.get(cand.id, 0) >= 2:
                continue
            s, _ = _jacket_score(cand, top, bottom, occasion)
            s += jacket_use.get(cand.id, 0) * 2.0  # rotation pressure
            if s < best_score:
                best, best_score = cand, s
        if best is not None:
            jacket_use[best.id] = jacket_use.get(best.id, 0) + 1
        return best

    for worst, score, kind, payload in merged:
        if kind == "set":
            top, bottom, t_tier, b_tier = payload
            if (top.id, bottom.id) in seen_pairs:
                continue
            if top_use.get(top.id, 0) >= 2 or bot_use.get(bottom.id, 0) >= 2:
                continue
            # Visual-variety caps: same-looking top at most twice, same
            # bottom look at most 3x — cards must not read as repeats.
            t_sig, b_sig = _item_sig(top), _item_sig(bottom)
            if top_sig_use.get(t_sig, 0) >= 2 or bot_sig_use.get(b_sig, 0) >= 3:
                continue
            shoe = _pick_rotated_shoe(shoe_pool, shoe_use)
            jacket = _pick_jacket(top, bottom)
            opt = _build_set_option(occasion, priority, top, bottom, t_tier, b_tier, shoe, jacket)
            top_use[top.id] = top_use.get(top.id, 0) + 1
            bot_use[bottom.id] = bot_use.get(bottom.id, 0) + 1
            top_sig_use[t_sig] = top_sig_use.get(t_sig, 0) + 1
            bot_sig_use[b_sig] = bot_sig_use.get(b_sig, 0) + 1
            if shoe is not None:
                shoe_use[shoe.id] = shoe_use.get(shoe.id, 0) + 1
            options.append(opt)
        else:
            dress, d_tier = payload
            if (dress.id, None) in seen_pairs:
                continue
            if top_use.get(("d", dress.id), 0) >= 1:
                continue
            shoe = _pick_rotated_shoe(shoe_pool, shoe_use)
            if shoe is not None:
                shoe_use[shoe.id] = shoe_use.get(shoe.id, 0) + 1
            options.append(_build_dress_option(occasion, priority, dress, d_tier, shoe))
            top_use[("d", dress.id)] = 1

    options = _spread_adjacent(options)

    # No early break: pools are tiny (tens of pairs), so build the full
    # diverse ranking. total = everything available, page = slice of the
    # acceptance-ordered list (pass 1: unique garments first, pass 2: rest).
    # Slicing acceptance order means pages can never overlap, and page 1
    # always leads with distinct garments.
    taken = _accept_in_order(options)
    total = len(taken)

    # Personalization scoring & notes for skin tone & body profile
    if skin_tone or body_type:
        st_clean = (skin_tone or "").strip().lower()
        bt_clean = (body_type or "").strip()
        affinity = None
        for k, v in SKIN_TONE_COLOR_AFFINITY.items():
            if k in st_clean:
                affinity = v
                break

        for opt in taken:
            match_notes = list(opt.get("match_notes", []))
            top_obj = opt.get("top") or opt.get("dress") or {}
            top_color = (top_obj.get("color") or "").lower()
            if affinity and top_color:
                if top_color in affinity["best"]:
                    opt["match_score"] = min(99, opt.get("match_score", 90) + 4)
                    match_notes.insert(0, f"✨ {top_color.capitalize()} matches your {skin_tone} complexion")
                elif top_color in affinity.get("avoid", set()):
                    opt["match_score"] = max(50, opt.get("match_score", 90) - 4)
            if bt_clean:
                match_notes.append(f"✨ Fitted for {bt_clean} body type")
            opt["match_notes"] = match_notes[:4]

    return taken[offset:offset + limit], total


def _accept_in_order(options):
    """Pass 1 takes options whose top AND bottom are both unseen (distinct
    garments first); pass 2 appends the remaining unseen pairs in rank
    order. Spreads repeats apart so small pools still read varied."""
    taken, seen_pairs = [], set()
    used_t, used_b = set(), set()
    rest = []
    for o in options:
        t = (o.get("top") or {}).get("id")
        b = (o.get("bottom") or {}).get("id") if o.get("bottom") else None
        if t not in used_t and b not in used_b:
            taken.append(o)
            seen_pairs.add((t, b))
            used_t.add(t)
            used_b.add(b)
        else:
            rest.append(o)
    for o in rest:
        t = (o.get("top") or {}).get("id")
        b = (o.get("bottom") or {}).get("id") if o.get("bottom") else None
        if (t, b) not in seen_pairs:
            taken.append(o)
            seen_pairs.add((t, b))
    return _spread_adjacent(taken)


def _spread_adjacent(options):
    """Reorder so consecutive cards never show the same top look or the same
    bottom look back-to-back (the 'two shirts in a row' complaint). Ranking
    is preserved as much as possible: always take the best remaining option
    that looks different from the previous card."""
    if len(options) < 3:
        return options

    def _sig(o):
        t = o.get("top") or {}
        b = o.get("bottom") or {}
        return ((t.get("category"), t.get("color"), t.get("pattern")),
                (b.get("category"), b.get("color"), b.get("pattern")) if b else None)

    remaining = list(options)
    ordered = [remaining.pop(0)]
    while remaining:
        last_top, last_bot = _sig(ordered[-1])
        pick = 0
        for idx, cand in enumerate(remaining):
            c_top, c_bot = _sig(cand)
            if c_top != last_top and c_bot != last_bot:
                pick = idx
                break
        ordered.append(remaining.pop(pick))
    return ordered


def recommend_outfit(items, occasion, exclude_ids=None, gender="all"):
    """Backward-compatible single best outfit (chatbot + old clients)."""
    occasion = (occasion or "").lower().strip()
    gender = normalize_gender(gender)
    if occasion not in OCCASION_DRESSCODE:
        return {
            "message": f"'{occasion}' is not a recognized occasion. Please choose from: {', '.join(OCCASION_DRESSCODE)}.",
            "outfit": None,
        }
    page, _total = recommend_outfits(items, occasion, limit=1, offset=0, exclude_ids=exclude_ids, gender=gender)
    if page:
        page[0]["gender"] = gender
        return page[0]
    code = OCCASION_DRESSCODE[occasion]
    priority = code["styles"]
    tops = [i for i in items if _fine_cat(i) in code["tops"]
            and _gender_ok(getattr(i, "gender", "unisex"), gender)]
    bottoms = [i for i in items if _fine_cat(i) in code["bottoms"]
               and _gender_ok(getattr(i, "gender", "unisex"), gender)]
    style_ok = [i for i in tops if _tier(getattr(i, "style", ""), priority) < 999]
    bottom_ok = [i for i in bottoms if _tier(getattr(i, "style", ""), priority) < 999]
    who = f" for {gender}" if gender != "all" else ""

    def _gap_line(cat_pool, label):
        """Exact inventory readout so small pools get actionable advice:
        how many gender-matching pieces exist in the right category at all
        (fixable via Edit) vs nothing to work with (go shopping)."""
        if gender == "all":
            return None
        exact = sum(1 for i in cat_pool
                    if (getattr(i, "gender", "unisex") or "unisex").strip().lower() == gender)
        flex = sum(1 for i in cat_pool
                   if (getattr(i, "gender", "unisex") or "unisex").strip().lower() == "unisex")
        return (f"{label}: 0 {gender} {label}s in {priority[0]} style "
                f"(you own {exact} {gender} + {flex} unisex {label}s — "
                f"correct one's style in Edit or add a {priority[0]} {label})")

    missing = []
    if not style_ok:
        missing.append(_gap_line(tops, "top")
                       if gender != "all"
                       else (f"top ({'/'.join(code['tops'])} in {priority[0]} style — t-shirts are not allowed for {occasion})"
                             if occasion in ("wedding", "office", "interview") else "top"))
    if not bottom_ok:
        missing.append(_gap_line(bottoms, "bottom")
                       if gender != "all"
                       else (f"bottom ({'/'.join(code['bottoms'])} in {priority[0]} style — jeans are not allowed for {occasion})"
                             if occasion in ("wedding", "office", "interview") else "bottom"))
    return {
        "message": (f"Your wardrobe is missing a suitable {' and '.join(missing)} for {occasion}{who}. "
                    f"Add '{priority[0]}' style items to unlock it."),
        "outfit": None,
    }
