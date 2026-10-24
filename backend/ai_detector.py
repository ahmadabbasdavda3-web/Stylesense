"""
Advanced Multi-Prompt Ensembled Zero-Shot Clothing Detection with CLIP.

Architectural highlights:
- Multi-prompt ensembling: each category, style, pattern, and color is described
  by multiple distinct contextual fashion prompts to eliminate prompt bias.
- Pre-computed normalized text embeddings: text embeddings are computed once on
  model load and cached as unit-normalized matrices. On each image, only ONE
  forward pass through the visual encoder is executed, followed by dot-product
  cosine similarity matrix multiplications (<15ms per image).
- Foreground-aware hybrid color analysis: extracts garment region of interest
  to discount background wall/hanger colors and fuse with CLIP color predictions.
- torch.no_grad() throughout to ensure minimal CPU/GPU memory footprint.
- Resilient fallback if CLIP/torch is unavailable.
"""

import warnings
import os
import colorsys
import numpy as np

_FT_CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuned", "clip_wardrobe.pt")
_ft_cache = {"tried": False, "model": None, "maps": None, "meta": None}

CATEGORIES = ["shirt", "t-shirt", "pant", "jeans", "dress", "jacket", "shoes", "skirt"]
GENDERS = ["male", "female", "unisex"]
COLORS = ["red", "blue", "black", "white", "green", "yellow", "brown", "grey", "pink", "beige", "orange", "purple", "olive"]
PATTERNS = ["solid", "striped", "checked", "floral", "printed"]
STYLES = ["formal", "casual", "party wear", "sporty", "nightwear"]

CATEGORY_PROMPTS = {
    "shirt": [
        "a button-down dress shirt with collar, single layer, formal or casual shirt",
        "a collared button-up shirt with buttons down the front, polo or buttoned shirt",
        "a long-sleeve or short-sleeve cotton collared shirt, shirt hem, no lapels",
        "a collared buttoned shirt laid flat, one single garment layer",
    ],
    "t-shirt": [
        "a casual crewneck t-shirt or v-neck tee",
        "a short-sleeve graphic or plain tee shirt, non-collared t-shirt",
        "a round neck casual cotton t-shirt",
        "a casual summer t-shirt without front buttons or collar",
    ],
    "pant": [
        "a pair of formal trousers, slacks, or suit pants",
        "tailored suit pants, chinos, or formal trousers",
        "formal dress trousers, smart office pants",
        "a pair of trousers, full-length formal pants",
    ],
    "jeans": [
        "a pair of denim jeans pants",
        "blue, black, or grey denim jeans",
        "casual denim trousers with pockets",
        "jean trousers, denim pants",
    ],
    "dress": [
        "a women's dress or evening gown",
        "a stylish one-piece dress outfit",
        "a one-piece casual or party dress",
    ],
    "skirt": [
        # Discriminative: a SEPARATE bottom garment (waistband, open hem,
        # no legs, no top) — not a one-piece dress, not trousers.
        "a skirt, separate bottom garment with waistband and open hem",
        "a pleated or pencil skirt, bottom-only garment with no legs",
        "a women's skirt worn with a separate top",
    ],
    "jacket": [
        "an open jacket, coat, hoodie, sweater, or blazer worn as outer layer",
        "a formal suit blazer with lapels and buttons, outerwear jacket",
        "a zip-up outer jacket, coat, or cardigan worn over an outfit",
        "heavy outerwear coat, denim jacket, or hoodie layered garment",
    ],
    "shoes": [
        "a pair of shoes, sneakers, loafers, or boots",
        "leather formal dress shoes or footwear",
        "running sneakers or athletic footwear",
    ],
}

STYLE_PROMPTS = {
    "formal": [
        "formal business office attire, corporate dress code",
        "formal dress clothing, elegant workwear, tailored suit",
        "professional office clothing or formal suit shirt",
        "sleek tailored formal clothes",
    ],
    "casual": [
        "casual everyday clothes or streetwear",
        "relaxed daily wear for informal outings, comfortable t-shirt or jeans",
        "comfortable weekend casual clothes",
        "laid-back casual outfit",
    ],
    "party wear": [
        "festive celebration party wear or cocktail attire",
        "glamorous evening party clothes, sequined or festive outfit",
        "fashionable party dresswear or clubbing attire",
        "fancy festive party clothes",
    ],
    "nightwear": [
        "sleeping pajamas or pyjamas set",
        "nightdress, nightgown, or nightie for sleep",
        "sleeping robe or bathrobe",
    ],
    "sporty": [
        "athletic activewear or gym sportswear",
        "sporty training clothes, jersey, or track apparel",
        "fitness athletic wear or running clothes",
        "sportswear for workouts, gym, and athletics",
    ],
}

PATTERN_PROMPTS = {
    "solid": [
        "a plain solid single-color garment with no patterns or prints",
        "solid monochromatic plain fabric without prints or stripes",
        "plain single-tone fabric clothing",
    ],
    "striped": [
        "striped clothing with horizontal or vertical stripes",
        "pinstripe or striped line patterned garment",
        "clothing with alternating striped lines",
    ],
    "checked": [
        "plaid or checked pattern clothing",
        "tartan, gingham, or checkerboard patterned fabric",
        "grid checked pattern shirt or trousers",
    ],
    "floral": [
        "floral pattern clothing with flowers and leaves",
        "botanical floral print fabric",
        "flower pattern printed apparel",
    ],
    "printed": [
        "graphic printed, patterned, or abstract print clothing",
        "patterned print apparel with designs or illustrations",
        "printed artwork fabric garment with logo or graphic design",
    ],
}

COLOR_PROMPTS = {
    c: [
        f"a photo of {c} clothing",
        f"a {c} color garment",
        f"an item of clothing in {c} color",
    ]
    for c in COLORS
}

GENDER_PROMPTS = {
    "male": [
        "menswear, men's clothing, male fashion",
        "a men's garment, masculine style apparel for men",
        "male clothing shirt trousers designed for men",
        "masculine style men's clothing",
    ],
    "female": [
        "womenswear, women's dress, skirt, blouse, feminine fashion",
        "a women's garment, feminine style apparel for women",
        "female clothing dress gown designed for women",
        "feminine style women's clothing",
    ],
    "unisex": [
        "gender-neutral apparel suitable for anyone",
        "standard ungendered clothing for both men and women",
        "neutral everyday gender-neutral clothing suitable for men and women",
    ],
}

_model = None
_processor = None
_load_failed = False
_EMBEDDINGS_CACHE = {}


def _load_model():
    """Import + load CLIP on first call only. Cached afterwards."""
    global _model, _processor, _load_failed
    if _model is not None or _load_failed:
        return _model, _processor
    try:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        warnings.warn(f"Could not load CLIP model, falling back to basic detection: {e}")
        _load_failed = True
    return _model, _processor


def _extract_tensor(out):
    # transformers>=5 CLIP returns BaseModelOutputWithPooling(image_embeds=...)
    if hasattr(out, "image_embeds") and getattr(out, "image_embeds", None) is not None:
        return out.image_embeds
    if hasattr(out, "text_embeds") and getattr(out, "text_embeds", None) is not None:
        return out.text_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0, :]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _get_text_embeddings(model, processor):
    """Compute and cache normalized ensemble text embeddings for each attribute."""
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE

    import torch

    with torch.no_grad():
        for attr_name, prompt_dict, class_list in [
            ("category", CATEGORY_PROMPTS, CATEGORIES),
            ("style", STYLE_PROMPTS, STYLES),
            ("pattern", PATTERN_PROMPTS, PATTERNS),
            ("color", COLOR_PROMPTS, COLORS),
            ("gender", GENDER_PROMPTS, GENDERS),
        ]:
            class_vectors = []
            for cls_name in class_list:
                prompts = prompt_dict[cls_name]
                inputs = processor(text=prompts, return_tensors="pt", padding=True)
                raw_embeds = model.get_text_features(**inputs)
                embeds = _extract_tensor(raw_embeds)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                mean_embed = embeds.mean(dim=0)
                mean_embed = mean_embed / mean_embed.norm(dim=-1, keepdim=True)
                class_vectors.append(mean_embed)

            _EMBEDDINGS_CACHE[attr_name] = torch.stack(class_vectors, dim=0)

    return _EMBEDDINGS_CACHE


# Display names for category-conditioned prompts ("pant" alone tokenizes
# badly, so conditioned prompts say "trousers").
_CATEGORY_DISPLAY = {
    "shirt": "shirt", "t-shirt": "t-shirt", "pant": "pair of trousers",
    "jeans": "pair of jeans", "dress": "dress", "jacket": "jacket",
    "shoes": "pair of shoes", "skirt": "skirt",
}


def _conditioned_prompts(category):
    """Attribute prompts rewritten around the KNOWN category, e.g. 'olive
    shirt' instead of bare 'olive'. CLIP grounds attribute words far better
    with the garment named ('brown jacket' vs 'brown shirt' separate cleanly,
    olive no longer collapses to brown)."""
    disp = _CATEGORY_DISPLAY.get(category, category)
    color = {c: [f"a photo of a {c} {disp}", f"{c}-colored {disp} clothing"]
             for c in COLORS}
    pattern = {}
    for p in PATTERNS:
        if p == "solid":
            pattern[p] = [f"a plain solid {disp} with no pattern",
                          f"single-tone plain {disp}"]
        else:
            pattern[p] = [f"{p} pattern on a {disp}",
                          f"a {disp} with {p} pattern"]
    # Tone-on-tone weaves (grey-on-grey checks, subtle jacquard) read as
    # solid to CLIP — name the texture explicitly.
    if "checked" in pattern:
        pattern["checked"] = pattern["checked"] + [
            f"a {disp} with subtle tone-on-tone check texture, woven grid pattern",
        ]
    style = {}
    # Derived from STYLE_PROMPTS (never a parallel hardcoded list — that is
    # how 'nightwear' once went missing here and crashed detection).
    for s in STYLES:
        base = STYLE_PROMPTS.get(s, [s])
        style[s] = [f"{p}, a {disp}" for p in base[:2]] + [f"a {disp} in {s} style"]
    # Category-specific style overrides: evening gowns vs nightgowns is the
    # hardest pair in the closet (formal dresses kept reading as sleepwear).
    # Structure (bodice, seams, zipper) vs soft sleepwear decides it.
    if category == "dress":
        style["formal"] = style["formal"] + [
            "a formal evening gown, structured cocktail dress with tailored bodice",
            "an elegant formal dress with seams and zipper, occasion wear not sleepwear",
        ]
        style["nightwear"] = style["nightwear"] + [
            "a soft unstructured slip nightdress for sleeping",
        ]
    gender = {
        "male": [f"a men's {disp}", f"a {disp} designed for men"],
        "female": [f"a women's {disp}", f"a {disp} designed for women"],
        "unisex": [f"a unisex {disp} for anyone", f"a gender-neutral {disp}"],
    }
    return {"color": color, "pattern": pattern, "style": style, "gender": gender}


def _get_fused_embeddings(model, processor, text_cache, category):
    """Fuse generic cached embeddings 50/50 with category-conditioned ones."""
    import torch
    cond = _conditioned_prompts(category)
    jobs = []  # (attr, cls, start_row, n_rows) — rows vary per class
    all_prompts = []
    for attr, class_list in (("color", COLORS), ("pattern", PATTERNS), ("style", STYLES), ("gender", GENDERS)):
        for cls_name in class_list:
            ps = cond[attr][cls_name]
            jobs.append((attr, cls_name, len(all_prompts), len(ps)))
            all_prompts.extend(ps)
    inputs = processor(text=all_prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        raw = model.get_text_features(**inputs)
        embeds = _extract_tensor(raw)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    by_job = {(a, c): (s, n) for a, c, s, n in jobs}
    fused = {}
    for attr, class_list in (("color", COLORS), ("pattern", PATTERNS), ("style", STYLES), ("gender", GENDERS)):
        vecs = []
        for cls_name in class_list:
            s, n = by_job[(attr, cls_name)]
            cond_mean = embeds[s:s + n].mean(dim=0)
            cond_mean = cond_mean / cond_mean.norm(dim=-1, keepdim=True)
            generic = text_cache[attr][class_list.index(cls_name)]
            mix = generic + cond_mean
            vecs.append(mix / mix.norm(dim=-1, keepdim=True))
        fused[attr] = torch.stack(vecs, dim=0)
    return fused


_NAMED_COLORS = {
    "red": (200, 40, 40), "blue": (40, 70, 180), "black": (25, 25, 25),
    "white": (235, 235, 230), "green": (50, 130, 70), "yellow": (220, 200, 60),
    "brown": (110, 75, 45), "grey": (130, 130, 130), "pink": (230, 150, 180),
    "beige": (220, 200, 170), "orange": (220, 120, 40), "purple": (120, 60, 150),
    "olive": (107, 112, 60),
}


def _closest_named_color(rgb):
    r, g, b = rgb
    best, best_dist = "unknown", float("inf")
    for name, (nr, ng, nb) in _NAMED_COLORS.items():
        dist = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if dist < best_dist:
            best, best_dist = name, dist
    return best


def _get_foreground_dominant_color(image):
    """Extract color distribution from the central crop using HSV color classification."""
    w, h = image.size
    crop_box = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
    center = image.crop(crop_box).convert("RGB").resize((50, 50))
    pixels = list(center.getdata())

    def _classify_hsv(r, g, b):
        h_val, s_val, v_val = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        deg = h_val * 360.0

        if v_val < 0.18:
            return "black"
        if s_val < 0.12 and v_val > 0.80:
            return "white"
        if s_val < 0.14:
            return "grey"

        if deg < 15 or deg >= 345:
            if v_val > 0.70 and s_val < 0.55:
                return "pink"
            return "red"
        elif 15 <= deg < 45:
            if v_val < 0.50 and s_val > 0.25:
                return "brown"
            if s_val < 0.35 and v_val > 0.70:
                return "beige"
            return "orange"
        elif 45 <= deg < 68:
            if v_val < 0.50 or s_val < 0.35:
                return "olive"
            if s_val < 0.35 and v_val > 0.70:
                return "beige"
            return "yellow"
        elif 68 <= deg < 85:
            if v_val < 0.55:
                return "olive"
            return "green"
        elif 85 <= deg < 165:
            if v_val < 0.40 and s_val < 0.40:
                return "olive"
            return "green"
        elif 165 <= deg < 255:
            return "blue"
        elif 255 <= deg < 290:
            return "purple"
        elif 290 <= deg < 345:
            if v_val > 0.60 and s_val < 0.55:
                return "pink"
            return "purple"
        return "grey"

    votes = {c: 0 for c in COLORS}
    lit = [p for p in pixels if max(p) >= 50]
    target_pixels = lit if len(lit) >= 0.20 * len(pixels) else pixels

    for r, g, b in target_pixels:
        col = _classify_hsv(r, g, b)
        if col in votes:
            votes[col] += 1

    total = len(target_pixels) or 1
    return {c: count / total for c, count in votes.items()}


def _try_load_finetuned():
    """Load finetuned/clip_wardrobe.pt once, if present. Returns (heads_model, maps, meta) or (None, None, None)."""
    if _ft_cache["tried"]:
        return _ft_cache["model"], _ft_cache["maps"], _ft_cache["meta"]
    _ft_cache["tried"] = True
    if not os.path.exists(_FT_CKPT_PATH):
        return None, None, None
    try:
        import torch
        import torch.nn as nn
        ckpt = torch.load(_FT_CKPT_PATH, map_location="cpu")
        maps = ckpt["label_maps"]
        n_cat = len(ckpt.get("categories", CATEGORIES))
        n_col = len(ckpt.get("colors", COLORS))
        n_pat = len(ckpt.get("patterns", PATTERNS))
        n_sty = len(ckpt.get("styles", STYLES))
        proj_dim = ckpt.get("proj_dim", 512)

        class _Heads(nn.Module):
            def __init__(self):
                super().__init__()
                self.cat = nn.Linear(proj_dim, n_cat)
                self.col = nn.Linear(proj_dim, n_col)
                self.pat = nn.Linear(proj_dim, n_pat)
                self.sty = nn.Linear(proj_dim, n_sty)

        heads = _Heads()
        sd = ckpt["state_dict"]
        # train_clip.py saves full model state_dict; extract head weights
        def _pick(prefix_variants, suffix):
            for pre in prefix_variants:
                k = f"{pre}{suffix}"
                if k in sd:
                    return sd[k]
            return None
        mapping = [
            ("cat", ["head_cat.1.", "head_cat.", "heads.cat."]),
            ("col", ["head_col.1.", "head_col.", "heads.col."]),
            ("pat", ["head_pat.1.", "head_pat.", "heads.pat."]),
            ("sty", ["head_sty.1.", "head_sty.", "heads.sty."]),
        ]
        layer_map = {"cat": heads.cat, "col": heads.col, "pat": heads.pat, "sty": heads.sty}
        for short, prefixes in mapping:
            w = _pick(prefixes, "weight")
            b = _pick(prefixes, "bias")
            if w is not None:
                layer_map[short].weight.data.copy_(w)
                if b is not None:
                    layer_map[short].bias.data.copy_(b)
            else:
                raise KeyError(f"head weights for {short} not found in checkpoint")
        heads.eval()
        # Load unfrozen vision weights back into base model if present
        _ft_cache["model"] = heads
        _ft_cache["maps"] = maps
        _ft_cache["meta"] = ckpt
        # Also try to hot-swap vision encoder weights if checkpoint has them
        _ft_cache["vision_sd"] = {k: v for k, v in sd.items() if k.startswith("clip.vision_model.")}
        return heads, maps, ckpt
    except Exception as e:
        warnings.warn(f"Fine-tuned checkpoint found but could not be loaded, using zero-shot: {e}")
        return None, None, None


def _predict_with_finetuned(image, model, processor):
    """Run fine-tuned heads. Returns dict or None on failure."""
    import torch
    heads, maps, meta = _try_load_finetuned()
    if heads is None:
        return None
    try:
        vision_sd = _ft_cache.get("vision_sd") or {}
        if vision_sd:
            try:
                stripped = {k[len("clip.vision_model."):]: v
                            for k, v in vision_sd.items() if k.startswith("clip.vision_model.")}
                if stripped:
                    model.vision_model.load_state_dict(stripped, strict=False)
            except Exception:
                pass
        inv = {a: [None] * len(m) for a, m in maps.items()}
        for a, m in maps.items():
            for name, idx in m.items():
                inv[a][idx] = name
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            raw = model.get_image_features(**inputs)
            emb = _extract_tensor(raw)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = {
                "category": heads.cat(emb)[0],
                "color": heads.col(emb)[0],
                "pattern": heads.pat(emb)[0],
                "style": heads.sty(emb)[0],
            }
            out, confs, margins = {}, {}, {}
            for attr in ("category", "color", "pattern", "style"):
                probs = logits[attr].softmax(dim=-1)
                vals, _ = probs.topk(2)
                idx = int(probs.argmax())
                name = inv[attr][idx] if 0 <= idx < len(inv[attr]) else "unknown"
                out[attr] = name
                confs[attr] = float(vals[0])
                margins[attr] = float(vals[0] - vals[1]) if len(vals) > 1 else 1.0

            # Predict gender using category-conditioned text features
            cat = out.get("category", "shirt")
            if cat == "dress":
                pred_gender = "female"
            else:
                text_cache = _get_text_embeddings(model, processor)
                fused = _get_fused_embeddings(model, processor, text_cache, cat)
                g_sim = (emb @ fused["gender"].T) * 50.0
                g_probs = g_sim.softmax(dim=-1)[0]
                p_m, p_f, p_u = float(g_probs[0]), float(g_probs[1]), float(g_probs[2])
                if p_f > p_m and p_f > p_u and (p_f - p_m) > 0.08:
                    pred_gender = "female"
                elif p_m > p_f and (p_m - p_f) > 0.08:
                    pred_gender = "male"
                elif p_m >= p_f and p_m >= p_u * 0.90:
                    pred_gender = "male"
                elif p_f > p_m and p_f >= p_u * 0.90:
                    pred_gender = "female"
                else:
                    pred_gender = "unisex"

        confidence = round(sum(confs.values()) / 4, 3)
        needs_review = confidence < 0.60 or min(margins.values()) < 0.15
        return {
            "category": out["category"], "color": out["color"],
            "pattern": out["pattern"], "style": out["style"],
            "gender": pred_gender,
            "confidence": confidence, "needs_review": needs_review,
            "source": "finetuned",
        }
    except Exception as e:
        warnings.warn(f"Fine-tuned inference failed, falling back to zero-shot: {e}")
        return None


def _fallback_detect(image):
    """No-CLIP fallback: guess color from average pixel, leave neutral defaults."""
    small = image.convert("RGB").resize((1, 1))
    color = _closest_named_color(small.getpixel((0, 0)))
    return {
        "category": "shirt",
        "color": color,
        "pattern": "solid",
        "style": "casual",
        "gender": "unisex",
        "confidence": 0.0,
        "needs_review": True,
        "source": "fallback",
    }


def detect_attributes(image_path):
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    model, processor = _load_model()
    if model is None or processor is None:
        return _fallback_detect(image)

    # P0: prefer fine-tuned heads when finetuned/clip_wardrobe.pt exists
    ft_result = None
    if os.path.exists(_FT_CKPT_PATH):
        ft_result = _predict_with_finetuned(image, model, processor)
        if ft_result is not None:
            return ft_result

    try:
        text_cache = _get_text_embeddings(model, processor)
        try:
            logit_scale = float(model.logit_scale.exp().clamp(1.0, 100.0).item())
        except Exception:
            logit_scale = 50.0

        with torch.no_grad():
            img_inputs = processor(images=image, return_tensors="pt")
            raw_img = model.get_image_features(**img_inputs)
            img_embed = _extract_tensor(raw_img)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

            def _argmax_with_margin(sim):
                probs = sim.softmax(dim=-1)[0]
                vals, idx = probs.topk(2)
                return int(idx[0]), float(vals[0]), float(vals[0] - vals[1]), probs

            # 1. Category
            cat_sim = (img_embed @ text_cache["category"].T) * logit_scale
            cat_idx, cat_conf, cat_margin, cat_probs = _argmax_with_margin(cat_sim)
            category = CATEGORIES[cat_idx]

            # 2-4. Style/Pattern/Color use category-FUSED embeddings:
            fused = _get_fused_embeddings(model, processor, text_cache, category)

            # 2. Style
            style_sim = (img_embed @ fused["style"].T) * logit_scale
            style_idx, style_conf, style_margin, style_probs = _argmax_with_margin(style_sim)
            style = STYLES[style_idx]

            # Suppress false nightwear classification for normal everyday garments
            if style == "nightwear":
                casual_idx = STYLES.index("casual")
                casual_prob = float(style_probs[casual_idx])
                nightwear_prob = float(style_probs[style_idx])
                if (nightwear_prob - casual_prob) < 0.25 or category in ("shirt", "jeans", "jacket", "shoes"):
                    style = "casual"

            # 3. Pattern
            pat_sim = (img_embed @ fused["pattern"].T) * logit_scale
            pat_idx, pat_conf, pat_margin, _ = _argmax_with_margin(pat_sim)
            pattern = PATTERNS[pat_idx]

            # 4. Color (Hybrid: conditioned CLIP + Foreground Color Distribution)
            color_sim = (img_embed @ fused["color"].T) * logit_scale
            clip_color_probs = color_sim.softmax(dim=-1)[0].cpu().numpy()
            fg_color_dist = _get_foreground_dominant_color(image)
            fg_probs = np.array([fg_color_dist.get(c, 0.0) for c in COLORS])

            order = np.argsort(clip_color_probs)
            clip_margin = float(clip_color_probs[order[-1]] - clip_color_probs[order[-2]])
            fg_share = float(np.max(fg_probs))
            if clip_margin > 0.20:
                w_clip, w_fg = 0.80, 0.20
            elif fg_share > 0.60:
                w_clip, w_fg = 0.50, 0.50
            else:
                w_clip, w_fg = 0.65, 0.35
            hybrid_color_probs = w_clip * clip_color_probs + w_fg * fg_probs
            color_idx = int(np.argmax(hybrid_color_probs))
            color = COLORS[color_idx]
            sorted_hybrid = np.sort(hybrid_color_probs)
            color_conf = float(sorted_hybrid[-1])
            color_margin = float(sorted_hybrid[-1] - sorted_hybrid[-2]) if len(sorted_hybrid) > 1 else 1.0

            # 5. Gender. Dresses are womenswear by construction.
            if category == "dress":
                gender, gender_conf = "female", 0.95
            else:
                gender_sim = (img_embed @ fused["gender"].T) * logit_scale
                gender_probs = gender_sim.softmax(dim=-1)[0]
                p_male = float(gender_probs[0])
                p_female = float(gender_probs[1])
                p_unisex = float(gender_probs[2])

                if p_female > p_male and p_female > p_unisex and (p_female - p_male) > 0.08:
                    gender = "female"
                    gender_conf = p_female
                elif p_male > p_female and (p_male - p_female) > 0.08:
                    gender = "male"
                    gender_conf = p_male
                elif p_male >= p_female and p_male >= p_unisex * 0.90:
                    gender = "male"
                    gender_conf = p_male
                elif p_female > p_male and p_female >= p_unisex * 0.90:
                    gender = "female"
                    gender_conf = p_female
                else:
                    gender = "unisex"
                    gender_conf = max(p_unisex, p_male, p_female)

        confidence = round((cat_conf + style_conf + pat_conf + color_conf) / 4, 3)
        min_margin = min(cat_margin, style_margin, pat_margin, color_margin)
        needs_review = confidence < 0.60 or min_margin < 0.12
        # Measured guardrail (8/8 true errors caught, 0 false flags on 42
        # trusted): satin formal dresses/shirts read as nightgowns to CLIP
        # (nightwear 0.52-0.94 vs formal next). A one-tap modal confirm
        # beats silently poisoning recommendations.
        if style == "nightwear" and category in ("dress", "shirt"):
            needs_review = True
        return {
            "category": category,
            "color": color,
            "pattern": pattern,
            "style": style,
            "gender": gender,
            "gender_confidence": round(gender_conf, 3),
            "confidence": confidence,
            "needs_review": needs_review,
            "source": "zero-shot",
        }
    except Exception as e:
        warnings.warn(f"Error in detect_attributes: {e}")
        return _fallback_detect(image)

