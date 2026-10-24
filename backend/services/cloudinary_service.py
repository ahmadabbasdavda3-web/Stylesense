import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy initialization of Cloudinary SDK
_cloudinary_configured = False


def _check_cloudinary_config() -> bool:
    global _cloudinary_configured
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    if cloud_name and api_key and api_secret:
        try:
            import cloudinary
            import cloudinary.uploader
            cloudinary.config(
                cloud_name=cloud_name,
                api_key=api_key,
                api_secret=api_secret,
                secure=True
            )
            _cloudinary_configured = True
            return True
        except Exception as e:
            logger.warning(f"Could not configure Cloudinary SDK: {e}")
            return False
    return False


def upload_image_to_cloud(file_path: str) -> Optional[Dict[str, str]]:
    """Uploads local image file to Cloudinary if configured.
    Returns dict with 'secure_url' and 'thumbnail_url', or None if unconfigured/failed.
    """
    if not _check_cloudinary_config():
        return None

    try:
        import cloudinary.uploader
        response = cloudinary.uploader.upload(
            file_path,
            folder="stylesense_wardrobe",
            transformation=[
                {"quality": "auto", "fetch_format": "auto"}
            ]
        )
        secure_url = response.get("secure_url")
        public_id = response.get("public_id")
        
        # Build thumbnail transformation URL
        thumbnail_url = secure_url
        if public_id:
            thumbnail_url, _ = cloudinary.utils.cloudinary_url(
                public_id,
                width=300,
                height=400,
                crop="fill",
                quality="auto"
            )

        return {
            "secure_url": secure_url,
            "thumbnail_url": thumbnail_url,
            "public_id": public_id
        }
    except Exception as e:
        logger.error(f"Cloudinary upload failed for {file_path}: {e}")
        return None
