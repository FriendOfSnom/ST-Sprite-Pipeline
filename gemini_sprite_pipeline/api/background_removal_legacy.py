"""
Legacy background removal implementation.

This module contains the original threshold-based background removal method
that was used before upgrading to ML-based removal (rembg).

Preserved for:
- Fallback when rembg is not available
- Comparison/testing purposes
- Users who prefer the simpler/faster method
"""

from io import BytesIO
from PIL import Image


def strip_background_legacy(image_bytes: bytes) -> bytes:
    """
    Strip a flat-ish magenta background using threshold-based color matching.

    This is the original implementation that uses border pixel sampling
    and Euclidean distance threshold to identify and remove background pixels.

    Strategy:
      1) Load RGBA.
      2) Collect all opaque border pixels.
      3) Estimate background color as the average of those border pixels.
      4) Clear any pixel sufficiently close to that background color.

    NOTE: This method has known limitations:
      - Can leave faint halos around edges
      - May remove character colors similar to background
      - Struggles with isolated background "pockets" in complex shapes

    For better quality, use the ML-based strip_background() method instead.

    Args:
        image_bytes: Input image as bytes

    Returns:
        Image bytes with background removed (transparent)
    """
    BG_CLEAR_THRESH = 56  # Tweak this if too aggressive or too gentle

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        width, height = img.size
        pixels = img.load()

        border_samples = []

        # Sample top and bottom rows
        for x in range(width):
            for y in (0, height - 1):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    continue
                border_samples.append((r, g, b))

        # Sample left and right columns
        for y in range(height):
            for x in (0, width - 1):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    continue
                border_samples.append((r, g, b))

        if not border_samples:
            # Nothing to go on; just return original
            return image_bytes

        # Calculate average background color
        sample_count = float(len(border_samples))
        bg_r = sum(r for r, _, _ in border_samples) / sample_count
        bg_g = sum(g for _, g, _ in border_samples) / sample_count
        bg_b = sum(b for _, _, b in border_samples) / sample_count

        bg_clear_thresh_sq = BG_CLEAR_THRESH * BG_CLEAR_THRESH

        def is_background(r, g, b):
            """Check if pixel color is close to background."""
            dr = r - bg_r
            dg = g - bg_g
            db = b - bg_b
            return (dr * dr + dg * dg + db * db) <= bg_clear_thresh_sq

        # Create output image with background removed
        output = Image.new("RGBA", (width, height))
        output_pixels = output.load()

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    output_pixels[x, y] = (r, g, b, 0)
                elif is_background(r, g, b):
                    output_pixels[x, y] = (r, g, b, 0)
                else:
                    output_pixels[x, y] = (r, g, b, a)

        buffer = BytesIO()
        output.save(buffer, format="PNG")
        return buffer.getvalue()

    except Exception as e:
        print(f"  [WARN] strip_background_legacy failed, returning original bytes: {e}")
        return image_bytes
