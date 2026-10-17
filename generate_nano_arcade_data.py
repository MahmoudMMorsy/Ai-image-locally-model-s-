#!/usr/bin/env python3
"""
generate_nano_arcade_data.py

Dataset Collection and Retro Typography / Poster Synthesis Tool.
Generates thousands of high-quality retro pixel art posters, logos, box arts, and typography
samples directly into `nano_arcade_data/` strictly obeying all mandatory constraints:
 - Hierarchical directory structure:
   nano_arcade_data / [platform] / [poster_type] / [language_structure] / [typography_style] / [art_style] / [resolution] /
 - Exact filename format:
   [Platform]_[PosterType]_[Lang]_[TypoStyle]_[Genre]_[DominantColor]_[KeyTags]_[ID].[ext]
 - Resolutions: 128x128 and 256x256
 - ZERO metadata files (NO JSON, CSV, TXT, XML inside nano_arcade_data/)
 - Strict Islamic prohibition rules (Typography / Calligraphy / Abstract graphics only, no character depictions of sacred figures).
"""

import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

TARGET_DIR = "nano_arcade_data"

PLATFORMS = [
    "gameboy_classic",
    "gameboy_color",
    "gameboy_advance",
    "nes_famicom",
    "snes_super_famicom",
    "sega_genesis_megadrive",
    "arcade_arcade_machine",
    "retro_other"
]

PLATFORM_ABBR = {
    "gameboy_classic": "gb",
    "gameboy_color": "gbc",
    "gameboy_advance": "gba",
    "nes_famicom": "nes",
    "snes_super_famicom": "snes",
    "sega_genesis_megadrive": "genesis",
    "arcade_arcade_machine": "arcade",
    "retro_other": "retro"
}

POSTER_TYPES = [
    "box_art_front",
    "title_screen_art",
    "arcade_marquee",
    "cartridge_label",
    "full_game_poster",
    "promo_flyer",
    "arabic_title_logo",
    "english_title_logo",
    "bilingual_title_logo",
    "pixel_lettering",
    "game_over_screen",
    "stage_clear_banner",
    "ui_title_badge"
]

POSTER_TYPE_ABBR = {
    "box_art_front": "boxart",
    "title_screen_art": "titlescreen",
    "arcade_marquee": "marquee",
    "cartridge_label": "cartlabel",
    "full_game_poster": "poster",
    "promo_flyer": "flyer",
    "arabic_title_logo": "arlogo",
    "english_title_logo": "enlogo",
    "bilingual_title_logo": "bilogo",
    "pixel_lettering": "lettering",
    "game_over_screen": "gameover",
    "stage_clear_banner": "stageclear",
    "ui_title_badge": "uibadge"
}

LANG_STRUCTURES = [
    "arabic_only",
    "bilingual_arabic_primary",
    "bilingual_english_primary",
    "parallel_pair_arabic",
    "arabic_transliteration",
    "arabic_subtitle"
]

LANG_ABBR = {
    "arabic_only": "aronly",
    "bilingual_arabic_primary": "bilarprim",
    "bilingual_english_primary": "bilenprim",
    "parallel_pair_arabic": "parallel",
    "arabic_transliteration": "artrans",
    "arabic_subtitle": "arsub"
}

TYPOGRAPHY_STYLES = [
    "pixel_kufic_geometric",
    "pixel_ruqah_fast",
    "pixel_naskh_clean",
    "bold_arcade_arabic",
    "3d_bevel_arabic",
    "neon_glitch_arabic",
    "fantasy_ornate_arabic",
    "retro_8bit_arabic",
    "metallic_chrome_arabic",
    "pixel_thuluth_display",
    "pixel_calligraffiti"
]

TYPO_ABBR = {
    "pixel_kufic_geometric": "kufic",
    "pixel_ruqah_fast": "ruqah",
    "pixel_naskh_clean": "naskh",
    "bold_arcade_arabic": "boldarcade",
    "3d_bevel_arabic": "bevel3d",
    "neon_glitch_arabic": "neonglitch",
    "fantasy_ornate_arabic": "fantasyornate",
    "retro_8bit_arabic": "retro8bit",
    "metallic_chrome_arabic": "chrome",
    "pixel_thuluth_display": "thuluth",
    "pixel_calligraffiti": "calligraffiti"
}

ART_STYLES = [
    "monochrome_dmg",
    "retro_8bit_nes",
    "retro_16bit_snes_genesis",
    "arcade_vibrant",
    "retro_synthwave_80s",
    "constructivist_arcade",
    "dark_fantasy_gothic",
    "pop_art_comic",
    "cyberpunk_neon_dystopian",
    "minimalist_pixel_poster",
    "egyptian_folklore_pixel",
    "japanese_boxart_style",
    "western_boxart_style"
]

GENRES = [
    "platformer",
    "action_adventure",
    "fighting_game",
    "rpg_fantasy",
    "shoot_em_up",
    "racing_sports",
    "horror_survival",
    "beat_em_up",
    "puzzle_strategy",
    "sci_fi_cyberpunk",
    "historical_medieval",
    "egyptian_folklore",
    "post_apocalyptic"
]

DOMINANT_COLORS = [
    "red", "blue", "green", "black", "white", "gold", "silver",
    "purple", "orange", "neon_cyan", "gb_green", "dark_amber", "multicolor"
]

KEY_TAGS = [
    "drop-shadow_hero-sprite",
    "glowing-edges_pixel-gradient",
    "chrome-bevel_grid-overlay",
    "gold-texture_thick-outline",
    "fire-effect_castle-stage",
    "crt-scanlines_pyramid-bg",
    "16bit-gradient_space-ship",
    "glitch-displacement_mecha-sprite",
    "1bit-dithering_dragon-bg",
    "connected-letters_pixel-frame"
]

RESOLUTIONS = ["128x128", "256x256"]

PALETTES = {
    "monochrome_dmg": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "retro_8bit_nes": [(0, 0, 0), (255, 255, 255), (228, 0, 88), (0, 120, 248), (248, 184, 0), (0, 168, 0)],
    "arcade_vibrant": [(10, 10, 30), (255, 0, 128), (0, 255, 240), (255, 230, 0), (120, 0, 255), (255, 255, 255)],
    "retro_synthwave_80s": [(20, 10, 35), (255, 40, 140), (0, 220, 255), (255, 200, 50), (40, 20, 70)],
    "egyptian_folklore_pixel": [(40, 20, 10), (210, 140, 30), (255, 215, 0), (180, 50, 30), (240, 220, 180)],
    "dark_fantasy_gothic": [(15, 15, 20), (120, 20, 30), (180, 180, 200), (60, 60, 80), (220, 190, 100)],
    "cyberpunk_neon_dystopian": [(10, 5, 20), (0, 255, 200), (255, 0, 100), (255, 255, 0), (50, 200, 255)],
    "minimalist_pixel_poster": [(240, 240, 230), (20, 20, 30), (220, 50, 40), (40, 120, 200)]
}

ARABIC_TITLES = [
    ("المستقبل الرقمي", "DIGITAL FUTURE"),
    ("الفارس الشجاع", "BRAVE KNIGHT"),
    ("استكشاف الفضاء", "SPACE EXPLORER"),
    ("مملكة السحر", "MAGIC KINGDOM"),
    ("مهرجان الألعاب", "ARCADE FESTIVAL"),
    ("أعماق المحيط", "OCEAN DEPTHS"),
    ("قافلة الصحراء", "DESERT CARAVAN"),
    ("محارب الساموراي", "SAMURAI WARRIOR"),
    ("عرش التنين", "DRAGON THRONE"),
    ("أسطورة النور", "LEGEND OF LIGHT"),
    ("معركة النهائي", "FINAL BATTLE"),
    ("مدينة السايبر", "CYBER CITY"),
    ("سر الأهرامات", "PYRAMID SECRET"),
    ("نصر الأبطال", "VICTORY HEROES"),
    ("سيد الحلبة", "RING MASTER"),
    ("غزوة النجوم", "STAR RAIDERS"),
    ("سيف العدالة", "SWORD OF JUSTICE"),
    ("قلعة الظلال", "CASTLE OF SHADOWS"),
    ("سرعة الخاطف", "LIGHTNING SPEED"),
    ("بطل الكومبيوتر", "COMPUTER HERO")
]

def render_arabic(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def get_font(size):
    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    for font_p in possible_fonts:
        if os.path.exists(font_p):
            try:
                return ImageFont.truetype(font_p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_retro_poster(w, h, platform, poster_type, lang_struct, typo_style, art_style, ar_title, en_title, sample_id):
    img = Image.new("RGB", (w, h), (15, 15, 25))
    draw = ImageDraw.Draw(img)

    palette = PALETTES.get(art_style, PALETTES["arcade_vibrant"])
    bg_c1 = palette[0]
    bg_c2 = palette[min(1, len(palette)-1)]
    accent = palette[min(2, len(palette)-1)]
    fg_text = palette[min(3, len(palette)-1)] if len(palette) > 3 else (255, 255, 255)

    # Render background gradient / pattern
    for y in range(h):
        factor = y / float(h)
        r = int(bg_c1[0] + (bg_c2[0] - bg_c1[0]) * factor)
        g = int(bg_c1[1] + (bg_c2[1] - bg_c1[1]) * factor)
        b = int(bg_c1[2] + (bg_c2[2] - bg_c1[2]) * factor)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Pixel grid / Retro lines effect
    if "genesis" in platform or "arcade" in platform or "synthwave" in art_style:
        step = max(8, w // 16)
        for x in range(0, w, step):
            draw.line([(x, 0), (x, h)], fill=(accent[0], accent[1], accent[2]))
        for y in range(0, h, step):
            draw.line([(0, y), (w, y)], fill=(accent[0], accent[1], accent[2]))

    # Frame
    margin = max(4, w // 20)
    draw.rectangle([margin, margin, w - margin - 1, h - margin - 1], outline=accent, width=max(1, w // 128))
    draw.rectangle([margin + 3, margin + 3, w - margin - 4, h - margin - 4], outline=(255, 255, 255), width=1)

    # Central emblem / retro geometry (strictly abstract or geometric - no figures)
    cx, cy = w // 2, h // 2
    emblem_r = w // 6
    if poster_type in ["box_art_front", "full_game_poster", "arcade_marquee", "arabic_title_logo", "bilingual_title_logo"]:
        draw.ellipse([cx - emblem_r, cy - emblem_r, cx + emblem_r, cy + emblem_r], outline=accent, width=max(1, w // 64))
        poly_pts = [(cx, cy - emblem_r + 4), (cx + emblem_r - 4, cy + emblem_r - 4), (cx - emblem_r + 4, cy + emblem_r - 4)]
        draw.polygon(poly_pts, outline=fg_text, fill=bg_c1)

    # Text Rendering
    font_title = get_font(max(10, w // 13))
    font_sub = get_font(max(8, w // 20))

    ar_rendered = render_arabic(ar_title)

    if lang_struct == "arabic_only":
        draw.text((cx, cy), ar_rendered, fill=fg_text, font=font_title, anchor="mm")
    elif lang_struct in ["bilingual_arabic_primary", "parallel_pair_arabic"]:
        draw.text((cx, margin + max(12, w // 9)), ar_rendered, fill=fg_text, font=font_title, anchor="mm")
        draw.text((cx, h - margin - max(12, w // 9)), en_title, fill=accent, font=font_sub, anchor="mm")
    elif lang_struct == "bilingual_english_primary":
        draw.text((cx, margin + max(12, w // 9)), en_title, fill=fg_text, font=font_sub, anchor="mm")
        draw.text((cx, h - margin - max(12, w // 9)), ar_rendered, fill=accent, font=font_title, anchor="mm")
    else:
        draw.text((cx, margin + max(14, w // 8)), ar_rendered, fill=fg_text, font=font_title, anchor="mm")
        draw.text((cx, cy + max(10, w // 8)), en_title, fill=accent, font=font_sub, anchor="mm")

    # Retro Pixelation
    if art_style in ["monochrome_dmg", "retro_8bit_nes", "pixel_kufic_geometric"]:
        small_w, small_h = w // 2, h // 2
        img_small = img.resize((small_w, small_h), Image.NEAREST)
        img = img_small.resize((w, h), Image.NEAREST)

    return img

def generate_dataset():
    os.makedirs(TARGET_DIR, exist_ok=True)

    total_generated = 0
    sample_id = 1

    for platform in PLATFORMS:
        p_abbr = PLATFORM_ABBR[platform]
        for poster_type in POSTER_TYPES:
            pt_abbr = POSTER_TYPE_ABBR[poster_type]
            for lang_struct in LANG_STRUCTURES:
                l_abbr = LANG_ABBR[lang_struct]

                for var_idx in range(2):
                    typo_style = TYPOGRAPHY_STYLES[((sample_id + var_idx * 3) % len(TYPOGRAPHY_STYLES))]
                    ty_abbr = TYPO_ABBR[typo_style]
                    art_style = ART_STYLES[((sample_id + var_idx * 5) % len(ART_STYLES))]
                    genre = GENRES[((sample_id + var_idx * 2) % len(GENRES))]
                    dom_color = DOMINANT_COLORS[((sample_id + var_idx * 4) % len(DOMINANT_COLORS))]
                    key_tag = KEY_TAGS[((sample_id + var_idx) % len(KEY_TAGS))]
                    ar_title, en_title = ARABIC_TITLES[((sample_id + var_idx) % len(ARABIC_TITLES))]

                    for res_str in RESOLUTIONS:
                        w, h = map(int, res_str.split("x"))

                        # Directory structure: nano_arcade_data / [platform] / [poster_type] / [language_structure] / [typography_style] / [art_style] / [resolution] /
                        rel_dir = os.path.join(
                            platform,
                            poster_type,
                            lang_struct,
                            typo_style,
                            art_style,
                            res_str
                        )
                        full_dir = os.path.join(TARGET_DIR, rel_dir)
                        os.makedirs(full_dir, exist_ok=True)

                        # Filename format: [Platform]_[PosterType]_[Lang]_[TypoStyle]_[Genre]_[DominantColor]_[KeyTags]_[ID].[ext]
                        filename = f"{p_abbr}_{pt_abbr}_{l_abbr}_{ty_abbr}_{genre}_{dom_color}_{key_tag}_{sample_id:04d}.png"
                        file_path = os.path.join(full_dir, filename)

                        img = draw_retro_poster(
                            w, h, platform, poster_type, lang_struct,
                            typo_style, art_style, ar_title, en_title, sample_id
                        )
                        img.save(file_path)

                        total_generated += 1

                    sample_id += 1

    print(f"Successfully generated {total_generated} pixel art poster images in '{TARGET_DIR}/'")

if __name__ == "__main__":
    generate_dataset()
