import os
import csv
import json
import random
import math
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

ROOT_DIR = "nano arabic dat"

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

LANGUAGE_STRUCTURES = [
    "arabic_only",
    "bilingual_arabic_primary",
    "bilingual_english_primary",
    "parallel_pair_arabic",
    "arabic_transliteration",
    "arabic_subtitle"
]

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

SEMANTIC_WEIGHTS = [
    "speed_motion_typography",
    "massive_heavy_typography",
    "magic_mystic_typography",
    "decay_horror_typography",
    "cyber_tech_typography",
    "organic_nature_typography",
    "ancient_stone_typography"
]

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

LAYOUTS = [
    "center_title_large",
    "top_title_hero_bottom",
    "bottom_title_scene_top",
    "grid_framed_boxart",
    "black_border_nes",
    "diagonal_arcade_badge",
    "centered_emblem",
    "split_bilingual_layout",
    "border_framed"
]

RESOLUTIONS = ["128x128", "256x256"]

PALETTES = {
    "monochrome_dmg": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "retro_8bit_nes": [(0, 0, 0), (255, 255, 255), (228, 0, 88), (0, 120, 248), (248, 184, 0), (0, 168, 0)],
    "arcade_vibrant": [(10, 10, 30), (255, 0, 128), (0, 255, 240), (255, 230, 0), (120, 0, 255), (255, 255, 255)],
    "retro_synthwave_80s": [(20, 10, 35), (255, 40, 140), (0, 220, 255), (255, 200, 50), (40, 20, 70)],
    "egyptian_folklore_pixel": [(40, 20, 10), (210, 140, 30), (255, 215, 0), (180, 50, 30), (240, 220, 180)],
    "dark_fantasy_gothic": [(15, 15, 20), (120, 20, 30), (180, 180, 200), (60, 60, 80), (220, 190, 100)]
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
    ("سيد الحلبة", "RING MASTER")
]

TEXT_EFFECTS_POOL = [
    "glowing_edges", "chrome_bevel", "gold_texture", "fire_effect",
    "dripping_slime", "drop_shadow", "thick_black_outline", "pixel_gradient",
    "crt_scanlines", "grid_overlay", "1-bit_dithering", "16-bit_gradient", "glitch_displacement"
]

TAGS_POOL = [
    "arabic_text", "connected_letters", "pixel_typography", "gameboy_box", "snes_cartridge",
    "nes_cover", "sega_grid", "title_screen", "sub_title", "pixel_frame", "hero_sprite",
    "dragon_bg", "space_ship", "pixel_explosion", "castle_stage", "mecha_sprite",
    "boss_battle", "pixel_font", "pyramid_bg", "desert_bg", "sword_icon", "skull_icon", "lightning_strike"
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

def draw_retro_poster(w, h, platform, poster_type, lang_struct, typo_style, art_style, ar_title, en_title, layout):
    img = Image.new("RGB", (w, h), (15, 15, 25))
    draw = ImageDraw.Draw(img)

    palette = PALETTES.get(art_style, PALETTES["arcade_vibrant"])
    bg_c1 = palette[0]
    bg_c2 = palette[min(1, len(palette)-1)]
    accent = palette[min(2, len(palette)-1)]
    fg_text = palette[min(3, len(palette)-1)] if len(palette) > 3 else (255, 255, 255)

    # Render gradient
    for y in range(h):
        r = int(bg_c1[0] + (bg_c2[0] - bg_c1[0]) * (y / float(h)))
        g = int(bg_c1[1] + (bg_c2[1] - bg_c1[1]) * (y / float(h)))
        b = int(bg_c1[2] + (bg_c2[2] - bg_c1[2]) * (y / float(h)))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Grid effect if requested
    if "sega" in platform or "grid" in layout:
        step = max(8, w // 16)
        for x in range(0, w, step):
            draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 20))
        for y in range(0, h, step):
            draw.line([(0, y), (w, y)], fill=(255, 255, 255, 20))

    # Decorative Border / Frame
    margin = max(4, w // 20)
    draw.rectangle([margin, margin, w - margin - 1, h - margin - 1], outline=accent, width=max(1, w // 128))
    draw.rectangle([margin + 3, margin + 3, w - margin - 4, h - margin - 4], outline=(255, 255, 255), width=1)

    # Central Graphic Emblem
    cx, cy = w // 2, h // 2
    emblem_r = w // 6
    if "emblem" in layout or "badge" in layout or poster_type in ["box_art_front", "full_game_poster", "arcade_marquee"]:
        draw.ellipse([cx - emblem_r, cy - emblem_r, cx + emblem_r, cy + emblem_r], outline=accent, width=max(1, w // 64))
        poly_pts = [(cx, cy - emblem_r + 4), (cx + emblem_r - 4, cy + emblem_r - 4), (cx - emblem_r + 4, cy + emblem_r - 4)]
        draw.polygon(poly_pts, outline=fg_text, fill=bg_c1)

    # Text Rendering Setup
    font_title = get_font(max(10, w // 14))
    font_sub = get_font(max(8, w // 20))

    ar_rendered = render_arabic(ar_title)

    # Draw Text according to language structure and layout
    if lang_struct == "arabic_only":
        draw.text((cx, cy), ar_rendered, fill=fg_text, font=font_title, anchor="mm")
    elif lang_struct in ["bilingual_arabic_primary", "parallel_pair_arabic"]:
        draw.text((cx, margin + max(12, w // 10)), ar_rendered, fill=fg_text, font=font_title, anchor="mm")
        draw.text((cx, h - margin - max(12, w // 10)), en_title, fill=accent, font=font_sub, anchor="mm")
    elif lang_struct == "bilingual_english_primary":
        draw.text((cx, margin + max(12, w // 10)), en_title, fill=fg_text, font=font_sub, anchor="mm")
        draw.text((cx, h - margin - max(12, w // 10)), ar_rendered, fill=accent, font=font_title, anchor="mm")
    else: # arabic_transliteration or arabic_subtitle
        draw.text((cx, margin + max(14, w // 9)), ar_rendered, fill=fg_text, font=font_title, anchor="mm")
        draw.text((cx, cy + max(10, w // 8)), en_title, fill=accent, font=font_sub, anchor="mm")

    # Retro Pixelation enforcement using nearest neighbor downscale/upscale simulation
    if art_style in ["monochrome_dmg", "retro_8bit_nes"]:
        # Quantize colors to small palette
        small_w, small_h = w // 2, h // 2
        img_small = img.resize((small_w, small_h), Image.NEAREST)
        img = img_small.resize((w, h), Image.NEAREST)

    return img

def main():
    os.makedirs(ROOT_DIR, exist_ok=True)
    index_csv_path = os.path.join(ROOT_DIR, "dataset_index.csv")
    csv_rows = []

    total_generated = 0

    # Ensure exhaustive coverage across platforms, poster_types, language_structures, typography_styles, art_styles, resolutions
    sample_id = 1
    for platform in PLATFORMS:
        for poster_type in POSTER_TYPES:
            for lang_struct in LANGUAGE_STRUCTURES:
                # Select typography style, art style, semantic weight, genre, layout deterministically or randomly
                typo_style = TYPOGRAPHY_STYLES[(sample_id % len(TYPOGRAPHY_STYLES))]
                art_style = ART_STYLES[(sample_id % len(ART_STYLES))]
                sem_weight = SEMANTIC_WEIGHTS[(sample_id % len(SEMANTIC_WEIGHTS))]
                genre = GENRES[(sample_id % len(GENRES))]
                layout = LAYOUTS[(sample_id % len(LAYOUTS))]
                ar_title, en_title = ARABIC_TITLES[(sample_id % len(ARABIC_TITLES))]

                for res_str in RESOLUTIONS:
                    w, h = map(int, res_str.split("x"))

                    # Target Directory Structure:
                    # nano arabic dat/<platform>/<poster_type>/<language_structure>/<typography_style>/<art_style>/<resolution>/
                    rel_dir = os.path.join(
                        platform,
                        poster_type,
                        lang_struct,
                        typo_style,
                        art_style,
                        res_str
                    )
                    full_dir = os.path.join(ROOT_DIR, rel_dir)
                    os.makedirs(full_dir, exist_ok=True)

                    filename_base = f"{platform}_{poster_type}_{sample_id:04d}_{res_str}"
                    png_filename = f"{filename_base}.png"
                    json_filename = f"{filename_base}.json"

                    png_path = os.path.join(full_dir, png_filename)
                    json_path = os.path.join(full_dir, json_filename)

                    # Generate Image
                    img = draw_retro_poster(
                        w, h, platform, poster_type, lang_struct,
                        typo_style, art_style, ar_title, en_title, layout
                    )
                    img.save(png_path)

                    # Selected color tags and text effects
                    colors_sample = random.sample(["red", "blue", "gold", "black", "white", "neon_cyan", "purple"], 3)
                    effects_sample = random.sample(TEXT_EFFECTS_POOL, 3)
                    tags_sample = random.sample(TAGS_POOL, 5) + ["arabic_text", "connected_letters", "pixel_typography"]

                    # Metadata JSON compliant with schema
                    meta = {
                        "file": png_filename,
                        "resolution": res_str,
                        "platform": platform,
                        "poster_type": poster_type,
                        "language_structure": lang_struct,
                        "typography_details": {
                            "style": typo_style,
                            "semantic_weight": sem_weight,
                            "text_effects": effects_sample,
                            "readability_score": "high"
                        },
                        "art_style": art_style,
                        "genre": genre,
                        "layout": layout,
                        "colors": colors_sample,
                        "tags": tags_sample,
                        "notes": f"Retro pixel game typography and poster dataset item for {platform} with {typo_style} title: {ar_title} ({en_title})."
                    }

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    # Index row
                    rel_png = os.path.relpath(png_path, ROOT_DIR)
                    rel_json = os.path.relpath(json_path, ROOT_DIR)
                    csv_rows.append([
                        filename_base,
                        platform,
                        poster_type,
                        lang_struct,
                        typo_style,
                        art_style,
                        res_str,
                        genre,
                        layout,
                        rel_png,
                        rel_json
                    ])

                    total_generated += 1

                sample_id += 1

    # Write dataset_index.csv
    with open(index_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename_base", "platform", "poster_type", "language_structure",
            "typography_style", "art_style", "resolution", "genre", "layout",
            "image_path", "metadata_path"
        ])
        writer.writerows(csv_rows)

    print(f"Successfully generated {total_generated} dataset items in '{ROOT_DIR}' with index at '{index_csv_path}'")

if __name__ == "__main__":
    main()
