import os
from poster_generator_256.poster_engine import BilingualPosterEngine

def main():
    out_dir = "examples/poster_showcase_256"
    os.makedirs(out_dir, exist_ok=True)
    engine = BilingualPosterEngine()

    posters = [
        ("poster_01_cyberpunk", "مدينة المستقبل", "NEON CITY 2099", "Cyberpunk"),
        ("poster_02_cinema", "أسطورة الفارس", "LEGEND OF KNIGHT", "Cinema"),
        ("poster_03_sci_fi", "رحلة الفضاء", "GALAXY DISCOVERY", "Sci-Fi"),
        ("poster_04_fantasy", "عالم السحر", "REALM OF MAGIC", "Fantasy"),
        ("poster_05_retro", "مهرجان الألعاب", "RETRO ARCADE FEST", "Retro")
    ]

    print("Generating 256x256 bilingual (Arabic & English) poster samples...")
    for filename, title_ar, title_en, category in posters:
        filepath = os.path.join(out_dir, f"{filename}.png")
        poster_img = engine.generate_poster(
            title_ar=title_ar,
            title_en=title_en,
            category=category,
            width=256,
            height=256
        )
        poster_img.save(filepath)
        print(f" Saved: {filepath}")

    print("All 5 bilingual poster samples generated successfully!")

if __name__ == "__main__":
    main()
