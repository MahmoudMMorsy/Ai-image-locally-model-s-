import os
from poster_generator_256.poster_engine import BilingualPosterEngine

def main():
    out_dir = "examples/poster_showcase_256"
    os.makedirs(out_dir, exist_ok=True)
    engine = BilingualPosterEngine()

    posters = [
        ("poster_01_cyber_future", "مدينة المستقبل الرقمية", "FUTURE CYBER CITY", "Cyberpunk"),
        ("poster_02_space_odyssey", "مكوك استكشاف المجرة", "GALACTIC EXPLORER", "Sci-Fi"),
        ("poster_03_legend_knight", "أسطورة الفارس الشجاع", "LEGEND OF THE BRAVE KNIGHT", "Cinema"),
        ("poster_04_magic_realm", "مملكة السحر والغموض", "REALM OF MYSTIC MAGIC", "Fantasy"),
        ("poster_05_arcade_retro", "مهرجان الألعاب الكلاسيكية", "RETRO ARCADE FESTIVAL", "Retro"),
        ("poster_06_ocean_depths", "أعماق المحيط الغامض", "DEPTHS OF THE MYSTIC OCEAN", "Cinema"),
        ("poster_07_desert_caravan", "قافلة الصحراء الذهبيه", "GOLDEN DESERT CARAVAN", "Cinema"),
        ("poster_08_neon_samurai", "محارب الساموراي النيون", "NEON SAMURAI WARRIOR", "Cyberpunk"),
        ("poster_09_solar_energy", "طاقة الشمس المتجددة", "SOLAR POWER INNOVATION", "Sci-Fi"),
        ("poster_10_dragon_peak", "قمة التنين الأسود", "PEAK OF THE BLACK DRAGON", "Fantasy")
    ]

    print("Generating 10 high-quality 256x256 bilingual poster samples...")
    for idx, (filename, title_ar, title_en, category) in enumerate(posters, start=1):
        filepath = os.path.join(out_dir, f"{filename}.png")
        poster_img = engine.generate_poster(
            title_ar=title_ar,
            title_en=title_en,
            category=category,
            width=256,
            height=256,
            seed=idx * 10
        )
        poster_img.save(filepath)
        print(f"[{idx}/10] Saved: {filepath}")

    print("All 10 bilingual poster samples generated successfully!")

if __name__ == "__main__":
    main()
