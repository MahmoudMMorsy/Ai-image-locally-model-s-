import os
from pixel_art_engine.engine import PixelSpriteEngine

def main():
    output_dir = "examples/showcase_39_characters"
    os.makedirs(output_dir, exist_ok=True)
    engine = PixelSpriteEngine()

    prompts = [
        "Cyberpunk Hacker", "Flame Elementalist", "Shadow Assassin", "Golden Mech",
        "Elven Ranger", "Obsidian Knight", "Ice Wizard", "Retro Orc Warrior",
        "Frost Dragon", "Xenomorph Alien", "Vampire Count", "Skeleton Archer",
        "Samurai Master", "Space Marine Sentinel", "Neon Rogue", "Demon Lord",
        "Holy Paladin", "Steampunk Dwarf", "Valkyrie Warrior", "Necromancer",
        "Pirate Captain", "Galactic Bounty Hunter", "Minotaur Berserker", "Gargoyle Guardian",
        "Celestial Angel", "Lich Sorcerer", "Desert Nomad", "Toxic Mutant",
        "Clockwork Golem", "Phoenix Warden", "Abyssal Kraken", "Sun Priest",
        "Thunder Berserker", "Gladiator Champion", "Shadow Necromancer",
        "Chrono Templar", "Bionic Huntress", "Rune Enchanter", "Void Sentinel"
    ]

    print(f"Generating {len(prompts)} distinct character sprites...")
    for idx, prompt in enumerate(prompts, start=1):
        filename = f"char_{idx:02d}_{prompt.lower().replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        sprite = engine.generate_sprite(prompt=prompt, seed=idx)
        sprite.save(filepath)
        print(f"[{idx}/39] Saved: {filepath}")

    print("All 39 character images generated successfully!")

if __name__ == "__main__":
    main()
