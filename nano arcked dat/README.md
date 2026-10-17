# nano arcked dat

Comprehensive, classified, multi-taxonomy pixel art image dataset for open-source pixel art generative AI model training.

## Dataset Overview

`nano arcked dat` contains **1,692 unique pixel art images** paired 1-to-1 with rich JSON sidecars, indexed in a master CSV catalog and JSONL manifest.

- **Primary Focus**: Pixel art (Arcade, 8-bit, 16-bit, Retro, Clean, Detailed) at 32x32, 64x64, 128x128, and close resolutions.
- **Anatomy & Anatomy Diversity**: Comprehensive coverage of full-body humanoids, warriors, mages, archers, beasts, animals, monsters, dragons, robots, civilians, weapons, and environmental structures.
- **Sidecar Metadata**: Every PNG image is paired with a matching `.json` sidecar in the same directory.
- **Master Indexes**: `dataset_index.csv` and `manifests/dataset.jsonl`.

## Folder Hierarchy

All assets are organized in the following folder structure:

```
nano arcked dat/
  data/
    <type>/
      <gender>/
        <style>/
          <race>/
            <source>/
              <pose>/
                <filename>.png
                <filename>.json
  dataset_index.csv
  manifests/
    dataset.jsonl
    quarantine.jsonl
```

### Example Sidecar JSON (`<filename>.json`)

```json
{
  "file": "acdf422c305fe0f4_animal.png",
  "type": "animal",
  "gender": "androgynous",
  "race": "animal_like",
  "style": "sci_fi_pixel",
  "source": "cyberpunk",
  "pose": "full_body",
  "colors": [
    "multicolor"
  ],
  "tags": [
    "examples",
    "showcase",
    "nano",
    "pixel",
    "cyberpunk",
    "cat"
  ],
  "resolution": "64x64",
  "notes": "Aggregated asset from examples_showcase (64x64 sci_fi_pixel animal)"
}
```

## Taxonomy & Classifications

1. **Type (`type`)**:
   - *Characters*: `warrior`, `knight`, `mage`, `archer`, `rogue`, `monk`, `assassin`, `berserker`, `paladin`, `necromancer`, `summoner`, `bard`, `thief`, `hunter`, `samurai`, `ninja`, `pirate`, `viking`, `gladiator`, `soldier`, `gunslinger`, `cyber_soldier`.
   - *Creatures*: `beast`, `animal`, `dragon`, `demon`, `angel`, `undead`, `zombie`, `skeleton`, `vampire`, `ghost`, `golem`, `elemental`, `fairy`, `elf`, `dwarf`, `orc`, `goblin`, `troll`, `giant`, `mermaid`, `centaur`, `minotaur`, `werewolf`, `robot`, `mecha`, `cyborg`, `android`, `alien`, `monster`, `creature`.
   - *Civilians & Roles*: `civilian`, `merchant`, `noble`, `king`, `queen`, `prince`, `princess`, `child`, `elder`, `farmer`, `blacksmith`, `alchemist`, `priest`, `witch`, `wizard`, `scholar`, `dancer`, `cook`, `guard`.
   - *Items & Non-Living*: `weapon`, `armor`, `shield`, `helmet`, `staff`, `sword`, `bow`, `gun`, `vehicle`, `spaceship`, `structure`, `building`, `tree`, `plant`, `rock`, `crystal`, `potion`, `chest`, `door`, `portal`, `furniture`, `food`, `item`.

2. **Gender (`gender`)**:
   `male`, `female`, `androgynous`, `non_human`, `genderless`.

3. **Race / Breed (`race`)**:
   `human`, `elf`, `dark_elf`, `high_elf`, `dwarf`, `orc`, `goblin`, `troll`, `dragonkin`, `beastkin`, `animal_like`, `undead`, `demon`, `angel`, `robot`, `cyborg`, `elemental`, `fairy`, `merfolk`, `giant`, `alien`, `hybrid`.

4. **Style (`style`)**:
   `arcade_pixel`, `retro_8bit`, `retro_16bit`, `realistic_pixel`, `anime_pixel`, `chibi`, `cartoon`, `dark_fantasy`, `fantasy_pixel`, `sci_fi_pixel`, `horror_pixel`, `clean_pixel`, `detailed_pixel`.

5. **Source / Universe (`source`)**:
   `game_original`, `rpg_fantasy`, `dark_fantasy`, `sci_fi`, `cyberpunk`, `steampunk`, `horror`, `mythology_greek`, `mythology_norse`, `mythology_egypt`, `mythology_other`, `historical`, `medieval`, `modern`, `post_apocalyptic`, `anime_series`, `cartoon_western`, `movie`, `comic`, `oc_original`, `generic`.

6. **Pose (`pose`)**:
   `full_body`, `portrait`, `action`, `idle`, `walking`, `running`, `attacking`, `casting`, `sitting`, `flying`, `lying`, `side_view`, `front_view`, `back_view`.

7. **Dominant Colors (`colors`)**:
   `red`, `blue`, `green`, `black`, `white`, `gold`, `silver`, `purple`, `orange`, `pink`, `brown`, `multicolor`, `dark`, `bright`.

8. **Free Tags (`tags`)**:
   Extracted equipment, elemental traits, anatomy features, and visual keywords (e.g. `sword`, `armor`, `cape`, `helmet`, `muscular`, `fire`, `ice`, `wings`, `horns`).

## Sacred Religious Exclusion Policy

The dataset enforces a strict zero-tolerance prohibition filter against depicting sacred Islamic figures or religious symbols:
- No depictions of God or divine entities.
- No depictions of Prophets and Messengers (Muhammad, Jesus, Moses, Abraham, Noah, etc.).
- No depictions of Companions / Sahaba (Abu Bakr, Umar, Uthman, Ali, etc.).
- No embodied sacred Islamic symbols (e.g., Kaaba as a character, Quran as a living entity).

Any candidate assets matching restricted keywords are automatically excluded or logged to `manifests/quarantine.jsonl`.

## Verification

Run the automated verification script:

```bash
python3 verify_nano_arcked_dataset.py
```

This verifies:
- 1:1 PNG <-> JSON sidecar matching.
- JSON schema validity and taxonomy level depths.
- Image readability and integrity via PIL.
- Zero religious figure prohibition violations.
- CSV index and JSONL manifest integrity.
