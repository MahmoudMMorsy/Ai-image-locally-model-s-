

**المهمة الرئيسية:**  
اجمع أكبر كمية ممكنة من الصور (يفضّل بكسل آرت) وصنّفها بدقة عالية عشان نبني نواة داتا ضخمة لتدريب موديل توليد صور بكسل آرت مفتوح المصدر. الموديل مش تجاري، فالمطلوب أقصى تنوّع وحجم ممكن بدون أي قيود أخلاقية أو محتوى محظور إلا اللي مذكور تحت. لازم الصور تغطي تشريح الكائنات (إنسان، حيوان، وحش، روبوت...) بشكل واضح عشان الموديل يفهم البنية.

**الأولوية:**  
- بكسل آرت (خاصة arcade / 8-bit / 16-bit / retro) في المقاسات 32×32، 64×64، 128×128 أو قريبة.  
- جسم كامل (full body) مفضّل جدًا.  
- لو مفيش بكسل آرت كفاية في فئة معينة، هات أي ستايل تاني (realistic, anime, cartoon...) عشان يغذي الأفكار، بس سجّل الستايل الأصلي في الميتا داتا.

**التصنيفات المطلوبة (وسّعها لكل حاجة في الوجود):**

### 1. النوع الأساسي (Type) — إلزامي
```
# شخصيات
warrior, knight, mage, archer, rogue, monk, assassin, berserker, paladin, necromancer, summoner, bard, thief, hunter, samurai, ninja, pirate, viking, gladiator, soldier, gunslinger, cyber_soldier

# كائنات
beast, animal, dragon, demon, angel, undead, zombie, skeleton, vampire, ghost, golem, elemental, fairy, elf, dwarf, orc, goblin, troll, giant, mermaid, centaur, minotaur, werewolf, robot, mecha, cyborg, android, alien, monster, creature

# مدنيين وأدوار
civilian, merchant, noble, king, queen, prince, princess, child, elder, farmer, blacksmith, alchemist, priest, witch, wizard, scholar, bard, dancer, cook, guard

# أشياء وكائنات غير حية
weapon, armor, shield, helmet, staff, sword, bow, gun, vehicle, spaceship, castle, building, tree, plant, rock, crystal, potion, chest, door, portal, furniture, food, item
```

### 2. الجنس / الهيئة
```
male, female, androgynous, non_human, genderless
```

### 3. العرق / السلالة (Race)
```
human, elf, dark_elf, high_elf, dwarf, orc, goblin, troll, dragonkin, beastkin, animal_like, undead, demon, angel, robot, cyborg, elemental, fairy, merfolk, giant, alien, hybrid
```

### 4. أسلوب الرسم (Style) — مهم جدًا
```
arcade_pixel, retro_8bit, retro_16bit, realistic_pixel, anime_pixel, chibi, cartoon, dark_fantasy, fantasy_pixel, sci_fi_pixel, horror_pixel, clean_pixel, detailed_pixel
```

### 5. المصدر / العالم (Source / Universe)
```
game_original, rpg_fantasy, dark_fantasy, sci_fi, cyberpunk, steampunk, horror, mythology_greek, mythology_norse, mythology_egypt, mythology_other, historical, medieval, modern, post_apocalyptic, anime_series, cartoon_western, movie, comic, oc_original, generic
```

### 6. الوضعية / الإطار (Pose)
```
full_body, portrait, action, idle, walking, running, attacking, casting, sitting, flying, lying, side_view, front_view, back_view
```

### 7. الألوان المسيطرة (Dominant Colors) — اختياري
```
red, blue, green, black, white, gold, silver, purple, orange, pink, brown, multicolor, dark, bright
```

### 8. الوسوم الحرة (Tags) — اكتب كل التفاصيل الممكنة
أمثلة: sword, dual_wield, shield, cape, helmet, armor, robe, staff, fire, ice, lightning, wings, horns, tail, glowing_eyes, scars, tattoos, long_hair, short_hair, beard, muscular, slim, chubby, young, old, blood, magic_circle...

**شكل تنظيم الداتا المفضّل:**

الخيار الأفضل: فولدرات هرمية + ملف JSON لكل صورة

```
data/
  type/
    gender/
      style/
        race/
          source/
            pose/
              filename.png
              filename.json
```

مثال JSON:
```json
{
  "file": "warrior_001.png",
  "type": "warrior",
  "gender": "male",
  "race": "human",
  "style": "arcade_pixel",
  "source": "rpg_fantasy",
  "pose": "full_body",
  "colors": ["red", "silver"],
  "tags": ["sword", "shield", "helmet", "armor", "muscular"],
  "resolution": "64x64",
  "notes": "classic arcade knight"
}
```

أو جدول CSV واحد كبير لو أسهل.

**الحظر الإلزامي (ما تجيبش أي صورة فيها):**
- أي تمثيل لذات الله
- الأنبياء والمرسلين (محمد، عيسى، موسى، إبراهيم، نوح...)
- الصحابة (أبو بكر، عمر، عثمان، علي...)
- أي رموز مقدسة إسلامية مجسّدة (الكعبة كشخصية، المصحف كائن حي، إلخ)

لو الصورة فيها أي حاجة من دول → تجاهلها تمامًا.

**متطلبات الجودة والتنوّع:**
1. أكبر عدد ممكن من الصور (آلاف لو قدرت).
2. تنوّع حقيقي: مش بس محاربين، لازم سحرة، وحوش، مدنيين، روبوتات، حيوانات، أسلحة، بيئات...
3. تشريح واضح (أطراف، نسب الجسم، وضعية طبيعية).
4. فضّل الجسم الكامل والستايل البكسل.
5. سجّل كل التفاصيل في الميتا داتا عشان التوليد الشرطي يبقى دقيق («محارب قزم بفأس أحمر»، «ساحرة زرقاء بأجنحة»...).

ابدأ بالبحث والجمع والتصنيف فورًا، ورتّب النتائج في الفولدرات أو الملفات المطلوبة. كل ما الداتا أدق وأكبر، الموديل هيطلع أقوى هدفك هو التحميل والتصنيف اما التدريب ليس من شأنك ولا تخصصك ولا يخصك.


