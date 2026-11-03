# UNIFIED PIPELINE — مسار واحد للمشروع

Updated: 2026-11-04

هدف واحد: مولد أركيد 64×64 يطلع **شخصية جديدة** (مش نسخة تدريب) + نص عربي/إنجليزي → صورة، بجودة قريبة من السبرايتات.

---

## ماذا نجمع (وماذا نترك)

| مصدر | الدور بعد التجميع |
|------|-------------------|
| `mm.trine/` (~5100 PNG) | **PRIMARY** تدريب شخصيات/أوضاع |
| `nano arcked dat/` | **PRIMARY** إضافي (تصنيفات + official_cc0 + data) |
| `nano_arcade_data/` | إضافي بعد فلتر 64×64 |
| `examples/` | اختياري |
| `arcade_t2i/` | **محرك التدريب T2I الرسمي** (manifest + palette + maskdiff + text) |
| `flux_student/` | **مسار مساعد**: teacher sd-turbo + student صغير (distill) — مش بديل عن mm.trine |
| `models/nano_pixel_*` | أوزان قديمة/جزئية — لا تعتبر production إلا بعد إعادة تدريب على N كامل |
| `dataset_clean.zip` (101) | **ممنوع** كمصدر وحيد |
| `pixel_art_engine/` / posters | منفصل عن هدف الشخصية من الصفر |

---

## المسار الموحد (ترتيب إلزامي)

### المرحلة 0 — فهرسة كل الصور
1. امسح: `mm.trine/**`, `nano arcked dat/**`, `nano_arcade_data/**`, `examples/**`
2. فلتر: 64×64 NEAREST، إسقاط الإطارات شبه الفارغة
3. اكتب: `arcade_t2i/manifests/manifest_unified.jsonl`
   - path, source, class, caption_ar, caption_en, char_hash (من mm.trine إن وُجد)
4. **Hard stop** إذا N < 1000 — غلط في المسار
5. الهدف: **N ≥ 15000** إن أمكن (حسب JULES_T2I_PLAN)

### المرحلة 1 — Palette + indices
1. palette 48 لون مشترك
2. `arcade_t2i/checkpoints/indices_unified_u8.pt` شكل `[N,64,64]`
3. تحقق reconstruction من index → PNG

### المرحلة 2 — نموذج من الصفر (الأساسي)
1. Discrete **masked diffusion** على indices (مرجع: `nano_pixel_mob/.../discrete_diffusion` + `arcade_t2i/model_t2i.py`)
2. تدريب على **كل** الـ indices — مش 26 أو 101
3. Gate: silhouette واضح لفارس/ساحر/روبوت من sampling حر (مش inpaint فقط)

### المرحلة 3 — نص → صورة
1. captions من المسار + vocab (`arcade_t2i/bilingual_vocab.json`, `text_lang.py`)
2. conditioning على الـ diffusion
3. `generate_t2i.py --prompt "فارس"` و `--prompt "knight"`

### المرحلة 4 — flux_student (مساعد بالتوازي)
1. يكمّل توليد أزواج teacher من برومبت فقط
2. student صغير CPU للتجربة السريعة
3. **لا يستبدل** تدريب المرحلة 2 على mm.trine

### المرحلة 5 — أنيميشن
1. فقط بعد Gate المرحلة 2/3
2. نفس `char_hash` من mm.trine — أوضاع متتابعة → GIF

---

## نقطة الدخول للوكلاء (Jules / Grok / Manus)

| مهمة | مجلد العمل |
|------|------------|
| فهرسة + تدريب T2I الكامل | `arcade_t2i/` |
| خطة T2I التفصيلية | `arcade_t2i/JULES_T2I_PLAN.md` |
| قواعد الداتا | `AGENTS.md` |
| خطة من الصفر | `PLAN_FROM_SCRATCH.md` |
| Distill سريع | `flux_student/` |
| هذا الملف | `UNIFIED_PIPELINE.md` |

### أوامر مختصرة
```bash
# 1) بناء manifest موحّد
cd arcade_t2i && python build_manifest.py --full-repo

# 2) تدريب
python train_t2i.py --manifest manifests/manifest_unified.jsonl

# 3) تقييم
python generate_t2i.py --prompt "فارس" --out samples/eval_knight.png
python run_eval.py

# 4) مساعد distill (اختياري)
cd ../flux_student && nohup python3 -u scripts/continuous_loop.py >> continuous.log 2>&1 &
```

---

## حالة صادقة الآن (2026-11-04)

| مسار | حالة |
|------|------|
| arcade_t2i free-gen | GATE FAIL تاريخيًا (silhouette ضعيف) |
| flux_student | ~36 زوج معلم، loss ~0.64 عند Jules؛ محلي أقل داتا |
| من الصفر على mm.trine كامل | **لم يكتمل** كـ production |
| نص→صورة جاهز للشحن | **لا** |

---

## قاعدة ذهبية

- مسار واحد للإنتاج: **arcade_t2i + كل صور الريبو**
- flux_student = مساعد
- أي تقرير "تم" بدون N كبير وبدون عينات فارس/ساحر/روبوت واضحة = مرفوض
