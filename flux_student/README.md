# flux_student — قاعدة صغيرة من التقطير / داتا المعلّم

## حجم الموديل (الطالب)
| إعداد | Parameters | fp16 | fp32 |
|--------|------------|------|------|
| **base=48 (تجريبي سريع)** | **~10M** | **~20 MB** | ~40 MB |
| **base=96 (الافتراضي)** | **~27M** | **~53 MB** | ~106 MB |
| base=128 | ~43M | ~87 MB | ~173 MB |

هدف التشغيل: CPU-first، مقاسات 32 / 64 / 128 / 256، LoRA لاحقًا من عندك.

## المساران بالتوازي
1. **Teacher data:** `scripts/generate_teacher_data.py`  
   - يفضّل FLUX.1-schnell  
   - إن فشل التحميل/المساحة → sd-turbo تلقائيًا  
   - يحفظ الخام + أهرام 32/64/128/256 (NEAREST للصغير)
2. **Train student:** `scripts/train_student.py` على الصور المصغّرة + MSE noise + recon خفيف

## أوامر
```bash
python scripts/generate_teacher_data.py --n 48 --teacher auto
python scripts/train_student.py --size 64 --epochs 50 --cpu --base 48
```

## ملاحظة صريحة
- صور التدريب = مخرجات المعلّم (توليد)، وليست استخراج من داخل أوزان FLUX.
- FLUX الكامل ثقيل؛ على بيئة محدودة يُستخدم sd-turbo كجسر تشغيل فوري لنفس الـ pipeline.
