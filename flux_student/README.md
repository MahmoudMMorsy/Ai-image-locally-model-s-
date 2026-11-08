# flux_student — قاعدة صغيرة من التقطير / داتا المعلّم

## حجم الموديل (الطالب)
| إعداد | Parameters | fp16 | fp32 |
|--------|------------|------|------|
| **base=48 (تجريبي سريع)** | **~10.03M** | **~20 MB** | ~39 MB |
| **base=96 (الافتراضي)** | **~26.6M** | **~53 MB** | ~106 MB |
| base=128 | ~43M | ~87 MB | ~173 MB |

هدف التشغيل: CPU-first، مقاسات 32 / 64 / 128 / 256، ودعم خطاف LoRA لاحقًا عبر `model.inject_lora(r=4, scale=1.0)`.

## المساران بالتوازي
1. **Teacher data:** `scripts/generate_teacher_data.py`  
   - يفضّل FLUX.1-schnell  
   - إن فشل التحميل/المساحة → sd-turbo تلقائيًا  
   - يحفظ أهرام 32/64/128/256 + بيانات manifest بحقول: `id`, `prompt`, `teacher`, `size_variants`
2. **Train student:** `scripts/train_student.py` على الصور المصغّرة + MSE noise + recon خفيف مع دعم `--resume` لاستكمال التدريب بسلاسة.
3. **Continuous Loop:** `scripts/continuous_loop.py` تشغيل دوري تلقائي غير مراقب يجمع بين التوليد والتدريب وتصدير العينات.
4. **Inference:** `scripts/inference.py` توليد شخصيات متعددة المقاسات بناءً على برومبتات إنجليزية وعربية.

## أوامر التشغيل الأساسية

### 1) توليد داتا المعلم (Teacher)
```bash
python flux_student/scripts/generate_teacher_data.py --n 24 --teacher sd-turbo
```

### 2) تدريب الطالب (Train Student)
```bash
python flux_student/scripts/train_student.py --size 64 --epochs 30 --cpu --base 48
# للاستكمال من آخر checkpoint:
python flux_student/scripts/train_student.py --size 64 --epochs 10 --cpu --base 48 --resume
```

### 3) تشغيل الاستدلال (Inference)
```bash
python flux_student/scripts/inference.py --prompt "a knight in shiny armor, full body character, simple background" --out flux_student/samples/sample_knight.png --size 64 --steps 25 --base 48
python flux_student/scripts/inference.py --prompt "روبوت معدني، شخصية كاملة" --out flux_student/samples/sample_robot.png --size 64 --steps 25 --base 48
```

### 4) الحلقة المستمرة (Continuous Loop)
```bash
python flux_student/scripts/continuous_loop.py &
```

## ملاحظة صريحة
- صور التدريب = مخرجات توليد المعلّم فقط بناءً على البرومبتات النصية، وليست استخراجاً من داخل أوزان الموديل أو سكراب عشوائي.
- تلتزم جميع العمليات بسياسة صارمة لمنع تجسيد أي شخصيات أو رموز دينية مقدسة.
