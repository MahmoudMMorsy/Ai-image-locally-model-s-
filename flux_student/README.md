# flux_student — قاعدة صغيرة من التقطير / داتا المعلّم

مشروع تدريب نموذج ذكاء اصطناعي خفيف ومدمج (Student T2I Denoiser) لتوليد صور الشخصيات (Full-body Characters) باللغتين العربية والإنجليزية، مدرب من الصفر على مخرجات موديل معلّم (Teacher: `stabilityai/sd-turbo` أو `FLUX.1-schnell`) مع أهرام متعددة المقاسات (32×32, 64×64, 128×128, 256×256).

## حجم الموديل (StudentT2I)
| إعداد | Parameters | fp16 | fp32 | الحجم على القرص |
|--------|------------|------|------|-----------------|
| **base=48 (الافتراضي السريع)** | **~10.03M** | **~20 MB** | ~40 MB | **~39 MB** |
| **base=96** | **~26.6M** | **~53 MB** | ~106 MB | ~103 MB |
| base=128 | ~43M | ~87 MB | ~173 MB | ~170 MB |

## هيكل المشروع الفعلي على القرص
```
flux_student/
├── student/
│   ├── __init__.py
│   ├── model.py                # StudentT2I denoiser مع hook لـ LoRALinear
│   └── filter.py               # فلتر الأمان الديني الصارم (is_safe_prompt)
├── scripts/
│   ├── generate_teacher_data.py # توليد داتا من المعلم + أهرام المقاسات
│   ├── train_student.py        # تدريب الطالب مع دعم resume
│   ├── continuous_loop.py      # الحلقة المستمرة غير التفاعلية
│   ├── inference.py            # توليد استدلال بمقاسات متعددة
│   └── prompts_seed.txt        # برومبتات شخصيات كاملة (عربي + إنجليزي)
├── data/
│   ├── raw/                    # صور المعلم الأصلية (256x256)
│   ├── resized/{32,64,128,256}/# الأهرام المصغرة
│   └── manifest.jsonl          # بيانات: id, prompt, teacher, size_variants
├── checkpoints/
│   ├── student_latest.pt       # أحدث checkpoint دوري
│   └── student_final.pt        # الـ checkpoint الرئيسي (39 MB)
├── samples/                    # عينات استدلال حقيقية من الطالب
├── tests/
│   └── test_pipeline.py        # اختبارات معمارية، فلتر، واستدلال
├── README.md
├── STATUS.md
├── TRAINING_LOG.md
├── DATA_POLICY.md
├── continuous.log
├── continuous_status.json
└── continuous.pid
```

## أوامر التشغيل الأساسية

### 1. توليد داتا من المعلم (Teacher Generation)
```bash
python flux_student/scripts/generate_teacher_data.py --n 24 --teacher sd-turbo --steps 1
```

### 2. تدريب الطالب (Student Training)
```bash
python flux_student/scripts/train_student.py --size 64 --epochs 25 --cpu --base 48 --batch 4 --lr 3e-4 --resume
```

### 3. الاستدلال وتوليد الشخصيات (Inference)
```bash
python flux_student/scripts/inference.py --prompt "a knight in shiny armor, full body character" --size 64 --out flux_student/samples/knight.png
python flux_student/scripts/inference.py --prompt "روبوت معدني، شخصية كاملة" --size 128 --out flux_student/samples/robot.png
```

### 4. تشغيل الحلقة المستمرة في الخلفية (Continuous Loop)
```bash
python flux_student/scripts/continuous_loop.py > flux_student/continuous_run.log 2>&1 &
echo $! > flux_student/continuous.pid
```
لمتابعة الحالة:
```bash
cat flux_student/continuous_status.json
tail -f flux_student/continuous.log
```
