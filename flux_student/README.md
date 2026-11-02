# flux_student — قاعدة صغيرة من التقطير / داتا المعلّم

مشروع توليد شخصيات كرتونية وأركيد من النصوص مباشرة عبر تدريب موديل طالب مضغوط (Student T2I) مستخلص من موديل معلّم مفتوح (`sd-turbo` أو `FLUX.1-schnell`).

## حجم الموديل (الطالب)
| إعداد | Parameters | fp16 | fp32 (حجم الـ Checkpoint) |
|--------|------------|------|---------------------------|
| **base=48 (المرحلة الأولى)** | **~10.03M** | **~20 MB** | **~38.3 MB** |
| **base=96 (المرحلة الثانية)** | **~26.6M** | **~53 MB** | **~106 MB** |
| base=128 | ~43M | ~87 MB | ~173 MB |

## المسارات والتشغيل
1. **توليد داتا المعلّم:** `scripts/generate_teacher_data.py`
   - توليد من البرومبت فقط دون أي سكراب عشوائي
   - حفظ الأصل في `data/raw/` وتصغير لـ `data/resized/{32,64,128,256}/`
   - تسجيل الميتاداتا في `data/manifest.jsonl`

2. **تدريب الطالب:** `scripts/train_student.py`
   - تدريب denoiser على داتا 64×64
   - حفظ الـ checkpoints في `checkpoints/student_final.pt`

3. **الاستدلال (Inference):** `scripts/inference.py`
   - توليد صور شخصيات من البرومبتات العربية والإنجليزية

4. **الحلقة المستمرة:** `scripts/continuous_loop.py`
   - حلقة خلفية مستقلة تقوم بالتوليد والتدريب وتحديث الـ checkpoints وحفظ العينات

## أوامر التشغيل
```bash
# 1. توليد داتا من المعلّم
python3 flux_student/scripts/generate_teacher_data.py --n 16 --teacher sd-turbo --steps 1

# 2. تدريب موديل الطالب
python3 flux_student/scripts/train_student.py --size 64 --epochs 25 --batch 4 --cpu --base 48

# 3. الاستدلال وتوليد شخصية
python3 flux_student/scripts/inference.py --prompt "a knight in shiny armor, full body character, simple background" --out flux_student/samples/sample_knight.png --size 64

# 4. تشغيل الحلقة المستمرة في الخلفية
python3 flux_student/scripts/continuous_loop.py > flux_student/continuous.log 2>&1 &
```
