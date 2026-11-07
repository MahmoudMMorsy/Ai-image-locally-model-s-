# UNIFIED PIPELINE — مسار واحد

Updated: 2026-11-07

## هدف
مولد أركيد 64×64: شخصية جديدة + نص AR/EN → صورة، جودة قريبة من السبرايتات.

## تجميع المسارات
| مصدر | دور |
|------|-----|
| mm.trine (~5109 PNG) | PRIMARY |
| nano arcked dat | PRIMARY إضافي |
| nano_arcade_data | بعد فلتر 64×64 |
| arcade_t2i/ | محرك التدريب الرئيسي |
| flux_student/ | مساعد distill فقط |
| dataset_clean (101) | ممنوع كمصدر وحيد |

## ترتيب التنفيذ
0. فهرسة → `arcade_t2i/manifests/manifest_unified.jsonl`
1. palette48 + indices
2. discrete / T2I train على N الكامل
3. نص→صورة بعد gate
4. أنيميشن = نفس char_hash فقط
5. flux_student بالتوازي كمساعد

## تعارضات
- لا تدمج كل PRs المفتوحة لـ flux_student دفعة واحدة.
- فرع العمل المنسّق: `unified-pipeline-exec`
- hard stop إذا N < 1000 بعد الفهرسة.
