# NanoPixel Android Mobile AI Workflow 📱🤖

هذا المجلد يحتوي على كود وسكريبتات تشغيل وتدريب نموذج **NanoPixel AI** محلياً على هواتف الأندرويد بدون إنترنت.

## المكونات الأساسية:
1. `app/src/main/java/com/nanopixel/app/PixelArtMobileEngine.kt`: محرك تشغيل النموذج على الأندرويد باستخدام **ONNX Runtime Mobile**.
2. `scripts/export_android_onnx.py`: سكريبت تصدير وضغط النموذج لصيغة `.onnx` و `.tflite` خفيفة للموبيل (<15MB).
3. `scripts/test_android_workflow.py`: سكريبت محاكاة واختبار تشغيل وتدريب النموذج محلياً.

## خطوات الاستخدام على الأندرويد:
1. قم بفتح المجلد في **Android Studio**.
2. ابني التطبيق وركب محرك `onnxruntime-android`.
3. اضغط على "Train Locally" لتدريب النموذج على صور الهاتف محلياً.
4. ادخل الوصف النصي واضغط "Generate Sprite" لتوليد شخصية البكسل فورياً.

## 256x256 Poster Generation Model (ONNX):
- Model Path: `poster_generator_256/weights/poster_generator_256.onnx`
- Supports 256x256 resolution poster synthesis with Arabic & English text conditioning.
- Optimized for Android NPU / ARM64 execution using ONNX Runtime Mobile.
