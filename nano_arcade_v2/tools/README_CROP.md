# Auto Spritesheet Crop

```bash
python nano_arcade_v2/tools/auto_spritesheet_crop.py sheet.jpg --out crops/ --size 64
```

- Detects regular grid automatically
- Removes sheet background
- Content-bbox + center pad to square
- Exports individual RGBA sprites + contact sheet + meta.json

Integrated into training data prep: run on every new sheet before adding to chars_only indices.
