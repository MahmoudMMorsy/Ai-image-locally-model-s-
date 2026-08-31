import os
import zipfile
from PIL import Image

def prepare_dataset():
    raw_zip = "dataset_raw.zip"
    out_dir = "android_mobile_workflow/app/src/main/assets/mobile_dataset"
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(raw_zip):
        with zipfile.ZipFile(raw_zip, 'r') as zip_ref:
            zip_ref.extractall("android_mobile_workflow/tmp_raw")

        count = 0
        for root, dirs, files in os.walk("android_mobile_workflow/tmp_raw"):
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg'):
                    img_path = os.path.join(root, f)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((256, 256), Image.NEAREST)
                    img.save(os.path.join(out_dir, f"mobile_sample_{count:03d}.png"))
                    count += 1
                    if count >= 10:
                        break
        print(f"Packaged {count} mobile training samples into {out_dir}")

if __name__ == "__main__":
    prepare_dataset()
