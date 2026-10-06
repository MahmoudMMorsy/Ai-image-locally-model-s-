import os
import torch
from PIL import Image
import torchvision.transforms as T
from train_arabic_poster_model import LightweightPosterUNet, render_arabic, get_font
from PIL import ImageDraw

def generate_showcase():
    out_dir = "examples/2026-10-04_nano_arabic_poster_showcase"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightPosterUNet().to(device)
    weights_p = "models/nano_arabic_poster/nano_arabic_poster.pt"
    model.load_state_dict(torch.load(weights_p, map_location=device))
    model.eval()

    # 1. Text-to-Image Neural & Stochastic Generation
    print("Generating Text-to-Image poster samples...")
    prompts = [
        ("الفارس الاسطوري", "LEGENDARY KNIGHT", "showcase_t2i_01.png"),
        ("مغامرة الفضاء", "SPACE ADVENTURE", "showcase_t2i_02.png"),
        ("عرش التنين", "DRAGON THRONE", "showcase_t2i_03.png")
    ]

    for ar_txt, en_txt, filename in prompts:
        # Random initial latent/noise input
        z = torch.randn((1, 3, 256, 256), device=device) * 0.5 + 0.5
        tokens = torch.tensor([[hash(ar_txt) % 1000] * 10], dtype=torch.long, device=device)

        with torch.no_grad():
            out_tensor = model(z, tokens, noise_level=0.0)

        out_img = T.ToPILImage()(out_tensor.squeeze(0).cpu())

        # Overlay crisp typography
        draw = ImageDraw.Draw(out_img)
        font_ar = get_font(20)
        font_en = get_font(14)

        draw.text((128, 40), render_arabic(ar_txt), fill=(255, 230, 100), font=font_ar, anchor="mm")
        draw.text((128, 216), en_txt, fill=(0, 240, 255), font=font_en, anchor="mm")

        save_p = os.path.join(out_dir, filename)
        out_img.save(save_p)
        print(f"Saved Text-to-Image output: {save_p}")

    # 2. Image-to-Image Neural Conversion
    print("Generating Image-to-Image poster conversion sample...")
    input_img_p = "nano arabic dat/gameboy_classic/box_art_front/bilingual_arabic_primary/pixel_kufic_geometric/monochrome_dmg/256x256/gameboy_classic_box_art_front_0001_256x256.png"
    if os.path.exists(input_img_p):
        src_img = Image.open(input_img_p).convert("RGB")
        src_tensor = T.ToTensor()(src_img).unsqueeze(0).to(device)

        # Image-to-Image stochastic transformation
        with torch.no_grad():
            i2i_tensor = model(src_tensor, text_tokens=None, noise_level=0.15)

        i2i_img = T.ToPILImage()(i2i_tensor.squeeze(0).cpu())
        save_i2i_p = os.path.join(out_dir, "showcase_i2i_converted.png")
        i2i_img.save(save_i2i_p)
        print(f"Saved Image-to-Image output: {save_i2i_p}")

    print("Showcase generation complete!")

if __name__ == "__main__":
    generate_showcase()
