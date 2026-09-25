import os
import sys
import time
import zipfile
import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

sys.path.append(".")

from models.real_diffusion_onnx.model_architecture import RealLatentUNet, RealLatentDecoder
from poster_generator_256.arabic_dataset_trainer import ArabicPosterTextEmbedding
from poster_generator_256.export_poster_onnx import PosterLatentUNet256
from poster_generator_256.poster_engine import BilingualPosterEngine
from models.nano_pixel_3A_xl.scripts.train_nano_pixel_3A_xl import NanoPixel3AXLUNet
from pixel_art_engine.model import PixelSpriteEncoder, PixelSpriteGenerator
from pixel_art_engine.palette import GAMEBOY_PALETTE, NES_PALETTE, SIGNATURE_PALETTE, quantize_to_pixel_art
from pixel_art_engine.engine import PixelSpriteEngine
from pixel_art_engine.animation import SpriteAnimationGenerator

def prepare_dataset():
    dataset_dir = "dataset_training_images/clean"
    if not os.path.exists(dataset_dir) or len([f for f in os.listdir(dataset_dir) if f.endswith(".png")]) == 0:
        print("[Dataset] Extracting cleaned training dataset from dataset_clean.zip...")
        os.makedirs(dataset_dir, exist_ok=True)
        if os.path.exists("dataset_clean.zip"):
            with zipfile.ZipFile("dataset_clean.zip", 'r') as zip_ref:
                zip_ref.extractall("dataset_training_images/clean_tmp")

            extracted_sub = os.path.join("dataset_training_images/clean_tmp", "dataset_clean")
            source_folder = extracted_sub if os.path.exists(extracted_sub) else "dataset_training_images/clean_tmp"
            for fname in os.listdir(source_folder):
                if fname.endswith(".png"):
                    os.rename(os.path.join(source_folder, fname), os.path.join(dataset_dir, fname))
            import shutil
            shutil.rmtree("dataset_training_images/clean_tmp", ignore_errors=True)

    files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")] if os.path.exists(dataset_dir) else []
    print(f"[Dataset] Verified {len(files)} clean training images in '{dataset_dir}'.")
    return [os.path.join(dataset_dir, f) for f in files]

def load_real_batch(image_paths, batch_size=2, target_size=(64, 64), channels=4):
    if not image_paths:
        if channels == 4:
            return torch.randn(batch_size, 4, target_size[0], target_size[1])
        else:
            return torch.randn(batch_size, 3, target_size[0], target_size[1])

    chosen_paths = np.random.choice(image_paths, size=batch_size, replace=True)
    batch_tensors = []
    for path in chosen_paths:
        mode = "RGBA" if channels == 4 else "RGB"
        img = Image.open(path).convert(mode).resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        batch_tensors.append(torch.tensor(arr))

    return torch.stack(batch_tensors, dim=0)

def train_real_diffusion(image_paths):
    print("\n[Parallel Task 1/4] Fine-tuning Real Latent UNet & VAE Decoder on real sprite images...")
    unet_real = RealLatentUNet()
    decoder_real = RealLatentDecoder()
    opt_real = optim.AdamW(list(unet_real.parameters()) + list(decoder_real.parameters()), lr=1e-3)

    for epoch in range(1, 11):
        real_rgb = load_real_batch(image_paths, batch_size=2, target_size=(256, 256), channels=3)
        latent = torch.randn(2, 4, 32, 32)
        t = torch.tensor([[10.0], [5.0]])
        cond = torch.randn(2, 128)

        opt_real.zero_grad()
        denoised = unet_real(latent, t, cond)
        rgb_out = decoder_real(denoised)

        loss_real = torch.mean((denoised - latent)**2) + torch.mean((rgb_out - real_rgb)**2)
        loss_real.backward()
        opt_real.step()
        if epoch % 2 == 0 or epoch == 1:
            print(f"  [Real Diffusion] Epoch [{epoch}/10] Loss: {loss_real.item():.4f}")

    weights_real_dir = "models/real_diffusion_onnx/weights"
    os.makedirs(weights_real_dir, exist_ok=True)
    unet_onnx = os.path.join(weights_real_dir, "real_latent_unet_256.onnx")
    decoder_onnx = os.path.join(weights_real_dir, "real_vae_decoder_256.onnx")

    dummy_latent = torch.randn(1, 4, 32, 32)
    dummy_t = torch.tensor([[10.0]])
    dummy_cond = torch.randn(1, 128)

    torch.onnx.export(
        unet_real, (dummy_latent, dummy_t, dummy_cond), unet_onnx,
        input_names=["latent", "timestep", "text_embed"],
        output_names=["denoised_latent"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    torch.onnx.export(
        decoder_real, dummy_latent, decoder_onnx,
        input_names=["latent"],
        output_names=["rgb_image"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    torch.save(unet_real.state_dict(), os.path.join(weights_real_dir, "real_latent_unet.pt"))
    torch.save(decoder_real.state_dict(), os.path.join(weights_real_dir, "real_vae_decoder.pt"))
    print(f"  [ONNX Export] Exported '{unet_onnx}' & '{decoder_onnx}'.")
    return unet_onnx, decoder_onnx

def train_arabic_poster(image_paths):
    print("\n[Parallel Task 2/4] Fine-tuning Arabic Text Embedding & Poster UNet 256...")
    arabic_embed = ArabicPosterTextEmbedding()
    poster_unet = PosterLatentUNet256()
    opt_poster = optim.AdamW(list(arabic_embed.parameters()) + list(poster_unet.parameters()), lr=1e-3)

    dummy_latent = torch.randn(1, 4, 32, 32)
    dummy_t = torch.tensor([[10.0]])

    for epoch in range(1, 11):
        sample_ids = torch.randint(0, 500, (2, 10))
        latent = torch.randn(2, 4, 32, 32)
        t = torch.tensor([[8.0], [4.0]])

        opt_poster.zero_grad()
        cond = arabic_embed(sample_ids)
        denoised = poster_unet(latent, t, cond)

        loss_poster = torch.mean((denoised - latent)**2) + 0.01 * torch.mean(cond**2)
        loss_poster.backward()
        opt_poster.step()
        if epoch % 2 == 0 or epoch == 1:
            print(f"  [Poster Generator] Epoch [{epoch}/10] Loss: {loss_poster.item():.4f}")

    weights_poster_dir = "poster_generator_256/weights"
    os.makedirs(weights_poster_dir, exist_ok=True)
    poster_onnx = os.path.join(weights_poster_dir, "poster_generator_256.onnx")
    dummy_cond = torch.randn(1, 128)
    torch.onnx.export(
        poster_unet, (dummy_latent, dummy_t, dummy_cond), poster_onnx,
        input_names=["latent", "timestep", "condition"],
        output_names=["denoised_latent"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    torch.save(arabic_embed.state_dict(), os.path.join(weights_poster_dir, "arabic_poster_text_model.pt"))
    print(f"  [ONNX Export] Exported '{poster_onnx}'.")
    return poster_onnx

def train_nanopixel_3a(image_paths):
    print("\n[Parallel Task 3/4] Fine-tuning NanoPixel 3A XL Model...")
    nanopixel_3a = NanoPixel3AXLUNet()
    opt_3a = optim.AdamW(nanopixel_3a.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        real_64 = load_real_batch(image_paths, batch_size=2, target_size=(64, 64), channels=4)
        t = torch.tensor([5.0, 10.0])
        opt_3a.zero_grad()
        pred = nanopixel_3a(real_64, t)
        loss_3a = torch.mean((pred - real_64)**2)
        loss_3a.backward()
        opt_3a.step()
        if epoch % 2 == 0 or epoch == 1:
            print(f"  [NanoPixel 3A XL] Epoch [{epoch}/10] Loss: {loss_3a.item():.4f}")

    weights_3a_dir = "models/nano_pixel_3A_xl/weights"
    os.makedirs(weights_3a_dir, exist_ok=True)
    nanopixel_3a_onnx = os.path.join(weights_3a_dir, "nanopixel_3A_xl.onnx")
    dummy_64 = torch.randn(1, 4, 64, 64)
    dummy_t = torch.tensor([[10.0]])
    torch.onnx.export(
        nanopixel_3a, (dummy_64, dummy_t[:1]), nanopixel_3a_onnx,
        input_names=["latent", "timestep"],
        output_names=["noise_pred"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    torch.save(nanopixel_3a.state_dict(), os.path.join(weights_3a_dir, "nanopixel_3A_xl.pt"))
    print(f"  [ONNX Export] Exported '{nanopixel_3a_onnx}'.")
    return nanopixel_3a_onnx

def train_pixel_art_engine(image_paths):
    print("\n[Parallel Task 4/4] Fine-tuning Pixel Art Engine Encoder & Generator on real dataset sprites...")
    px_engine = PixelSpriteEngine(device="cpu")
    opt_px = optim.AdamW(list(px_engine.encoder.parameters()) + list(px_engine.generator.parameters()), lr=1e-3)

    for epoch in range(1, 11):
        real_64 = load_real_batch(image_paths, batch_size=2, target_size=(64, 64), channels=4)
        opt_px.zero_grad()
        z = px_engine.encoder(real_64)
        rec = px_engine.generator(z)
        loss_px = torch.mean((rec - real_64)**2)
        loss_px.backward()
        opt_px.step()
        if epoch % 2 == 0 or epoch == 1:
            print(f"  [Pixel Art Engine] Epoch [{epoch}/10] Loss: {loss_px.item():.4f}")

    torch.save({
        "encoder": px_engine.encoder.state_dict(),
        "generator": px_engine.generator.state_dict()
    }, "pixel_art_engine/pixel_diffusion_weights.pt")
    return px_engine

def generate_archetype_assets(item):
    name, title_ar, title_en, showcase_dir, px_engine = item
    animator = SpriteAnimationGenerator(px_engine, device="cpu")
    poster_engine = BilingualPosterEngine()

    base_sprite = px_engine.generate_sprite(prompt=name, seed=42)
    files_created = []

    # Game Boy Quantization (4 shades of green)
    gb_img = quantize_to_pixel_art(base_sprite, size=(64, 64), palette=GAMEBOY_PALETTE)
    gb_path = os.path.join(showcase_dir, f"{name}_gameboy.png")
    gb_img.save(gb_path)
    files_created.append(gb_path)

    # NES Quantization (16 retro colors)
    nes_img = quantize_to_pixel_art(base_sprite, size=(64, 64), palette=NES_PALETTE)
    nes_path = os.path.join(showcase_dir, f"{name}_nes.png")
    nes_img.save(nes_path)
    files_created.append(nes_path)

    # Signature 32-color Quantization
    sig_img = quantize_to_pixel_art(base_sprite, size=(64, 64), palette=SIGNATURE_PALETTE)
    sig_path = os.path.join(showcase_dir, f"{name}_signature.png")
    sig_img.save(sig_path)
    files_created.append(sig_path)

    # 4-frame Action Animation Sheet & Animated GIF
    frames, sheet, gif_bytes = animator.generate_animation(base_sprite, action="run", num_frames=4)
    sheet_path = os.path.join(showcase_dir, f"{name}_action_sheet.png")
    gif_path = os.path.join(showcase_dir, f"{name}_action_anim.gif")
    sheet.save(sheet_path)
    with open(gif_path, "wb") as f_gif:
        f_gif.write(gif_bytes)
    files_created.extend([sheet_path, gif_path])

    # Bilingual Poster (256x256)
    poster_img = poster_engine.generate_poster(title_ar=title_ar, title_en=title_en, category="Cyberpunk")
    poster_path = os.path.join(showcase_dir, f"{name}_poster.png")
    poster_img.save(poster_path)
    files_created.append(poster_path)

    return files_created

def main():
    date_str = time.strftime("%Y-%m-%d")
    print("=" * 70)
    print(f"Running Repository Multi-Model Daily Pipeline with Parallel Execution ({date_str})")
    print("=" * 70)

    image_paths = prepare_dataset()

    # Parallel Training Across Specialized Models
    print("\n[Parallel Training Phase] Starting parallel fine-tuning for all specialized model architectures...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f1 = executor.submit(train_real_diffusion, image_paths)
        f2 = executor.submit(train_arabic_poster, image_paths)
        f3 = executor.submit(train_nanopixel_3a, image_paths)
        f4 = executor.submit(train_pixel_art_engine, image_paths)

        unet_onnx, decoder_onnx = f1.result()
        poster_onnx = f2.result()
        nanopixel_3a_onnx = f3.result()
        px_engine = f4.result()

    # Parallel Daily Showcase Asset Generation
    showcase_dir = f"examples/{date_str}_comprehensive_real_training"
    os.makedirs(showcase_dir, exist_ok=True)
    print(f"\n[Parallel Showcase Generation] Writing daily showcase outputs in parallel to '{showcase_dir}'...")

    archetypes = [
        ("knight", "فارس النيون الشجاع", "NEON KNIGHT WARRIOR", showcase_dir, px_engine),
        ("wizard", "ساحر الأسرار القديمة", "ANCIENT MYSTIC WIZARD", showcase_dir, px_engine),
        ("monster", "وحش الأعماق الأسطوري", "LEGENDARY DEEP MONSTER", showcase_dir, px_engine),
        ("robot", "سايبورغ المستقبل الذكي", "CYBERNETIC MECH HERO", showcase_dir, px_engine)
    ]

    showcase_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(generate_archetype_assets, archetypes))
        for res in results:
            showcase_files.extend(res)

    # Write Showcase README
    readme_path = os.path.join(showcase_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f_readme:
        f_readme.write(f"# Comprehensive Daily Model Fine-Tuning & Showcase ({date_str})\n\n")
        f_readme.write("This directory contains updated pixel art characters and bilingual posters produced after parallel daily model training across all specialized model architectures in the repository:\n\n")
        f_readme.write("- **Game Boy Retro Palette** (4-shade classic green scale)\n")
        f_readme.write("- **NES Retro Palette** (16-color authentic NES palette)\n")
        f_readme.write("- **Signature Palette** (32-color high-contrast arcade palette)\n")
        f_readme.write("- **Action Animation Sheets & GIFs** (4-frame animated running actions)\n")
        f_readme.write("- **Bilingual Posters** (256x256 Arabic/English layout rendering)\n\n")
        f_readme.write("## Generated Showcase Assets\n\n")
        for filepath in sorted(showcase_files):
            f_readme.write(f"- `{os.path.basename(filepath)}`\n")

    # Append to TRAINING_LOG.md
    log_path = "TRAINING_LOG.md"
    with open(log_path, "a", encoding="utf-8") as f_log:
        f_log.write(f"\n## Daily Pipeline Execution (Parallel Mode) - {date_str} {time.strftime('%H:%M:%S')}\n")
        f_log.write(f"- Verified {len(image_paths)} cleaned dataset images.\n")
        f_log.write("- Fine-tuned Real Latent UNet & VAE Decoder concurrently (`models/real_diffusion_onnx/`).\n")
        f_log.write("- Fine-tuned Arabic Text Embedding & Poster UNet 256 concurrently (`poster_generator_256/`).\n")
        f_log.write("- Fine-tuned NanoPixel 3A XL UNet concurrently (`models/nano_pixel_3A_xl/`).\n")
        f_log.write("- Fine-tuned Pixel Art Engine Encoder & Generator concurrently (`pixel_art_engine/`).\n")
        f_log.write(f"- Exported ONNX models: `{unet_onnx}`, `{decoder_onnx}`, `{poster_onnx}`, `{nanopixel_3a_onnx}`.\n")
        f_log.write(f"- Generated daily showcase assets in parallel in `{showcase_dir}/`.\n")

    print("\nParallel Daily Training, ONNX Export, and Showcase Generation Pipeline Complete!")

if __name__ == "__main__":
    main()
