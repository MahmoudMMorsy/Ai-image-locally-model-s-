"""
Sequential Character Animation Generator
Generates animated action sequences (Running, Walking, Attacking, Idle, Jumping)
for 64x64 pixel art characters while preserving identity and style.
"""
import torch
import numpy as np
from PIL import Image
import imageio
from typing import List, Dict, Tuple
from .palette import quantize_to_pixel_art, SIGNATURE_PALETTE
from .model import PixelSpriteEncoder, PixelSpriteGenerator


ACTION_TRANSFORMS = {
    "idle": [
        {"y_shift": 0, "leg_angle": 0, "arm_angle": 0, "head_tilt": 0},
        {"y_shift": -1, "leg_angle": 0, "arm_angle": 1, "head_tilt": 0},
        {"y_shift": -1, "leg_angle": 0, "arm_angle": 2, "head_tilt": 1},
        {"y_shift": 0, "leg_angle": 0, "arm_angle": 0, "head_tilt": 0},
    ],
    "walk": [
        {"y_shift": 0, "leg_angle": -2, "arm_angle": 3, "head_tilt": 0},
        {"y_shift": -1, "leg_angle": -1, "arm_angle": 1, "head_tilt": 0},
        {"y_shift": 0, "leg_angle": 2, "arm_angle": -3, "head_tilt": 0},
        {"y_shift": -1, "leg_angle": 1, "arm_angle": -1, "head_tilt": 0},
    ],
    "run": [
        {"y_shift": -2, "leg_angle": -4, "arm_angle": 5, "head_tilt": 1},
        {"y_shift": -1, "leg_angle": -2, "arm_angle": 2, "head_tilt": 1},
        {"y_shift": -3, "leg_angle": 4, "arm_angle": -5, "head_tilt": 1},
        {"y_shift": -1, "leg_angle": 2, "arm_angle": -2, "head_tilt": 1},
    ],
    "attack": [
        {"y_shift": 0, "leg_angle": -1, "arm_angle": -2, "head_tilt": 0},
        {"y_shift": -1, "leg_angle": -2, "arm_angle": -5, "head_tilt": -1}, # Windup
        {"y_shift": 0, "leg_angle": 3, "arm_angle": 8, "head_tilt": 1},   # Slash/Strike
        {"y_shift": 0, "leg_angle": 1, "arm_angle": 2, "head_tilt": 0},   # Recovery
    ],
    "jump": [
        {"y_shift": 1, "leg_angle": -1, "arm_angle": -1, "head_tilt": 0}, # Crouch
        {"y_shift": -4, "leg_angle": -3, "arm_angle": 4, "head_tilt": 1}, # Launch
        {"y_shift": -6, "leg_angle": -2, "arm_angle": 2, "head_tilt": 1}, # Peak
        {"y_shift": -2, "leg_angle": 1, "arm_angle": -2, "head_tilt": 0}, # Fall
    ]
}


class SpriteAnimationGenerator:
    """
    Generates consistent sequential animation frames for any given 64x64 pixel character.
    """
    def __init__(self, engine=None, device="cpu"):
        self.device = torch.device(device)
        if engine is not None:
            self.encoder = engine.encoder
            self.generator = engine.generator
        else:
            self.encoder = PixelSpriteEncoder(latent_dim=64).to(self.device)
            self.generator = PixelSpriteGenerator(latent_dim=64, condition_dim=32).to(self.device)

        self.encoder.eval()
        self.generator.eval()

    def _apply_frame_transform(self, char_tensor: torch.Tensor, transform: dict) -> torch.Tensor:
        """Applies motion offsets, limb rotations, and pixel shifts for dynamic action poses."""
        y_shift = transform.get("y_shift", 0)
        leg_angle = transform.get("leg_angle", 0)
        arm_angle = transform.get("arm_angle", 0)
        head_tilt = transform.get("head_tilt", 0)

        t = char_tensor.clone()

        # 1. Vertical body bounce shift
        if y_shift != 0:
            t = torch.roll(t, shifts=y_shift, dims=2)

        # 2. Leg motion region (bottom half of 64x64: y in range [44, 64])
        if leg_angle != 0:
            leg_shift = int(np.sign(leg_angle) * max(1, abs(leg_angle) // 2))
            t[:, :, 44:60, :] = torch.roll(t[:, :, 44:60, :], shifts=leg_shift, dims=3)

        # 3. Arm motion region (middle sides: y in range [24, 44])
        if arm_angle != 0:
            arm_shift = int(np.sign(arm_angle) * max(1, abs(arm_angle) // 2))
            t[:, :, 24:44, :32] = torch.roll(t[:, :, 24:44, :32], shifts=arm_shift, dims=2)
            t[:, :, 24:44, 32:] = torch.roll(t[:, :, 24:44, 32:], shifts=-arm_shift, dims=2)

        # 4. Head tilt region (top section: y in range [0, 24])
        if head_tilt != 0:
            t[:, :, :24, :] = torch.roll(t[:, :, :24, :], shifts=head_tilt, dims=3)

        return t

    def generate_animation(
        self,
        base_character: Image.Image,
        action: str = "run",
        num_frames: int = 4
    ) -> Tuple[List[Image.Image], Image.Image, bytes]:
        """
        Generates a sequence of animation frames for the specified action.
        Returns:
            - frames: List of 64x64 PIL RGBA images
            - sheet: Combined horizontal sprite sheet PIL Image
            - gif_bytes: Encoded animated GIF bytes
        """
        if base_character.mode != "RGBA":
            base_character = base_character.convert("RGBA")

        base_64 = base_character.resize((64, 64), Image.Resampling.NEAREST)
        arr = np.array(base_64, dtype=np.float32) / 255.0
        t_base = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device) # (1, 4, 64, 64)

        action_name = action.lower()
        transforms = ACTION_TRANSFORMS.get(action_name, ACTION_TRANSFORMS["run"])

        frames = []
        with torch.no_grad():
            z_latent = self.encoder(t_base)

            for i in range(num_frames):
                trans = transforms[i % len(transforms)]

                # Perturb latent slightly for motion dynamic while keeping character identity
                motion_noise = torch.randn_like(z_latent) * 0.03
                z_frame = z_latent + motion_noise

                cond = torch.zeros(1, 32, device=self.device)
                rec_tensor = self.generator(z_frame, cond)

                # Blend with transformed base to preserve 100% identity
                t_trans = self._apply_frame_transform(t_base, trans)
                blended_t = 0.75 * t_trans + 0.25 * rec_tensor

                blended_arr = (blended_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
                frame_raw = Image.fromarray(blended_arr, mode="RGBA")

                # Quantize each frame to signature palette
                frame_quant = quantize_to_pixel_art(frame_raw, size=(64, 64), palette=SIGNATURE_PALETTE)
                frames.append(frame_quant)

        # Create Sprite Sheet (horizontal grid)
        sheet = Image.new("RGBA", (64 * len(frames), 64), (0, 0, 0, 0))
        for idx, f in enumerate(frames):
            sheet.paste(f, (idx * 64, 0))

        # Create Animated GIF in memory
        gif_io = imageio.mimsave(
            imageio.RETURN_BYTES,
            [np.array(f) for f in frames],
            format="GIF",
            duration=0.15,
            loop=0
        )

        return frames, sheet, gif_io
