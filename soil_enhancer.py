import os
import io
import sys
import types
import torch
import numpy as np
from PIL import Image
import urllib.request
from pathlib import Path

# Some torchvision versions removed `functional_tensor`, but BasicsSR still imports it.
# Provide a tiny compatibility shim so the official Real-ESRGAN package can import cleanly.
try:
    import torchvision.transforms.functional as tv_functional

    functional_tensor_mod = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor_mod.rgb_to_grayscale = tv_functional.rgb_to_grayscale
    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional_tensor_mod)
except Exception:
    pass

# Try to import RealESRGANer, we will wrap it in a class to prevent crashing if not installed
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REAL_ESRGAN_AVAILABLE = True
except ImportError:
    REAL_ESRGAN_AVAILABLE = False


class SoilImageEnhancer:
    def __init__(self, models_dir="Models"):
        self.upsampler = None
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.weight_path = self.models_dir / "RealESRGAN_x4plus.pth"
        self.weight_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    def _ensure_weights(self):
        if not self.weight_path.exists():
            print(f"Downloading RealESRGAN weights to {self.weight_path}...")
            urllib.request.urlretrieve(self.weight_url, str(self.weight_path))
            print("Download complete.")
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Missing RealESRGAN weights at {self.weight_path}")

    def _init_model(self):
        if not REAL_ESRGAN_AVAILABLE:
            raise RuntimeError(
                "RealESRGAN dependencies could not be imported. "
                "Check torchvision/basicsr compatibility."
            )
        
        self._ensure_weights()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # RealESRGAN_x4plus architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(self.weight_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device
        )

    def enhance_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Enhances the input PIL Image using Real-ESRGAN.
        Returns a new upscaled and sharpened PIL Image.
        """
        if self.upsampler is None:
            self._init_model()

        if self.upsampler is None:
            raise RuntimeError("RealESRGAN upsampler could not be initialized.")

        # Convert PIL to CV2 BGR format (NumPy array)
        img_np = np.array(pil_image)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = img_np[:, :, ::-1] # RGB to BGR
        else:
            img_bgr = img_np

        try:
            with torch.no_grad():
                output_bgr, _ = self.upsampler.enhance(img_bgr, outscale=4)
            # Convert back to RGB PIL Image
            output_rgb = output_bgr[:, :, ::-1]
            return Image.fromarray(output_rgb)
        except Exception as e:
            raise RuntimeError(f"Real-ESRGAN enhancement failed: {e}") from e
