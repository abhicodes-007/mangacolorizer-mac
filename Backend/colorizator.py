import torch
import numpy as np
from torchvision.transforms import ToTensor
import coremltools as ct

from utils.utils import resize_pad, tile_process


class ColorizationStrategy:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None

    def load_weights(self, path):
        raise NotImplementedError

    def process_image(self, image, size):
        raise NotImplementedError


class AlacGANStrategy(ColorizationStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.params = config.alacgan
        self.load_weights(config.colorizer_path)

    def load_weights(self, path):
        try:
            self.model = ct.models.CompiledMLModel(path)
            print(f"[+] Loaded Core ML AlacGAN from {path}")
        except Exception as e:
            print(f"[-] Failed to load Core ML AlacGAN: {e}")
            self.model = None

    def process_image(self, image, size):
        target_size = size if size > 0 else self.params.image_size
        if target_size % 32 != 0: target_size = (target_size // 32) * 32

        processed_img, pad = resize_pad(image, target_size)
        
        # Prepare input for Core ML
        # CoreML model expects 'gray' input of shape (1, 1, H, W)
        img_tensor = ToTensor()(processed_img).unsqueeze(0).numpy().astype(np.float32)
        
        if self.model is None:
            print("[-] Core ML model not loaded.")
            return image
        
        try:
            out = self.model.predict({'gray': img_tensor})
            fake_color = out['rgb']  # shape (1, 3, H, W)
            
            result = fake_color[0].transpose(1, 2, 0)
            if pad[0] != 0: result = result[:-pad[0]]
            if pad[1] != 0: result = result[:, :-pad[1]]
            
            return (result * 255.0).clip(0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[-] Core ML prediction failed: {e}")
            return image



class MangaColorizator:
    def __init__(self, config):
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("[-] CUDA not available, falling back to CPU.")
            config.device = 'cpu'
        elif config.device == 'mps' and not torch.backends.mps.is_available():
            print("[-] MPS not available, falling back to CPU.")
            config.device = 'cpu'

        self.config = config
        self.current_image = None
        self.current_size = 576

        if config.colorizer_type == 'AlacGAN':
            self.strategy = AlacGANStrategy(config)
        else:
            raise Exception('Invalid colorizer type')

    def set_image(self, image, size=0):
        self.current_image = image
        self.current_size = size

    def colorize(self):
        if self.current_image is None:
            raise RuntimeError("Image not set. Call set_image() first.")

        return self.strategy.process_image(self.current_image, self.current_size)