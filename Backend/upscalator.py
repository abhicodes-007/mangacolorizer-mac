import numpy as np
import torch

from networks.RRDBNet import Upscaler as ESRGANNet
from utils.utils import tile_process


class UpscalingStrategy:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None

    def upscale(self, image, factor):
        raise NotImplementedError


class ESRGANStrategy(UpscalingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.model = ESRGANNet().to(self.device)
        self.load_weights(config.upscaler_path)
        self.model.eval()
        self.params = config.esrgan
        # Use FP16 on GPU for ~2x speedup (RRDBNet has no BatchNorm, so FP16 is safe)
        self.use_half = config.device in ('mps', 'cuda')
        if self.use_half:
            self.model = self.model.half()
            print(f"[+] ESRGAN: Using FP16 (half precision) on {config.device}")

    def load_weights(self, path):
        try:
            model_or_chkpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.generator = model_or_chkpt
            print(f"[+] Loaded ESRGAN weights  from {path}")
        except Exception as e:
            print(f"[-] Failed to load ESRGAN weights: {e}")

    def upscale(self, image, factor):
        img_tensor = torch.from_numpy(image).to(self.device)
        result = img_tensor.permute(2, 0, 1).unsqueeze(0).float()
        if self.use_half:
            result = result.half()

        with torch.inference_mode():
            if self.params.tile_size > 0:
                result = tile_process(
                    self.model,
                    result,
                    factor,
                    self.params.tile_size,
                    self.params.tile_pad
                )
            else:
                result = self.model(result)

        result = result.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        result = np.transpose(result[[2, 1, 0], :, :], (1, 2, 0))
        result = (result * 255.0).round().astype(np.uint8)
        result = result[:, :, ::-1]

        return result



class MangaUpscaler:
    def __init__(self, config):
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("[-] CUDA not available, falling back to CPU.")
            config.device = 'cpu'
        elif config.device == 'mps' and not torch.backends.mps.is_available():
            print("[-] MPS not available, falling back to CPU.")
            config.device = 'cpu'

        if config.upscaler_type == 'ESRGAN':
            self.strategy = ESRGANStrategy(config)
        else:
            raise Exception('Invalid upscaler type')

    def upscale(self, image, factor):
        if image.shape[2] == 4:
            image = image[:, :, :3]

        return self.strategy.upscale(image, factor)