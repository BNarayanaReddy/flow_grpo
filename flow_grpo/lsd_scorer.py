import torch
from wandb import Image

SR_TO_CUTOFF_BIN = {8: 80, 12: 128, 16: 170, 24: 256}


class LSD_Scorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32, sample_rate=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
    
    
    def stft_magnitude(self, audio, n_fft=1024, hop_length=256):
        """Compute STFT magnitude spectrum."""
        window = torch.hann_window(n_fft).to(audio.device)
        return torch.abs(torch.stft(audio, n_fft, hop_length, window=window, return_complex=True))
    
    def _lsd(self, a, b):
        return (a - b).square().mean(dim=1).sqrt().mean()
    
    @torch.no_grad()
    def __call__(self, predictions, references, sample_rate=24000):
        rewards = []
        for pred, target in zip(predictions, references):
            sample_ratio = sample_rate//1000
            if sample_ratio not in SR_TO_CUTOFF_BIN:
                raise ValueError(f"Unsupported sample rate {sample_rate}. Supported rates: {list(SR_TO_CUTOFF_BIN.keys())} kHz.")
            cutoff_bin = SR_TO_CUTOFF_BIN.get(sample_rate//1000, 256)
            sp = torch.log10(self.stft_magnitude(pred).square().clamp(min=1e-6))
            st = torch.log10(self.stft_magnitude(target).square().clamp(min=1e-6))
            lsd_hf = self._lsd(sp[..., cutoff_bin:, :], st[..., cutoff_bin:, :])
            rewards.append(torch.exp(-lsd_hf).item())
        return rewards

# Usage example
def main():
    scorer = LSD_Scorer(
        device="cuda",
        dtype=torch.float32,
    )

    preds, refs = [torch.randn(48000), torch.randn(48000)], [torch.randn(48000), torch.randn(48000)]
    rewards = scorer(preds, refs, sample_rate=24000)
    print(rewards)
    


if __name__ == "__main__":
    main()