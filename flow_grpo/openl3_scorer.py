import requests
import torch


class OpenL3Scorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
    @torch.no_grad()
    def __call__(self, predictions, references):
        for i in range(len(predictions)):
            predictions[i] = predictions[i].tolist()
            references[i] = references[i].tolist()
        response = requests.post(
            "http://localhost:8000/openl3_reward",
            json={
            "predictions": predictions,
            "references": references,
            },
        )
        rewards = response.json()["rewards"]
        return rewards


# Usage example
def main():
    preds, refs = [torch.randn(48000), torch.randn(48000)], [torch.randn(48000), torch.randn(48000)]
    
    scorer = OpenL3Scorer(
        device="cuda",
        dtype=torch.float32
    )
    rewards = scorer(preds, refs)
    print(rewards)
    

    

if __name__ == "__main__":
    main()