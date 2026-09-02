import torch
from transformers import AutoFeatureExtractor, AutoModel


MODEL_NAME = "facebook/wav2vec2-base"


class SpeechEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self, waveform, sample_rate=16000):
        """
        Convert raw speech waveform into pretrained
        speech representations.
        """

        audio = waveform.cpu().numpy()

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state

        # Temporal mean pooling
        embedding = hidden_states.mean(dim=1)

        return embedding.squeeze(0).cpu()


if __name__ == "__main__":
    print("Speech embedding module ready.")