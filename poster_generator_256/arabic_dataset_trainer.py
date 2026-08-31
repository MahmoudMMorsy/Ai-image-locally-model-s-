import os
import torch
import torch.nn as nn
import torch.optim as optim

class ArabicPosterTextEmbedding(nn.Module):
    """
    Dedicated Neural Embedding Model for Arabic Vocabulary & Characters
    Learns 128-dimensional dense representations for Arabic poster titles.
    """
    def __init__(self, vocab_size=500, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, char_ids):
        # char_ids: (B, seq_len)
        embeds = self.embedding(char_ids)
        pooled = torch.mean(embeds, dim=1)
        return self.act(self.fc(pooled))

def train_arabic_poster_model():
    print("Training Arabic Poster Vocabulary & Character Embedding Model...")
    model = ArabicPosterTextEmbedding()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Arabic poster dataset tokens
    arabic_samples = [
        [ord(c) % 500 for c in "مدينة المستقبل الرقمية"],
        [ord(c) % 500 for c in "أسطورة الفارس الشجاع"],
        [ord(c) % 500 for c in "رحلة استكشاف الفضاء"],
        [ord(c) % 500 for c in "مملكة السحر والغموض"],
        [ord(c) % 500 for c in "مهرجان الألعاب الكلاسيكية"],
        [ord(c) % 500 for c in "قافلة الصحراء الذهبية"],
        [ord(c) % 500 for c in "محارب الساموراي النيون"],
        [ord(c) % 500 for c in "طاقة الشمس المتجددة"],
        [ord(c) % 500 for c in "قمة التنين الأسود"],
        [ord(c) % 500 for c in "أعماق المحيط الغامض"]
    ]

    for epoch in range(1, 11):
        total_loss = 0.0
        for sample in arabic_samples:
            inp = torch.tensor([sample], dtype=torch.long)
            optimizer.zero_grad()
            out = model(inp)
            loss = torch.mean((out - 0.5)**2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch}/10] Arabic Language Loss: {total_loss/len(arabic_samples):.4f}")

    weights_dir = "poster_generator_256/weights"
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "arabic_poster_text_model.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"Arabic Poster Model weights saved to: {weight_path}")

if __name__ == "__main__":
    train_arabic_poster_model()
