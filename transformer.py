import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, num_features, output_dim, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.EmbeddingBag(input_dim, num_features, sparse=True)
        self.positional_encoding = PositionalEncoding(num_features)
        self.transformer = nn.Transformer(d_model=num_features, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers, dropout=dropout)
        self.fc = nn.Linear(num_features, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # Change to (seq_len, batch_size, num_features)
        src = self.positional_encoding(src)
        output = self.transformer(src)
        output = output.mean(dim=0)  # Aggregate over the sequence length
        output = self.fc(output)
        return output.squeeze()

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def evaluate_predictions(predictions, targets):
    _, predicted_indices = predictions.max(dim=1)
    correct_predictions = predicted_indices == targets
    accuracy = correct_predictions.float().mean().item()
    return accuracy

# 仮想の時系列データを生成
num_samples = 100
seq_len = 10
data = torch.randn(num_samples, seq_len, 3)  # 3カラムのデータ (timestamp, item_id, purchase_amount)

# 1週間後または1ヶ月後の勾配を予測するためのターゲットデータ生成
target_seq_len = 7  # 1週間後の場合は7、1ヶ月後の場合は30など
target = torch.randint(0, 5, (num_samples,))  # ターゲットデータ

# データセットとデータローダーを作成
dataset = TimeSeriesDataset(data, target)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルの初期化と損失関数、オプティマイザの定義
model = TransformerModel(input_dim=100, nhead=2, num_encoder_layers=2, num_features=64, output_dim=target_seq_len)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
epochs = 10
for epoch in range(epochs):
    for data_batch, target_batch in dataloader:
        optimizer.zero_grad()
        output = model(data_batch)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

    # エポックごとに評価
    with torch.no_grad():
        total_accuracy = 0.0
        num_batches = 0
        for data_batch, target_batch in dataloader:
            output = model(data_batch)
            accuracy = evaluate_predictions(output, target_batch)
            total_accuracy += accuracy
            num_batches += 1

        average_accuracy = total_accuracy / num_batches
        print(f'Epoch {epoch + 1}, Accuracy: {average_accuracy * 100:.2f}%')
