import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# ファインチューニングするデータのサンプル
# 以下は仮のデータです。実際のデータに置き換えてください。
training_data = ["This is a sample sentence.", "Another example sentence."]

# トークナイザーとモデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# ファインチューニング用のデータセットの作成
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze()}

# データセットとデータローダーの作成
train_dataset = CustomDataset(training_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# ファインチューニングのための設定
optimizer = AdamW(model.parameters(), lr=5e-5)

# ファインチューニングのエポック数
num_epochs = 3

# ファインチューニングのメインループ
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

# ファインチューニング後のモデルを保存
model.save_pretrained("fine_tuned_gpt2_model")
tokenizer.save_pretrained("fine_tuned_gpt2_model")





# predictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前学習済みのGPT-2モデルとトークナイザーの読み込み
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 入力テキスト
input_text = "Your input text goes here."

# テキストをトークン化してモデルに入力する
input_ids = tokenizer.encode(input_text, return_tensors='pt')

input_ids = input_ids.to(device)

# モデルの出力を生成する
with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

# 生成されたトークンをデコードしてテキストに変換
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 生成されたテキストを出力
print("Generated Text:", generated_text)
