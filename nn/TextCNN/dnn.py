import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import torch.optim as optim

# 文件参数
TRAIN_CSV_FILE = '../../FirstData/train_data.csv'
PRED_CSV_FILE = '../../FirstData/test_data.csv'
TEXT_COLUMN = 'message'
LABEL_COLUMN = 'label'

# 超参数

MAX_LEN = 512 # 分词器的最大长度
BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 0.0005
DROPOUT_PROB = 0.25
MASK_PROB = 0.25 # 训练过程中随机覆盖

# 随机数种子
SEED = 1048596

# 模型超参数
EMBEDDING_DIM = 256
KERNEL_SIZES = [3, 4, 5, 6, 7]
NUM_FILTERS = 100

torch.manual_seed(SEED)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

print("加载数据")
train_df = pd.read_csv(TRAIN_CSV_FILE)
pred_df = pd.read_csv(PRED_CSV_FILE)
print(f"成功从 {TRAIN_CSV_FILE} 读取 {len(train_df)} 条训练数据。")

class MessageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_col, label_col, max_len, is_train=False, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data[text_col]
        self.labels = self.data[label_col]
        self.max_len = max_len
        self.is_train = is_train
        self.mask_prob = mask_prob

        if self.is_train and self.mask_prob > 0:
            assert tokenizer.mask_token_id is not None, \
                "This tokenizer does not have a mask token. Set mask_prob=0 or use a different tokenizer."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.text.iloc[idx])
        label = int(self.labels.iloc[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        if self.is_train and self.mask_prob > 0:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(),
                                                                      already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            maskable_tokens_mask = ~special_tokens_mask & attention_mask.bool()
            rand = torch.rand(input_ids.shape)
            masking_mask = (rand < self.mask_prob) & maskable_tokens_mask

            input_ids[masking_mask] = self.tokenizer.mask_token_id

        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }

# 创建训练数据集
train_dataset = MessageDataset(train_df, tokenizer, TEXT_COLUMN, LABEL_COLUMN, MAX_LEN, is_train=True, mask_prob=MASK_PROB)
pred_dataset = MessageDataset(pred_df, tokenizer, TEXT_COLUMN, LABEL_COLUMN, MAX_LEN, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型架构

class DokiPaperClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, dropout_prob):
        super(DokiPaperClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).permute(0, 2, 1)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        concatenated = torch.cat(pooled, dim=1)
        dropped = self.dropout(concatenated)
        return self.fc(dropped).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


vocab_size = tokenizer.vocab_size
model = DokiPaperClassifier(vocab_size, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZES, DROPOUT_PROB).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n--- 开始训练模型 ---")
for epoch in range(NUM_EPOCHS):

    model.train()
    total_train_loss = 0
    total_train_correct = 0 # 精度

    # DataLoader 生成的小批量(mini-batch)数据
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # 精度统计
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        total_train_correct += (preds == labels.int()).sum().item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_dataset)
    print(f'epoch: {epoch}  total_train_loss: {total_train_loss:.4f} and avg_train_loss: {avg_train_loss:.4f} '
          f'total_train_correct: {total_train_correct} train_accuracy: {train_accuracy:.4f}')


model.eval()
total_val_loss = 0
total_val_correct = 0

with torch.no_grad():  # 在此模式下不计算梯度，节省资源
    for batch in pred_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        total_val_loss += loss.item()

        # 计算准确率
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        total_val_correct += (preds == labels.int()).sum().item()

avg_val_loss = total_val_loss / len(pred_loader)
val_accuracy = total_val_correct / len(pred_dataset)

print(f'Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | '
      f'训练损失: {avg_train_loss:.4f} | '
      f'验证损失: {avg_val_loss:.4f} | '
      f'验证准确率: {val_accuracy:.4f}')






