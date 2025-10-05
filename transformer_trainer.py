import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SMSDataset(Dataset):
    """短信数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用BERT分词器进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerSpamClassifier(nn.Module):
    """基于Transformer的垃圾短信分类器"""
    def __init__(self, model_name='bert-base-multilingual-cased', num_classes=2, dropout_rate=0.3):
        super(TransformerSpamClassifier, self).__init__()
        
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类头
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 冻结BERT的前几层（可选）
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(6):  # 冻结前6层
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]标记的表示进行分类
        pooled_output = outputs.pooler_output
        
        # Dropout和分类
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits

class DataAugmentation:
    """数据增强类"""
    def __init__(self, tokenizer, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
    
    def random_mask(self, text):
        """随机掩盖文本中的token"""
        tokens = self.tokenizer.tokenize(text)
        
        # 随机选择要掩盖的token（排除特殊token）
        mask_indices = []
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and random.random() < self.mask_prob:
                mask_indices.append(i)
        
        # 执行掩盖
        masked_tokens = tokens.copy()
        for idx in mask_indices:
            masked_tokens[idx] = '[MASK]'
        
        return self.tokenizer.convert_tokens_to_string(masked_tokens)
    
    def augment_text(self, text, num_augmentations=1):
        """对文本进行数据增强"""
        augmented_texts = [text]
        
        for _ in range(num_augmentations):
            augmented_text = self.random_mask(text)
            augmented_texts.append(augmented_text)
        
        return augmented_texts

def train_transformer_model():
    """训练Transformer模型"""
    print("基于Transformer的垃圾短信分类器")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载数据
    print("正在加载数据...")
    spam_data = pd.read_csv('spam_data.csv')
    ham_data = pd.read_csv('ham_data.csv')
    
    # 合并数据
    data = pd.concat([spam_data, ham_data], ignore_index=True)
    
    print(f"总数据量: {len(data)}")
    print(f"垃圾短信: {len(spam_data)}")
    print(f"正常短信: {len(ham_data)}")
    
    # 分割数据 (75% 训练, 25% 测试)
    X = data['message'].values
    y = data['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # 数据增强
    print("正在进行数据增强...")
    augmenter = DataAugmentation(tokenizer, mask_prob=0.15)
    
    augmented_X_train = []
    augmented_y_train = []
    
    for text, label in zip(X_train, y_train):
        # 原始文本
        augmented_X_train.append(text)
        augmented_y_train.append(label)
        
        # 增强文本 (每个原始文本生成2个增强版本)
        augmented_texts = augmenter.augment_text(text, num_augmentations=2)
        for aug_text in augmented_texts[1:]:  # 跳过原始文本
            augmented_X_train.append(aug_text)
            augmented_y_train.append(label)
    
    X_train_aug = np.array(augmented_X_train)
    y_train_aug = np.array(augmented_y_train)
    
    print(f"增强后训练集大小: {len(X_train_aug)}")
    
    # 创建数据集
    train_dataset = SMSDataset(X_train_aug, y_train_aug, tokenizer, max_length=64)
    test_dataset = SMSDataset(X_test, y_test, tokenizer, max_length=64)
    
    # 创建数据加载器 (GPU优化：增大batch size)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
    
    model = TransformerSpamClassifier()
    model.to(device)
    
    # 优化器和损失函数 (GPU优化)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 使用混合精度训练 (GPU优化)
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0
    epochs = 30  # GPU训练可以增加epoch数
    
    print("开始训练模型...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # 训练
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # 验证
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = total_loss / len(test_loader)
        val_acc = 100 * correct / total
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_transformer_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    # 最终评估
    print("\n正在评估模型...")
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算最终准确率
    final_accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"最终测试准确率: {final_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['正常短信', '垃圾短信']))
    
    # 绘制训练历史
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', color='blue')
    ax2.plot(val_accuracies, label='验证准确率', color='red')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('transformer_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常短信', '垃圾短信'],
                yticklabels=['正常短信', '垃圾短信'])
    plt.title('Transformer模型混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, tokenizer, final_accuracy

def predict_single_sms(model, tokenizer, text, device):
    """预测单条短信"""
    model.eval()
    
    # 预处理文本
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_name = '垃圾短信' if predicted_class == 1 else '正常短信'
    
    return {
        'text': text,
        'predicted_class': predicted_class,
        'class_name': class_name,
        'confidence': confidence,
        'probabilities': {
            '正常短信': probabilities[0][0].item(),
            '垃圾短信': probabilities[0][1].item()
        }
    }

if __name__ == "__main__":
    # 训练模型
    model, tokenizer, accuracy = train_transformer_model()
    
    # 测试单条短信预测
    print("\n=== 单条短信预测测试 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_texts = [
        "您的验证码是123456，请勿泄露给他人",
        "恭喜您中奖了！请点击链接领取奖品",
        "明天一起去吃饭吧",
        "免费领取价值1000元礼品，限时优惠！",
        "【xx】您的验证码是：123456，10分钟内有效",
        "亲爱的用户，您的账户异常，请立即处理"
    ]
    
    for text in test_texts:
        result = predict_single_sms(model, tokenizer, text, device)
        print(f"\n短信: {result['text']}")
        print(f"预测结果: {result['class_name']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"概率分布: {result['probabilities']}")
    
    print(f"\n训练完成！最终测试准确率: {accuracy:.4f}")
