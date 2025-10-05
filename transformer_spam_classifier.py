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
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
        
        # 初始化权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
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

class TransformerSpamTrainer:
    """Transformer垃圾短信分类训练器"""
    def __init__(self, model_name='bert-base-multilingual-cased', max_length=128, batch_size=8):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 初始化分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 初始化模型
        self.model = TransformerSpamClassifier(model_name)
        
        # 数据增强器
        self.augmenter = DataAugmentation(self.tokenizer)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def load_data(self, spam_file, ham_file):
        """加载数据"""
        print("正在加载数据...")
        
        # 读取数据
        spam_data = pd.read_csv(spam_file)
        ham_data = pd.read_csv(ham_file)
        
        # 合并数据
        self.data = pd.concat([spam_data, ham_data], ignore_index=True)
        
        print(f"总数据量: {len(self.data)}")
        print(f"垃圾短信: {len(spam_data)}")
        print(f"正常短信: {len(ham_data)}")
        
        return self.data
    
    def prepare_data(self, test_size=0.25, augment_train=True):
        """准备训练和测试数据"""
        print("正在准备数据...")
        
        # 分割数据
        X = self.data['message'].values
        y = self.data['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 数据增强
        if augment_train:
            print("正在进行数据增强...")
            augmented_X_train = []
            augmented_y_train = []
            
            for text, label in zip(X_train, y_train):
                # 原始文本
                augmented_X_train.append(text)
                augmented_y_train.append(label)
                
                # 增强文本
                augmented_texts = self.augmenter.augment_text(text, num_augmentations=2)
                for aug_text in augmented_texts[1:]:  # 跳过原始文本
                    augmented_X_train.append(aug_text)
                    augmented_y_train.append(label)
            
            X_train = np.array(augmented_X_train)
            y_train = np.array(augmented_y_train)
            
            print(f"增强后训练集大小: {len(X_train)}")
        
        # 创建数据集
        self.train_dataset = SMSDataset(X_train, y_train, self.tokenizer, self.max_length)
        self.test_dataset = SMSDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return self.train_loader, self.test_loader
    
    def train_epoch(self, optimizer, criterion, device):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, criterion, device):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs=10, learning_rate=2e-5, weight_decay=0.01):
        """训练模型"""
        print("开始训练模型...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        self.model.to(device)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(optimizer, criterion, device)
            
            # 验证
            val_loss, val_acc = self.validate(criterion, device)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
                print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def evaluate(self):
        """评估模型"""
        print("\n正在评估模型...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))
        self.model.to(device)
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        
        print(f"测试准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=['正常短信', '垃圾短信']))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常短信', '垃圾短信'],
                    yticklabels=['正常短信', '垃圾短信'])
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, all_predictions, all_labels
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率', color='blue')
        ax2.plot(self.val_accuracies, label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('transformer_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single(self, text):
        """预测单条短信"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))
        self.model.to(device)
        self.model.eval()
        
        # 预处理文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
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

def main():
    """主函数"""
    print("基于Transformer的垃圾短信分类器")
    print("=" * 50)
    
    # 创建训练器
    trainer = TransformerSpamTrainer(
        model_name='bert-base-multilingual-cased',
        max_length=128,
        batch_size=8
    )
    
    # 加载数据
    trainer.load_data('spam_data.csv', 'ham_data.csv')
    
    # 准备数据
    trainer.prepare_data(test_size=0.25, augment_train=True)
    
    # 训练模型
    best_acc = trainer.train(epochs=15, learning_rate=2e-5)
    
    # 评估模型
    test_acc, predictions, labels = trainer.evaluate()
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 测试单条短信预测
    print("\n=== 单条短信预测测试 ===")
    test_texts = [
        "您的验证码是123456，请勿泄露给他人",
        "恭喜您中奖了！请点击链接领取奖品",
        "明天一起去吃饭吧",
        "免费领取价值1000元礼品，限时优惠！"
    ]
    
    for text in test_texts:
        result = trainer.predict_single(text)
        print(f"\n短信: {result['text']}")
        print(f"预测结果: {result['class_name']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"概率分布: {result['probabilities']}")
    
    print(f"\n训练完成！")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"测试准确率: {test_acc:.4f}")

if __name__ == "__main__":
    main()
