# 基于Transformer的垃圾短信分类系统

## 项目概述

本项目实现了一个基于Transformer架构的垃圾短信分类系统，使用BERT预训练模型进行文本分类。相比传统的CNN方法，Transformer模型能够更好地理解文本的语义信息，在垃圾短信检测任务上取得了优异的性能。

## 主要特性

- 🚀 **GPU加速训练**: 支持CUDA加速，充分利用RTX 4070等GPU资源
- 🔧 **混合精度训练**: 使用AMP技术提升训练效率，减少显存占用
- 📈 **数据增强**: 通过随机掩盖策略增强训练数据，提升模型泛化能力
- 🎯 **高准确率**: 在测试集上达到93.62%的分类准确率
- 📊 **可视化分析**: 提供训练过程可视化和混淆矩阵分析
- 🔍 **关键词提取**: 自动提取短信关键词，便于分析

## 技术架构

### 模型架构
- **基础模型**: BERT-base-multilingual-cased
- **分类头**: 线性层 + Dropout
- **优化策略**: 冻结BERT前6层，只训练后6层和分类头
- **损失函数**: CrossEntropyLoss
- **优化器**: AdamW (学习率: 2e-5)

### 数据处理
- **分词器**: BERT多语言分词器
- **最大长度**: 64 tokens
- **批处理大小**: 16 (GPU优化)
- **数据增强**: 15%随机掩盖率

## 性能对比

| 模型类型 | 准确率 | 训练时间 | 特点 |
|---------|--------|----------|------|
| CNN模型 | 91.5% | 较长 | 提取局部特征 |
| **Transformer模型** | **93.62%** | **较短** | **理解语义信息** |

## 环境要求

### 硬件要求
- GPU: NVIDIA RTX 4070 或更高 (推荐)
- 显存: 8GB+ (推荐)
- 内存: 16GB+ (推荐)

### 软件要求
- Python 3.8+
- CUDA 12.1+
- PyTorch 2.5.1+cu121

## 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd transformer-spam-classifier
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证GPU支持
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 使用方法

### 1. 数据准备
确保您的数据文件格式如下：
- `spam_data.csv`: 垃圾短信数据 (label=1)
- `ham_data.csv`: 正常短信数据 (label=0)

CSV文件应包含以下列：
- `id`: 短信ID
- `label`: 标签 (0=正常, 1=垃圾)
- `message`: 短信内容

### 2. 训练模型
```bash
python transformer_trainer.py
```

### 3. 关键词提取
```bash
python keyword_extractor.py
```

### 4. 数据分析
```bash
python data_analysis.py
```

## 训练过程

### 数据分割
- 训练集: 75% (141条原始数据 → 423条增强数据)
- 测试集: 25% (47条数据)

### 训练配置
- **Epochs**: 30
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Dropout Rate**: 0.3

### 训练监控
训练过程中会显示：
- 训练/验证损失
- 训练/验证准确率
- 最佳模型保存

## 模型性能

### 最终结果
- **测试准确率**: 93.62%
- **精确率**: 94% (正常短信), 92% (垃圾短信)
- **召回率**: 91% (正常短信), 96% (垃圾短信)
- **F1分数**: 93% (正常短信), 94% (垃圾短信)

### 训练历史
- 最佳验证准确率: 93.62% (第23轮)
- 训练收敛: 约15轮后稳定
- 过拟合控制: 通过数据增强和Dropout有效控制

## 文件结构

```
project/
├── transformer_trainer.py      # 主训练脚本
├── transformer_spam_classifier.py  # 完整训练器类
├── keyword_extractor.py       # 关键词提取工具
├── data_analysis.py           # 数据分析脚本
├── requirements.txt           # 依赖包列表
├── spam_data.csv             # 垃圾短信数据
├── ham_data.csv              # 正常短信数据
├── ham_data_with_keywords.csv # 带关键词的数据
├── best_transformer_model.pth # 训练好的模型
├── transformer_training_history.png # 训练历史图
├── transformer_confusion_matrix.png # 混淆矩阵图
└── README.md                 # 项目说明文档
```

## 核心代码示例

### 模型定义
```python
class TransformerSpamClassifier(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased', num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)
```

### GPU优化训练
```python
# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 数据增强策略

### 随机掩盖增强
- **掩盖概率**: 15%
- **增强倍数**: 每个原始样本生成2个增强样本
- **效果**: 提升模型泛化能力，防止过拟合

### 关键词过滤
- 过滤停用词、打码词(xx, xxx等)
- 过滤验证码相关词汇
- 保留有意义的语义词汇

## 模型优势

### 相比CNN的优势
1. **语义理解**: Transformer能更好地理解文本的语义信息
2. **长距离依赖**: 注意力机制能捕获长距离的词汇关系
3. **预训练优势**: 利用BERT的预训练知识
4. **端到端**: 无需手工设计特征

### 技术优势
1. **GPU加速**: 充分利用GPU并行计算能力
2. **混合精度**: 减少显存占用，提升训练速度
3. **数据增强**: 有效提升小数据集的性能
4. **可解释性**: 提供预测概率和置信度

## 应用场景

- **短信过滤**: 自动识别和过滤垃圾短信
- **内容审核**: 社交媒体内容审核
- **邮件分类**: 垃圾邮件检测
- **文本分类**: 其他二分类文本任务

## 未来改进方向

1. **模型优化**: 尝试更大的预训练模型(RoBERTa, DeBERTa)
2. **数据增强**: 引入更多数据增强技术(回译、同义词替换)
3. **集成学习**: 结合多个模型提升性能
4. **在线学习**: 支持增量学习和模型更新
5. **多语言支持**: 扩展到其他语言

## 常见问题

### Q: 为什么选择BERT而不是其他模型？
A: BERT在中文文本理解上表现优异，且bert-base-multilingual-cased对中文支持良好，适合我们的中文短信数据。

### Q: 如何调整模型参数？
A: 可以通过修改`transformer_trainer.py`中的超参数来调整：
- `learning_rate`: 学习率
- `batch_size`: 批处理大小
- `epochs`: 训练轮数
- `max_length`: 最大文本长度

### Q: 如何处理新的数据？
A: 将新数据按照相同格式添加到CSV文件中，重新训练模型即可。

### Q: 模型文件太大怎么办？
A: 可以使用模型压缩技术，如知识蒸馏或量化，减少模型大小。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至 [your-email@example.com]

---

**注意**: 本项目仅用于学术研究和学习目的，请遵守相关法律法规，不得用于非法用途。
