# process_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os



ham_file = '../../FirstData/ham_data.csv'
spam_file = '../../FirstData/spam_data.csv'


train_file = '../../FirstData/train_data.csv'
test_file = '../../FirstData/test_data.csv'


TEXT_COLUMN = 'message'
LABEL_COLUMN = 'label'

# 定义测试集所占的比例
TEST_SIZE = 0.25

RANDOM_STATE = 1048596


try:

    ham_df = pd.read_csv(ham_file)
    spam_df = pd.read_csv(spam_file)

    print(f"成功读取 {len(ham_df)} 条正常短信。")
    print(f"成功读取 {len(spam_df)} 条垃圾短信。")


    ham_df[LABEL_COLUMN] = 0
    spam_df[LABEL_COLUMN] = 1

    full_df = pd.concat([ham_df, spam_df], ignore_index=True)

    print(f"合并后的总数据量: {len(full_df)} 条。")


    if TEXT_COLUMN not in full_df.columns:
        raise KeyError(f"错误：在合并后的数据中找不到指定的文本列 '{TEXT_COLUMN}'。")

    full_df = full_df[[TEXT_COLUMN, LABEL_COLUMN]]

    X = full_df[TEXT_COLUMN]
    y = full_df[LABEL_COLUMN]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # 关键参数：保持标签比例
    )


    train_df = pd.DataFrame({TEXT_COLUMN: X_train, LABEL_COLUMN: y_train})
    test_df = pd.DataFrame({TEXT_COLUMN: X_test, LABEL_COLUMN: y_test})

    print("\n数据分割完成:")
    print(f" - 训练集大小: {len(train_df)} 条")
    print(f" - 测试集大小: {len(test_df)} 条")

    # 检查训练集和测试集中的标签分布
    print(f"\n训练集标签分布:\n{train_df[LABEL_COLUMN].value_counts(normalize=True)}")
    print(f"\n测试集标签分布:\n{test_df[LABEL_COLUMN].value_counts(normalize=True)}")

    # index=False 表示不将DataFrame的索引写入到文件中
    train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig')

    print(f"\n成功将训练数据保存到: {train_file}")
    print(f"成功将测试数据保存到: {test_file}")

except Exception as e:
    print(f"发生了一个错误: {e}")