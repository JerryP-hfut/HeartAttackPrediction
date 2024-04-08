import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class HeartAttackDataset(Dataset):
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)
        self.features = self.data.iloc[:, 1:-1]  # 提取特征列（不包括第一列和最后一列）
        self.labels = self.data.iloc[:, -1].values.astype(int)  # 提取标签列并转换为整数类型
        self.encode_categorical_features()
        self.standardize_features()

    def encode_categorical_features(self):
        encoder = LabelEncoder()
        for column in self.features.select_dtypes(include=['object']).columns:
            self.features[column] = encoder.fit_transform(self.features[column])
    def standardize_features(self):
        scaler = StandardScaler()
        self.features = pd.DataFrame(scaler.fit_transform(self.features), columns=self.features.columns)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features.iloc[idx].values.astype(np.float32))  # 或者使用 dtype=torch.float
        label = torch.tensor(self.labels[idx].astype(np.float32)).unsqueeze(0)  # 或者使用 dtype=torch.float
        return feature, label
    
