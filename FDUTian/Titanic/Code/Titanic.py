from typing import List
import numpy as np
import random
import math
import torch
from pathlib import Path
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class Titanic(nn.Module):
    def __init__(self, embedding_dim, linear_dim_1, linear_dim_2):
        super(Titanic, self).__init__()
        self.SexEmbedding = nn.Embedding(2, embedding_dim)
        self.PclassEmbedding = nn.Embedding(4, embedding_dim)
        self.AgeEmbedding = nn.Embedding(81, embedding_dim)
        self.SibSpEmbedding = nn.Embedding(9, embedding_dim)
        self.ParchEmbedding = nn.Embedding(10, embedding_dim)
        self.EmbarkedEmbedding = nn.Embedding(4, embedding_dim)
        # add fare to factor
        self.FareEmbedding = nn.Embedding(53, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, linear_dim_1)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(linear_dim_1, linear_dim_2)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(linear_dim_2, 2)
        self.ac3 = nn.ReLU()

    def forward(self, sex, pclass, age, sibsp, parch, embarked, fare):
        Sex = self.SexEmbedding(sex)
        Pclass = self.PclassEmbedding(pclass)
        Age = self.AgeEmbedding(age)
        Sibsp = self.SibSpEmbedding(sibsp)
        Parch = self.ParchEmbedding(parch)
        Embarked = self.EmbarkedEmbedding(embarked)
        # add fare to factor
        Fare = self.FareEmbedding(fare)
        vector = Sex + Pclass + Age + Sibsp + Parch + Embarked + Fare
        x = self.fc1(vector)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        return x


data = pd.read_csv(Path("E://Kaggle/Titanic/Initial Data/train.csv"))
data_row = data.values.tolist()
random.shuffle(data_row)
train_data = data_row[:800]
valid_data = data_row[800:]
test_data = pd.read_csv(Path("E://Kaggle/Titanic/Initial Data/test.csv")).values.tolist()


class DataGeneretor(Dataset):
    def __init__(self, data):
        # add fare to factor
        sex, pclass, age, sibsp, parch, embarked, fare, survive = list(), list(), list(), list(), list(), list(), list(), list()
        for each_row in data:
            survive.append([int(each_row[1])])
            pclass.append([int(each_row[2])])
            if each_row[4] == "male":
                sex.append([1])
            elif each_row[4] == "female":
                sex.append([0])
            if math.isnan(each_row[5]):
                age.append([0])
            elif not math.isnan(each_row[5]):
                age.append([math.ceil(each_row[5])])
            sibsp.append([int(each_row[6])])
            parch.append([int(each_row[7])])
            # add fare to factor
            if math.isnan(each_row[9]):
                fare.append([52])
            elif not math.isnan(each_row[9]):
                fare.append([int(each_row[9] // 10)])
            if each_row[-1] == "S":
                embarked.append([1])
            elif each_row[-1] == "C":
                embarked.append([2])
            elif each_row[-1] == "Q":
                embarked.append([3])
            else:
                embarked.append([0])
            self.sex = torch.tensor(sex, device="cuda")
            self.pclass = torch.tensor(pclass, device="cuda")
            self.age = torch.tensor(age, device="cuda")
            self.sibsp = torch.tensor(sibsp, device="cuda")
            self.parch = torch.tensor(parch, device="cuda")
            self.embarked = torch.tensor(embarked, device="cuda")
            self.survive = torch.tensor(survive, device="cuda")
            # add fare to factor
            self.fare = torch.tensor(fare, device="cuda")
            
 
    def __getitem__(self, i):
        return (self.sex[i], self.pclass[i], self.age[i], self.sibsp[i], self.parch[i], self.embarked[i], self.fare[i], self.survive[i])

    def __len__(self):
        return len(self.sex)


class TestDataGeneretor(Dataset):
    def __init__(self, data):
        # add fare to factor
        sex, pclass, age, sibsp, parch, embarked, fare = list(), list(), list(), list(), list(), list(), list()
        for each_row in data:
            pclass.append([int(each_row[1])])
            if each_row[3] == "male":
                sex.append([1])
            elif each_row[3] == "female":
                sex.append([0])
            if math.isnan(each_row[4]):
                age.append([0])
            elif not math.isnan(each_row[4]):
                age.append([math.ceil(each_row[4])])
            sibsp.append([int(each_row[5])])
            parch.append([int(each_row[6])])
            # add fare to factor
            if math.isnan(each_row[8]):
                fare.append([52])
            elif not math.isnan(each_row[8]):
                fare.append([int(each_row[8] // 10)])
            if each_row[-1] == "S":
                embarked.append([1])
            elif each_row[-1] == "C":
                embarked.append([2])
            elif each_row[-1] == "Q":
                embarked.append([3])
            else:
                embarked.append([0])
            self.sex = torch.tensor(sex, device="cuda")
            self.pclass = torch.tensor(pclass, device="cuda")
            self.age = torch.tensor(age, device="cuda")
            self.sibsp = torch.tensor(sibsp, device="cuda")
            self.parch = torch.tensor(parch, device="cuda")
            self.embarked = torch.tensor(embarked, device="cuda")
            # add fare to factor
            self.fare = torch.tensor(fare, device="cuda")
            
 
    def __getitem__(self, i):
        return (self.sex[i], self.pclass[i], self.age[i], self.sibsp[i], self.parch[i], self.embarked[i], self.fare[i])

    def __len__(self):
        return len(self.sex)


def train(train_data: List, model: Titanic):
    TrainGenerator = DataGeneretor(train_data)
    TrainLoader = DataLoader(TrainGenerator, batch_size=2, shuffle=True)
    Criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    model.train()
    total_loss = 0
    cnt = 0
    pbar = tqdm(TrainLoader)
    for sex, pclass, age, sibsp, parch, embarked, fare, survive in pbar:
        cnt += 1
        outputs = model.forward(sex, pclass, age, sibsp, parch, embarked, fare)
        outputs = outputs.squeeze(1)
        survive = survive.squeeze(1)
        optimizer.zero_grad()
        loss = Criterion(outputs, survive)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_description(f"Loss: {total_loss / cnt}, Total Loss: {total_loss}")


def valid(valid_data: List, model: Titanic):
    ValidGenerator = DataGeneretor(valid_data)
    ValidLoader = DataLoader(ValidGenerator, batch_size=2, shuffle=True)
    model.eval()
    TP, FN, FP, TN = 0, 0, 0, 0
    with torch.no_grad():
        for sex, pclass, age, sibsp, parch, embarked, fare, survive in tqdm(ValidLoader):
            outputs = model.forward(sex, pclass, age, sibsp, parch, embarked, fare)
            outputs = outputs.argmax(dim=2)
            for i in range(outputs.shape[0]):
                if outputs[i][0] == 1 and survive[i][0] == 1:
                    TP += 1
                elif outputs[i][0] == 1 and survive[i][0] == 0:
                    FP += 1
                elif outputs[i][0] == 0 and survive[i][0] == 0:
                    FN += 1
                else:
                    TN += 1
    if TP == 0 and FP == 0:
        return 0
    recall = TP / (TP + FP)
    precision = (TP + FN) / (TP + FP + FN + TN)
    if recall == 0 or precision == 0:
        return 0
    return 2 * recall * precision / (recall + precision)


def processor(train_data: List, valid_data: List, model: Titanic, epoch: str, model_path: str):
    best_f1_score = 0
    patience = 0
    for epoch_num in range(epoch):
        train(train_data, model)
        valid_f1_score = valid(valid_data, model)
        if valid_f1_score >= best_f1_score:
            best_f1_score = valid_f1_score
            torch.save(model.state_dict(), model_path)
            patience = 0
            print(f"Epoch Num is {epoch_num}, F1 Update to {valid_f1_score}")
        else:
            patience += 1
            print(f"Epoch Num is {epoch_num}, F1 score is {valid_f1_score}, Not Update!")
            if patience >= 20:
                break


def predict(test_data: List, model: Titanic, model_path: str, result_path: str):
    model.load_state_dict(torch.load(model_path))
    TestGenerator = TestDataGeneretor(test_data)
    TestLoader = DataLoader(TestGenerator, batch_size=1)
    cnt = 0
    passengerId = list()
    survived = list()
    with torch.no_grad():
        for sex, pclass, age, sibsp, parch, embarked, fare in tqdm(TestLoader):
            outputs = model.forward(sex, pclass, age, sibsp, parch, embarked, fare)
            outputs = outputs.argmax(dim=2)
            for i in range(outputs.shape[0]):
                if outputs[i][0] == 0:
                    survived.append(0)
                else:
                    survived.append(1)
                passengerId.append(test_data[cnt][0])
                cnt += 1
    dataframe = pd.DataFrame({'PassengerId': passengerId, 'Survived': survived})
    dataframe.to_csv(result_path, index=False, sep=',')

if __name__ == "__main__":
    model = Titanic(512, 128, 16).to("cuda")
    processor(train_data, valid_data, model, 100, "E://Kaggle/Titanic/Generative Data/model_2.pt")
    predict(test_data, model, "E://Kaggle/Titanic/Generative Data/model_2.pt", "E://Kaggle/Titanic/Generative Data/result_2.csv")
