
'''
 *                             _ooOoo_
 *                            o8888888o
 *                            88" . "88
 *                            (| -_- |)
 *                            O\  =  /O
 *                         ____/`---'\____
 *                       .'  \\|     |//  `.
 *                      /  \\|||  :  |||//  \
 *                     /  _||||| -:- |||||-  \
 *                     |   | \\\  -  /// |   |
 *                     | \_|  ''\---/''  |   |
 *                     \  .-\__  `-`  ___/-. /
 *                   ___`. .'  /--.--\  `. . __
 *                ."" '<  `.___\_<|>_/___.'  >'"".
 *               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
 *               \  \ `-.   \_ __\ /__ _/   .-` /  /
 *          ======`-.____`-.___\_____/___.-`____.-'======
 *                             `=---='
 *          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *                     佛祖保佑        永无BUG
'''

import torch
import torch.nn as nn
import torch.optim as optim
from model import BPNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset.dataset_preparator import HeartAttackDataset
import predict

learning_rate = 0.0001

train_dataset = HeartAttackDataset('E:/BianCheng/Heart Attack Predictor/dataset/heart_attack_prediction_dataset.csv')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = BPNet.BPN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
def train(pre_epochs, epochs):
    losses = []
    accuracies = []
    max_accuracy = 0
    # 如果预训练过了就加载已有权重
    if pre_epochs >= 1:
        model.load_state_dict(torch.load('check_point.pth'))
    for epoch in range(pre_epochs, epochs):
        print("epoch=",epoch)
        for i,(features,label) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, epochs, i+1, len(train_loader), loss.item()))
            losses.append(loss.item())
        torch.save(model.state_dict(), 'check_point.pth') #每一个epoch训练结束保存一次模型
        acc = predict.eva()
        accuracies.append(acc)
        if acc > max_accuracy:
            max_accuracy = acc
            torch.save(model.state_dict(), f'best_model_acc_{str(acc)}.pth')
    plt.plot(losses)
    plt.xlabel("ITERATIONS")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
    
    plt.plot(accuracies)
    plt.xlabel("ITERATIONS")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.show()

if __name__ == '__main__':
    train(1, 1000)
