import torch
from model import BPNet
from torch.utils.data import DataLoader
from dataset.dataset_preparator import HeartAttackDataset

def eva():
    model = BPNet.BPN()
    model.load_state_dict(torch.load("check_point.pth"))
    model.eval()

    test_dataset = HeartAttackDataset('E:\\BianCheng\\Heart Attack Predictor\\dataset\\test.csv')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    close_count = 0
    total = 0
    tolerance = 0.5
    with torch.no_grad():
        for features, label in test_loader:
            output = model(features)
            output = output.squeeze()  # Ensure output is a 1D tensor for comparison
            total += label.size(0)
            difference = torch.abs(output - label.squeeze())  # Make sure label is also 1D for comparison
            close_count += (difference <= tolerance).sum().item()  # Correctly count predictions within tolerance
    if total > 0:
        close_ratio = close_count / total
        
        # print('Predictions within 0.2 of true value:', close_count)
        # print('Total samples:', total)
        print('Percentage of predictions within 0.2 of true value: {:.2f}%'.format(close_ratio*100.0))
        return close_ratio
    else:
        print('No samples to evaluate.')
