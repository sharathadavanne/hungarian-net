import torch
from train_hnet import HNetGRU, HungarianDataset
from IPython import embed
import matplotlib.pyplot as plot
import random

use_cuda = False
max_len = 2

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

model = HNetGRU(max_len=max_len).to(device)
model.eval()

model.load_state_dict(torch.load("data/hnet_model.pt" ))
test_data = HungarianDataset(train=False, max_len=max_len)

for _ in range(20):
    feat, labels = test_data.__getitem__(random.choice(range(len(test_data))))
    feat = torch.tensor(feat).unsqueeze(0).to(device).float()
    hidden = torch.zeros(1, 1, 128)
    pred, pred2, hidden = model(feat, hidden)
    pred = pred.squeeze().sigmoid().clone().detach().numpy()

    print(feat.squeeze().numpy().reshape(max_len,max_len))
    print(pred.reshape(max_len, max_len))
    print(labels[0].squeeze().reshape(max_len, max_len))
    print(labels[1].squeeze())
    print('\n\n')

    plot.plot(labels[0].reshape(-1), label='ref', color='red', linestyle='dashed', marker='o',  markerfacecolor='green', markersize=8)
    plot.plot(pred, label='predicted', color='blue')
    plot.legend()
    plot.ylim([0, 1])
    plot.show()




