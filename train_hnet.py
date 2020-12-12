from generate_hnet_training_data import load_obj
from IPython import embed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
import time

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.conv_Q.in_channels,
            self.conv_V.out_channels,
            self.conv_K.out_channels
            )


class HNetGRU(nn.Module):
    def __init__(self, use_pos_enc=True, max_len=4):
        super().__init__()
        hidden_size = 128
        self.nb_gru_layers = 1
        self.gru = nn.GRU(max_len, hidden_size, self.nb_gru_layers, batch_first=True)
        self.gru_hidden_size = hidden_size
        self.attn = AttentionLayer(hidden_size, hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, max_len)

    def initHidden(self):
        return torch.zeros(self.nb_gru_layers, 256, self.gru_hidden_size)

    def forward(self, query, hidden):
        # query - batch x seq x feature

        out, hidden = self.gru(query, hidden)
        # out - batch x seq x hidden

        out = out.permute((0, 2, 1))
        # out - batch x hidden x seq

        out = self.attn.forward(out)
        # out - batch x hidden x seq

        out = out.permute((0, 2, 1))
        out = torch.tanh(out)
        # out - batch x seq x hidden

        out = self.fc1(out)
        # out - batch x seq x feature

        out1 = out.view(out.shape[0], -1)
        # out1 - batch x (seq x feature)

        # out2 = torch.sum(out, dim=-1)
        out2, dmp = torch.max(out, dim=-1)

        # out2 - batch x seq x 1
        return out1.squeeze(), out2.squeeze(), hidden


class HungarianDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True, max_len=4):
        if train:
            self.data_dict = load_obj('data/hung_data_train')
        else:
            self.data_dict = load_obj('data/hung_data_test')
        self.max_len = max_len

        self.pos_wts = np.ones(self.max_len**2)
        self.f_scr_wts = np.ones(self.max_len**2)
        if train:
            loc_wts = np.zeros(self.max_len**2)
            for i in range(len(self.data_dict)):
                label = self.data_dict[i][3]
                loc_wts += label.reshape(-1)
            self.f_scr_wts = loc_wts / len(self.data_dict)
            self.pos_wts = (len(self.data_dict)-loc_wts) / loc_wts

    def __len__(self):
        return len(self.data_dict)

    def get_pos_wts(self):
        return self.pos_wts

    def get_f_wts(self):
        return self.f_scr_wts

    def __getitem__(self, idx):
        feat = self.data_dict[idx][2]
        label = self.data_dict[idx][3]

        label = [label.reshape(-1), label.sum(-1)]
        return feat, label


def main():
    batch_size = 256
    nb_epochs = 1000
    max_len = 2

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = HungarianDataset(train=True, max_len=max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, drop_last=True)

    f_score_weights = np.tile(train_dataset.get_f_wts(), batch_size)
    print(train_dataset.get_f_wts())

    test_loader = DataLoader(
        HungarianDataset(train=False, max_len=max_len),
        batch_size=batch_size, shuffle=True, drop_last=True)

    model = HNetGRU(max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters())

    criterion1 = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion2 = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion_wts = [1., 1.]

    best_loss = -1
    best_epoch = -1
    for epoch in range(1, nb_epochs + 1):
        train_start = time.time()
        # TRAINING
        model.train()
        train_loss, train_l1, train_l2 = 0, 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device).float()
            target1 = target[0].to(device).float()
            target2 = target[1].to(device).float()

            optimizer.zero_grad()

            hidden = model.initHidden().to(device)
            output1, output2, hidden = model(data, hidden)

            l1 = criterion1(output1, target1)
            l2 = criterion2(output2, target2)
            loss = criterion_wts[0]*l1 + criterion_wts[1]*l2

            loss.backward()
            optimizer.step()

            train_l1 += l1.item()
            train_l2 += l2.item()
            train_loss += loss.item()

        train_l1 /= len(train_loader.dataset)
        train_l2 /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        train_time = time.time()-train_start
        #TESTING
        test_start = time.time()
        model.eval()
        test_loss, test_l1, test_l2 = 0, 0, 0
        test_f = 0
        nb_test_batches = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device).float()
                target1 = target[0].to(device).float()
                target2 = target[1].to(device).float()

                hidden = model.initHidden().to(device)

                output1, output2, hidden = model(data, hidden)
                l1 = criterion1(output1, target1)
                l2 = criterion2(output2, target2)
                loss = criterion_wts[0]*l1 + criterion_wts[1]*l2

                test_l1 += l1.item()
                test_l2 += l2.item()
                test_loss += loss.item()  # sum up batch loss

                f_pred = (torch.sigmoid(output1).cpu().numpy() > 0.5).reshape(-1)
                f_ref = target1.cpu().numpy().reshape(-1)
                test_f += f1_score(f_ref, f_pred, zero_division=1, average='weighted', sample_weight=f_score_weights)
                nb_test_batches += 1

        test_l1 /= len(test_loader.dataset)
        test_l2 /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        test_f /= nb_test_batches
        test_time = time.time() - test_start
        if test_f > best_loss:
            best_loss = test_f
            best_epoch = epoch
            torch.save(model.state_dict(), "data/hnet_model.pt")
        print('Epoch: {}\t time: {:0.2f}/{:0.2f}\ttrain_loss: {:.4f} ({:.4f}, {:.4f})\ttest_loss: {:.4f} ({:.4f}, {:.4f})\tf_scr: {:.4f}\tbest_epoch: {}\tbest_f_scr: {:.4f}'.format(epoch, train_time, test_time, train_loss, train_l1, train_l2, test_loss, test_l1, test_l2, test_f, best_epoch, best_loss))
    print('Best epoch: {}\nBest loss: {}'.format(best_epoch, best_loss))


if __name__ == "__main__":
    main()

