import torch
import torch.nn as nn

class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_node = 1024


        self.fc1 = nn.Linear(self.n_obj, self.n_node)
        self.fc2 = nn.Linear(self.n_node, self.n_node)


        self.fc3_head1 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head2 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head3 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head4 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head5 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head6 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head7 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head8 = nn.Linear(self.n_node, self.n_dim)
        self.fc3_head9 = nn.Linear(self.n_node, self.n_dim)

    def forward(self, pref):
        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))


        head1 = torch.sigmoid(self.fc3_head1(x))
        head2 = torch.sigmoid(self.fc3_head2(x))
        head3 = torch.sigmoid(self.fc3_head3(x))
        head4 = torch.sigmoid(self.fc3_head4(x))
        head5 = torch.sigmoid(self.fc3_head5(x))
        head6 = torch.sigmoid(self.fc3_head6(x))
        head7 = torch.sigmoid(self.fc3_head7(x))
        head8 = torch.sigmoid(self.fc3_head8(x))
        head9 = torch.sigmoid(self.fc3_head9(x))

        return head1.to(torch.float64), head2.to(torch.float64), head3.to(torch.float64), head4.to(torch.float64), head5.to(torch.float64), head6.to(torch.float64), head7.to(torch.float64), head8.to(torch.float64), head9.to(torch.float64)
