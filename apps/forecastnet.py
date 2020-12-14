
import torch.nn as nn

class ForecastLSTM(nn.Module):
    def __init__(self, n_features, out_dim, n_steps, n_layers, hid_dim):

        super(ForecastLSTM, self).__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(n_features, hid_dim, num_layers=n_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x, h):
        """
        `(seq_len, batch, input_size)` input size

        """
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_steps, self.n_features)
        lstm_out, h = self.lstm(x, h)
        lstm_out = lstm_out.contiguous().view(batch_size, -1, self.hid_dim)[:, -1] # output size (batch*n_steps, hid_dim)
        output = self.fc(lstm_out)

        return output, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        # if torch.cuda.is_available():
        #     hidden = (weight.new(self.n_layers, batch_size, self.hid_dim).zero_().cuda(),
        #               weight.new(self.n_layers, batch_size, self.hid_dim).zero_().cuda())
        # else:
        hidden = (weight.new(self.n_layers, batch_size, self.hid_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hid_dim).zero_())
        return hidden
