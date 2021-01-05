
class ForecastLSTM(nn.Module):
    def __init__(self, n_features, out_dim, n_layers, hid_dim, batch_size):
        super(ForecastLSTM, self).__init__()
        self.n_features = n_features
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(n_features, hid_dim, num_layers=n_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x, h):
        """
        `(seq_len, batch, input_size)` input size

        """
        x = x.view(self.batch_size, -1, self.n_features)
        lstm_out, h = self.lstm(x, h)
        out = lstm_out.contiguous().view(-1, self.hid_dim)
        out = self.fc(out)
        out = out.view(self.batch_size, -1, self.out_dim)[:, -1]

        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, self.batch_size, self.hid_dim).zero_().cuda(),
                    weight.new(self.n_layers, self.batch_size, self.hid_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, self.batch_size, self.hid_dim).zero_(),
                    weight.new(self.n_layers, self.batch_size, self.hid_dim).zero_())
        return hidden
