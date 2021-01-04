
class ForecastLSTM(nn.Module):
    def __init__(self, n_features, out_dim, n_steps, n_layers, hid_dims):

        super(ForecastLSTM, self).__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.hid_dims = hid_dims
        self.n_layers = n_layers

        self.lstm = nn.LSTM(n_features, hid_dims[0], num_layers=n_layers, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(hid_dims[0], hid_dims[1], bias=False)
        self.fc2 = nn.Linear(hid_dims[1], out_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, h):
        """
        `(seq_len, batch, input_size)` input size

        """
        batch_size = x.size(0)
        x = x.view(-1, self.n_steps, self.n_features)
        lstm_out, h = self.lstm(x, h)
        lstm_out = lstm_out.contiguous().view(-1, self.n_steps, self.hid_dims[0])[:, -1] # output size (batch*n_steps, hid_dim)
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.hid_dim).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.hid_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hid_dims[0]).zero_(),
                    weight.new(self.n_layers, batch_size, self.hid_dims[0]).zero_())
        return hidden
