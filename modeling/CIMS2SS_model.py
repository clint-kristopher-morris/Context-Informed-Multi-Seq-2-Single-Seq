import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, n]
        b, c, n = x.size()

        # feature descriptor on the global temporal information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MultiheadAttentionV2(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, device='cpu', d_key=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.device = device

        d_key = d_model if d_key is None else d_key
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(288, 288)

        self.out_lin = nn.Linear(5, 5)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model // nhead])).to(device)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len_q, d_model = query.size()
        seq_len_k = key.size(1)

        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        q = q.view(batch_size, seq_len_q, self.nhead, d_model // self.nhead)
        k = k.view(batch_size, seq_len_k, self.nhead, d_model // self.nhead)
        v = v.view(batch_size, 5, self.nhead, 288 // self.nhead)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_q, d_model // self.nhead)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_k, d_model // self.nhead)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, 5, 288 // self.nhead)

        attn_weights = torch.bmm(q, k.permute(0, 2, 1)) / self.scale

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1)
        attn_output = self.out_lin(attn_output)
        attn_output = attn_output.permute(0, 2, 1)
        return attn_output, attn_weights


class DotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.permute(0, 2, 1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        x = torch.matmul(attention_weights, value)
        #         x = self.fc(x)

        return x, attention_weights


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, device='cpu'):
        super().__init__()

        self.self_attn = MultiheadAttentionV2(d_model, nhead, dropout=dropout, device=device)
        self.feed_forward = nn.Sequential(
            nn.Linear(288, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 288)
        )
        self.norm1 = nn.LayerNorm(288)
        self.norm2 = nn.LayerNorm(288)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # x, context, context,
        attn_output, attn_weights = self.self_attn(q, k, v, attn_mask=mask)

        x = v + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x, attn_weights


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, device='cpu', d_key=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.device = device

        d_key = d_model if d_key is None else d_key
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)

        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model // nhead])).to(device)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len_q, d_model = query.size()
        seq_len_k = key.size(1)

        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        q = q.view(batch_size, seq_len_q, self.nhead, d_model // self.nhead)
        k = k.view(batch_size, seq_len_k, self.nhead, d_model // self.nhead)
        v = v.view(batch_size, seq_len_k, self.nhead, d_model // self.nhead)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_q, d_model // self.nhead)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_k, d_model // self.nhead)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_k, d_model // self.nhead)

        attn_weights = torch.bmm(q, k.permute(0, 2, 1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.view(self.nhead, batch_size, seq_len_q, d_model // self.nhead)
        attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len_q, d_model)

        attn_output = self.out_lin(attn_output)
        return attn_output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, device='cpu'):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, device=device)
        #         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, transformer_encoder_layer, d_model, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            transformer_encoder_layer
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None):
        x = self.layer_norm(src)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.dropout(x)


class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self, d_model=256, num_heads=8, num_layers=5, dropout=0.4,
                 dim_feedforward=2048, device='cpu'):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.device = device

        self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads,
                                                     dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, self.d_model, num_layers, dropout=dropout)
        self.num_layers = num_layers

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        mask = self._generate_square_subsequent_mask(len(src))
        mask = mask.to(self.device)

        src = self.transformer_encoder(src, mask=mask)
        return src


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm1d(out_chanels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionBlockLong(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool,
                 red_7x7, out_7x7, red_15x15, out_15x15, strides=1):
        super(InceptionBlockLong, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1, stride=strides)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1, stride=strides))

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2, stride=strides))

        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=11, padding=5, stride=strides))

        self.branch5 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=25, padding=12, stride=strides))

        self.branch6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=strides),
            ConvBlock(in_channels, out_pool, kernel_size=1))

    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4, self.branch5, self.branch6)
        return torch.cat([branch(x) for branch in branches], 1)


class TimeSeriesNeighborPredictor(nn.Module):
    def __init__(self,
                 batch_size=8,
                 feat_channels=10,
                 disc_outsize=12,
                 middle_dim=32,
                 testing=False,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.disc_outsize = disc_outsize
        self.batch_size = batch_size
        self.feat_channels = feat_channels
        self.middle_dim = middle_dim

        self.cross_channel_attn = CrossAttentionLayer(d_model=2, nhead=8, dim_feedforward=1024, device=device)

        self.trans = Transformer(d_model=16, num_heads=16, num_layers=5, dropout=0.4,
                                 device=device)  # [seq_length, batchsize, embedding]
        self.trans2 = Transformer(d_model=16, num_heads=16, num_layers=5, dropout=0.4, device=device)
        self.activationf = nn.ReLU()
        self.fc2 = self.init_weights(nn.Linear(10, 27))

        self.testing = testing

        self.conv1 = ConvBlock(5, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(32, 64, kernel_size=9, stride=1, padding=1)

        self.inception3a = InceptionBlockLong(5, 40, 60, 80, 10, 20, 20, 10, 20, 5, 10)
        self.inception3b = InceptionBlockLong(260, 80, 80, 120, 20, 60, 40, 20, 40, 10, 20)
        self.inception4a = InceptionBlockLong(480, 120, 60, 130, 10, 30, 40, 10, 20, 5, 10)

        self.inceptionOut1 = InceptionBlockLong(6, 8, 12, 16, 2, 4, 4, 2, 4, 1, 2)
        self.inceptionOut2 = InceptionBlockLong(52, 16, 16, 24, 4, 12, 8, 4, 8, 2, 4)
        self.inceptionOut3 = InceptionBlockLong(16, 48, 32, 26, 10, 8, 8, 4, 8, 4, 8)

        self.inception1x1 = ConvBlock(480, 32, kernel_size=1, stride=1)

        self.inception1x1v2 = ConvBlock(96, 16, kernel_size=1, stride=1)
        self.inception1x1v3 = ConvBlock(16, 1, kernel_size=1, stride=1)

        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_out1 = nn.Linear(4448, 288)
        self.bn = nn.BatchNorm1d(17)

        self.q_dense = nn.ReLU(nn.Linear(5, 5))
        self.k_dense = nn.ReLU(nn.Linear(5, 5))
        self.dot_product_attention = DotProductAttention(d_model=5)

        self.attn_weights = None

    def init_weights(self, decoder):
        initrange = 0.1
        decoder.bias.data.zero_()
        decoder.weight.data.uniform_(-initrange, initrange)
        return decoder

    def forward(self, src_input, src_disc_in):
        # --- Pot Product Attention --- #
        ts_only = src_input[:, :5]
        #         ts_only = ts_only.permute(0,2,1)
        # if self.testing: print(f'shape TimeSeries Only 1: {ts_only.shape}')
        csf_dtw = src_disc_in[:, :, 0]
        # if self.testing: print(f'shape csf_dtw: {csf_dtw.shape}')
        query = self.q_dense(csf_dtw).reshape(-1, 5, 1)
        key = self.k_dense(csf_dtw).reshape(-1, 5, 1)
        # if self.testing: print(f'shape query: {query.shape}')
        # if self.testing: print(f'shape key: {key.shape}')
        dpa, weights = self.dot_product_attention(query, key, ts_only)
        # if self.testing: print(f'shape dot_product_attention: {dpa.shape}\n')

        self.attn_weights = weights

        # --- Time Series Inception --- #
        src = self.inception3a(dpa)
        # if self.testing: print(f'Time Series Inception 1: {src.shape}')
        src = self.inception3b(src)
        # if self.testing: print(f'Time Series Inception 2: {src.shape}')
        src = nn.MaxPool1d(kernel_size=2, padding=1, stride=2)(src)
        # if self.testing: print(f'Time Series Inception 3: {src.shape}')
        src = self.inception4a(src)
        # if self.testing: print(f'Time Series Inception 4: {src.shape}')
        src = self.inception1x1(src)
        # if self.testing: print(f'Time Series Inception 5: {src.shape}')
        src = self.avgpool(src)
        # if self.testing: print(f'Time Series Inception 6: {src.shape}\n')

        # --- FFN --- #
        # if self.testing: print(f'FFN 1: {src.shape}')
        src = src.reshape(src.shape[0], -1)
        # if self.testing: print(f'FFN 3: {src.shape}')
        src = self.dropout(src)
        # if self.testing: print(f'FFN 4: {src.shape}')
        src = self.fc_out1(src)
        src = src.reshape(src.shape[0], 1, -1)
        # if self.testing: print(f'FFN 5: {src.shape}\n')

        return src
