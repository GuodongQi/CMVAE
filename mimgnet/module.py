import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# The below code is adapted from github.com/juho-lee/set_transformers/blob/master/modules.py
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class CausalEncoder(nn.Module):
    def __init__(self, latent_size, bias=True, linear=True):
        super().__init__()
        self.latent_size = latent_size
        self.linear = linear
        if linear:
            self.I = torch.eye(latent_size).cuda()
            self.A = nn.Parameter(torch.tril(torch.zeros([self.latent_size, self.latent_size]), diagonal=-1).cuda(),
                                  requires_grad=True)
            # self.A = nn.Parameter(self.I,
            #                       requires_grad=False)
        else:
            dims = [self.latent_size, 32, 1]
            self.fc1 = nn.Linear(self.latent_size, self.latent_size * dims[1], bias=bias)
            layers = []
            for l in range(len(dims) - 2):
                layers.append(LocallyConnected(self.latent_size, dims[l + 1], dims[l + 2], bias=bias))
            self.fc2 = nn.ModuleList(layers)
            self.dims = dims

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # x: ..., n, d
        # get epsilon
        return x - self.mask(x)

    def mask(self, x):
        # x: ..., n, d
        # linear
        if self.linear:
            x_ = x.view(-1, x.shape[-2], x.shape[-1])
            x_ = x_.bmm(
                self.A.unsqueeze(0).repeat(x_.shape[0], 1, 1)
            )
            x_ = x_.view(x.shape)
        else:
            # nolinear
            x_ = self.fc1(x)  # [..., n, d*m1]
            x_ = x_.view(-1, self.dims[0], self.dims[1])
            for fc in self.fc2:
                x_ = F.elu(x_)  # [bn, d, m1]
                # x_ = F.relu(x_, inplace=True)  # [bn, d, m1]
                # x_ = F.leaky_relu(x_, 0.2, inplace=True)  # [bn, d, m1]
                x_ = fc(x_)  # [bn, d, m2]
            x_ = x_.squeeze(dim=2).view(x.shape)  # [bn, d] -> [..., n, d]
            # x_ = 0
        return x_

    def get_w(self):
        if self.linear:
            return self.A
        else:
            fc1_weight = self.fc1.weight.view(self.latent_size, -1, self.latent_size)
            W = fc1_weight.pow(2).sum(1).t()
            return W


class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.
    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not
    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]
    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )
