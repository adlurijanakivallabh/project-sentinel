import numpy as np
import time


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def batch_norm_2d(x, weight, bias, running_mean, running_var, eps=1e-5):
    rm = running_mean.reshape(1, -1, 1, 1)
    rv = running_var.reshape(1, -1, 1, 1)
    w = weight.reshape(1, -1, 1, 1)
    b = bias.reshape(1, -1, 1, 1)
    return w * (x - rm) / np.sqrt(rv + eps) + b


def im2col(x, Kh, Kw, Oh, Ow, dilation=1):
    N, Ci, Hp, Wp = x.shape
    cols = np.empty((N, Oh, Ow, Ci, Kh, Kw), dtype=x.dtype)
    for i in range(Kh):
        h = i * dilation
        for j in range(Kw):
            w = j * dilation
            cols[:, :, :, :, i, j] = x[:, :, h:h+Oh, w:w+Ow].transpose(0, 2, 3, 1)
    return cols.reshape(N * Oh * Ow, Ci * Kh * Kw)


def conv2d(x, weight, bias, padding=0, dilation=1):
    N, Ci, H, W = x.shape
    Co, _, Kh, Kw = weight.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    Kh_eff = (Kh - 1) * dilation + 1
    Kw_eff = (Kw - 1) * dilation + 1
    _, _, Hp, Wp = x.shape
    Oh = Hp - Kh_eff + 1
    Ow = Wp - Kw_eff + 1

    col = im2col(x, Kh, Kw, Oh, Ow, dilation)
    W_flat = weight.reshape(Co, -1)
    out = col @ W_flat.T
    if bias is not None:
        out += bias
    return out.reshape(N, Oh, Ow, Co).transpose(0, 3, 1, 2)


def conv2d_1x1(x, weight, bias):
    N, Ci, H, W = x.shape
    Co = weight.shape[0]
    x_flat = x.reshape(N, Ci, H * W).transpose(0, 2, 1).reshape(N * H * W, Ci)
    out = x_flat @ weight.reshape(Co, Ci).T
    if bias is not None:
        out += bias
    return out.reshape(N, H, W, Co).transpose(0, 3, 1, 2)


def maxpool2d(x, kernel_size=2):
    N, C, H, W = x.shape
    Oh = H // kernel_size
    Ow = W // kernel_size
    x_r = x[:, :, :Oh * kernel_size, :Ow * kernel_size]
    return x_r.reshape(N, C, Oh, kernel_size, Ow, kernel_size).max(axis=(3, 5))


def global_avg_pool(x):
    return x.mean(axis=(2, 3))


def linear(x, weight, bias):
    return x @ weight.T + bias


def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):
    gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh
    hs = h.shape[-1]
    i = sigmoid(gates[:, :hs])
    f = sigmoid(gates[:, hs:2*hs])
    g = np.tanh(gates[:, 2*hs:3*hs])
    o = sigmoid(gates[:, 3*hs:])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    hs = h.shape[-1]
    gi = x @ W_ih.T + b_ih
    gh = h @ W_hh.T + b_hh
    r = sigmoid(gi[:, :hs] + gh[:, :hs])
    z = sigmoid(gi[:, hs:2*hs] + gh[:, hs:2*hs])
    n = np.tanh(gi[:, 2*hs:] + r * gh[:, 2*hs:])
    return (1 - z) * n + z * h


class SentinelNumPy:

    def __init__(self, weights_path, zscore_path=None):
        print(f"[*] Loading weights: {weights_path}")
        self.w = dict(np.load(weights_path, allow_pickle=True))
        for k in list(self.w.keys()):
            self.w[k] = np.ascontiguousarray(self.w[k].astype(np.float32))
        print(f"[OK] Loaded {len(self.w)} weight tensors")

        if zscore_path:
            zs = np.load(zscore_path)
            self.zscore_mean = zs['mean'].astype(np.float32)
            self.zscore_std = zs['std'].astype(np.float32)
            self.zscore_std[self.zscore_std < 1e-7] = 1.0
            print(f"[OK] Z-score stats loaded")

    def _cbn(self, x, prefix, padding=1, dilation=1):
        x = conv2d(x, self.w[f'{prefix}.0.weight'], self.w[f'{prefix}.0.bias'],
                   padding=padding, dilation=dilation)
        x = batch_norm_2d(x, self.w[f'{prefix}.1.weight'], self.w[f'{prefix}.1.bias'],
                          self.w[f'{prefix}.1.running_mean'], self.w[f'{prefix}.1.running_var'])
        return gelu(x)

    def forward(self, x):
        B = x.shape[0]
        x_img = x.reshape(B, 1, 20, 30).astype(np.float32)

        b1 = self._cbn(x_img, 'cnn.0.branch1', padding=1, dilation=1)
        b2 = self._cbn(x_img, 'cnn.0.branch2', padding=3, dilation=3)
        b3 = self._cbn(x_img, 'cnn.0.branch3', padding=6, dilation=6)
        cat = np.concatenate([b1, b2, b3], axis=1)

        se = global_avg_pool(cat)
        se = relu(linear(se, self.w['cnn.0.se.se.2.weight'], self.w['cnn.0.se.se.2.bias']))
        se = sigmoid(linear(se, self.w['cnn.0.se.se.4.weight'], self.w['cnn.0.se.se.4.bias']))
        cat = cat * se.reshape(B, 192, 1, 1)

        cat = cat + conv2d_1x1(x_img, self.w['cnn.0.residual.weight'], self.w['cnn.0.residual.bias'])
        cat = maxpool2d(cat, 2)

        co = conv2d(cat, self.w['cnn.2.conv.0.weight'], self.w['cnn.2.conv.0.bias'], padding=1)
        co = batch_norm_2d(co, self.w['cnn.2.conv.1.weight'], self.w['cnn.2.conv.1.bias'],
                           self.w['cnn.2.conv.1.running_mean'], self.w['cnn.2.conv.1.running_var'])
        co = gelu(co)
        co = co + conv2d_1x1(cat, self.w['cnn.2.skip.weight'], self.w['cnn.2.skip.bias'])
        co = maxpool2d(co, 2)

        B2, C, T, F = co.shape
        seq = co.transpose(0, 2, 1, 3).reshape(B2, T, C * F)

        p = gelu_skip = layer_norm(
            linear(seq, self.w['temporal.proj.0.weight'], self.w['temporal.proj.0.bias']),
            self.w['temporal.proj.1.weight'], self.w['temporal.proj.1.bias'])

        SL = p.shape[1]
        HS = 128

        hf = np.zeros((B2, HS), dtype=np.float32)
        cf = np.zeros((B2, HS), dtype=np.float32)
        hb = np.zeros((B2, HS), dtype=np.float32)
        cb = np.zeros((B2, HS), dtype=np.float32)
        fwd0 = []
        bwd0 = [None] * SL
        for t in range(SL):
            hf, cf = lstm_cell(p[:, t], hf, cf,
                               self.w['temporal.lstm.weight_ih_l0'], self.w['temporal.lstm.weight_hh_l0'],
                               self.w['temporal.lstm.bias_ih_l0'], self.w['temporal.lstm.bias_hh_l0'])
            fwd0.append(hf)
            rt = SL - 1 - t
            hb, cb = lstm_cell(p[:, rt], hb, cb,
                               self.w['temporal.lstm.weight_ih_l0_reverse'], self.w['temporal.lstm.weight_hh_l0_reverse'],
                               self.w['temporal.lstm.bias_ih_l0_reverse'], self.w['temporal.lstm.bias_hh_l0_reverse'])
            bwd0[rt] = hb
        l0 = np.stack([np.concatenate([fwd0[t], bwd0[t]], -1) for t in range(SL)], 1)

        hf = np.zeros((B2, HS), dtype=np.float32)
        cf = np.zeros((B2, HS), dtype=np.float32)
        hb = np.zeros((B2, HS), dtype=np.float32)
        cb = np.zeros((B2, HS), dtype=np.float32)
        fwd1 = []
        bwd1 = [None] * SL
        for t in range(SL):
            hf, cf = lstm_cell(l0[:, t], hf, cf,
                               self.w['temporal.lstm.weight_ih_l1'], self.w['temporal.lstm.weight_hh_l1'],
                               self.w['temporal.lstm.bias_ih_l1'], self.w['temporal.lstm.bias_hh_l1'])
            fwd1.append(hf)
            rt = SL - 1 - t
            hb, cb = lstm_cell(l0[:, rt], hb, cb,
                               self.w['temporal.lstm.weight_ih_l1_reverse'], self.w['temporal.lstm.weight_hh_l1_reverse'],
                               self.w['temporal.lstm.bias_ih_l1_reverse'], self.w['temporal.lstm.bias_hh_l1_reverse'])
            bwd1[rt] = hb
        lstm_out = np.stack([np.concatenate([fwd1[t], bwd1[t]], -1) for t in range(SL)], 1)

        hg = np.zeros((B2, HS), dtype=np.float32)
        g0 = []
        for t in range(SL):
            hg = gru_cell(p[:, t], hg,
                          self.w['temporal.gru.weight_ih_l0'], self.w['temporal.gru.weight_hh_l0'],
                          self.w['temporal.gru.bias_ih_l0'], self.w['temporal.gru.bias_hh_l0'])
            g0.append(hg)
        gs0 = np.stack(g0, 1)

        hg = np.zeros((B2, HS), dtype=np.float32)
        g1 = []
        for t in range(SL):
            hg = gru_cell(gs0[:, t], hg,
                          self.w['temporal.gru.weight_ih_l1'], self.w['temporal.gru.weight_hh_l1'],
                          self.w['temporal.gru.bias_ih_l1'], self.w['temporal.gru.bias_hh_l1'])
            g1.append(hg)
        gru_out = np.stack(g1, 1)

        fused = layer_norm(
            linear(np.concatenate([lstm_out, gru_out], -1),
                   self.w['temporal.fusion.0.weight'], self.w['temporal.fusion.0.bias']),
            self.w['temporal.fusion.1.weight'], self.w['temporal.fusion.1.bias'])

        dm, nh, dk = 256, 8, 32
        Q = linear(fused, self.w['attention.W_q.weight'], self.w['attention.W_q.bias'])
        K = linear(fused, self.w['attention.W_k.weight'], self.w['attention.W_k.bias'])
        V = linear(fused, self.w['attention.W_v.weight'], self.w['attention.W_v.bias'])
        Q = Q.reshape(B2, SL, nh, dk).transpose(0, 2, 1, 3)
        K = K.reshape(B2, SL, nh, dk).transpose(0, 2, 1, 3)
        V = V.reshape(B2, SL, nh, dk).transpose(0, 2, 1, 3)
        aw = softmax(np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(dk), -1)
        ao = np.matmul(aw, V).transpose(0, 2, 1, 3).reshape(B2, SL, dm)
        ao = linear(ao, self.w['attention.W_o.weight'], self.w['attention.W_o.bias'])
        ao = layer_norm(ao + fused, self.w['attention.norm.weight'], self.w['attention.norm.bias'])

        pw = softmax(aw.mean(1).sum(-1), -1)
        pooled = np.einsum('bt,btd->bd', pw, ao)

        out = gelu(linear(layer_norm(pooled, self.w['classifier.0.weight'], self.w['classifier.0.bias']),
                          self.w['classifier.1.weight'], self.w['classifier.1.bias']))
        return linear(out, self.w['classifier.4.weight'], self.w['classifier.4.bias'])

    def predict(self, x):
        logits = self.forward(x)
        probs = softmax(logits / 0.5, -1)
        pred = np.argmax(probs, -1)
        return pred, np.max(probs, -1), probs


if __name__ == '__main__':
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    engine = SentinelNumPy(os.path.join(base, 'sentinel_weights.npz'),
                           os.path.join(base, 'zscore_stats.npz'))
    CLASS_NAMES = ['Normal', 'DoS/DDoS', 'PortScan/Recon', 'Web/Injection',
                   'Brute Force', 'Botnet/C2', 'Malware/Exploit', 'Infiltration']
    data = np.load(os.path.join(base, 'traffic_samples.npz'))
    for cls in CLASS_NAMES:
        if cls in data:
            s = data[cls][0:1].astype(np.float32)
            t0 = time.perf_counter()
            logits = engine.forward(s)
            ms = (time.perf_counter() - t0) * 1000
            pred = int(np.argmax(logits[0]))
            ok = "OK" if CLASS_NAMES[pred] == cls else "WRONG"
            print(f"[{ok}] {cls:20s} -> {CLASS_NAMES[pred]:20s}  {ms:.0f}ms")
