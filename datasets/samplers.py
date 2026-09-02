import random
from collections import defaultdict

from torch.utils.data import Sampler


class TemporalBucketSampler(Sampler):
    """按序列分桶 + 桶内连续中心帧访问的 batch sampler。

    目的: 让 worker 侧的帧 LRU 缓存命中滑窗 80% 的重叠帧,
    将每样本解码次数从 6 张降到 ~1.2 张 (CPU 降 ~5x)。

    每个 epoch:
      - 桶(pair)顺序随机
      - 桶内起始偏移随机 + 方向随机(循环移位)
      - 按 batch_size 切连续中心帧块
    每个 sample 每 epoch 恰好出现一次。
    """

    def __init__(self, samples, batch_size, seed=None):
        self.batch_size = batch_size
        self._rng = random.Random(seed)
        pair_indices = defaultdict(list)
        for i, s in enumerate(samples):
            pair_indices[s["sequence"]].append(i)
        self.pair_orders = []
        for seq in sorted(pair_indices.keys()):
            idxs = pair_indices[seq]
            order = idxs[self._rng.randint(0, len(idxs) - 1):] + idxs[: self._rng.randint(0, len(idxs) - 1)]
            self.pair_orders.append(list(order))

    def __iter__(self):
        pairs = list(range(len(self.pair_orders)))
        self._rng.shuffle(pairs)
        batches = []
        for p in pairs:
            order = list(self.pair_orders[p])
            if self._rng.random() < 0.5:
                order.reverse()
            shift = self._rng.randint(0, len(order) - 1)
            order = order[shift:] + order[:shift]
            for i in range(0, len(order), self.batch_size):
                batches.append(order[i : i + self.batch_size])
        self._rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum((len(o) + self.batch_size - 1) // self.batch_size for o in self.pair_orders)
