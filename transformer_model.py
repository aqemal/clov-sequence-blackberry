"""Small sequence-aware Transformer research demonstrator; CPU-friendly defaults."""
from collections import Counter
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from core import score

class EventVocabulary:
    def __init__(self, traces, max_words=1500):
        counts = Counter(token for trace in traces for token in trace)
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        self.stoi.update({t: i+2 for i,(t,_) in enumerate(counts.most_common(max_words-2))})
    def encode(self, events, max_len=48):
        ids = [self.stoi.get(t, 1) for t in events[:max_len]]
        return ids + [0] * (max_len - len(ids))

class SequenceTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=48, dimension=48, heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.position = nn.Embedding(max_len, dimension)
        self.attn = nn.MultiheadAttention(dimension, heads, dropout=0.05, batch_first=True)
        self.norm = nn.LayerNorm(dimension)
        self.ff = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.norm2 = nn.LayerNorm(dimension)
        self.clf = nn.Sequential(nn.Linear(dimension, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, ids, need_attention=False):
        mask = ids.eq(0)
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        x = self.embed(ids) + self.position(positions)
        attended, a = self.attn(x, x, x, key_padding_mask=mask, need_weights=need_attention,
                         average_attn_weights=False)
        x = self.norm(x + attended)
        x = self.norm2(x + self.ff(x))
        weights = (~mask).unsqueeze(-1)
        pooled = (x*weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.clf(pooled).squeeze(1), a

def train_transformer(train, test, epochs=8, max_len=48, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    vocab = EventVocabulary(train['event_list'])
    xtr = torch.tensor([vocab.encode(e, max_len) for e in train['event_list']], dtype=torch.long)
    xte = torch.tensor([vocab.encode(e, max_len) for e in test['event_list']], dtype=torch.long)
    ytr = torch.tensor(train['target'].to_numpy(), dtype=torch.float32)
    model = SequenceTransformer(len(vocab.stoi), max_len=max_len)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    losses=[]
    for _ in range(epochs):
        total=0.0
        for xs,ys in loader:
            optim.zero_grad()
            logits, _ = model(xs)
            loss=loss_fn(logits,ys)
            loss.backward()
            optim.step()
            total += float(loss.item())*len(xs)
        losses.append(total/len(xtr))
    model.eval()
    with torch.no_grad():
        probs=torch.sigmoid(model(xte)[0]).numpy()
    return {'model':model, 'vocab':vocab, 'probability':probs,
            'metrics':score(test['target'].to_numpy(),probs), 'losses':losses,'max_len':max_len}

def predict_one(result, events):
    model, vocab, max_len = result['model'], result['vocab'], result['max_len']
    seq=events[:max_len]
    encoded=torch.tensor([vocab.encode(events,max_len)],dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, attention=model(encoded,need_attention=True)
        prob=float(torch.sigmoid(logits)[0].item())
    n=len(seq)
    # Mean attention over heads. This describes attention allocation, NOT causal feature attribution.
    matrix=attention[0].mean(dim=0).numpy()[:n,:n]
    importance=matrix.mean(axis=0)
    return prob, seq, matrix, importance
