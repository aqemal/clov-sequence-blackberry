"""Offline binary classification benchmarks over already-audited behavioural traces.
No APK loading, execution or sandbox orchestration.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from core import score, train_baseline
from transformer_model import EventVocabulary, train_transformer
from verified_training import inspect_partitions

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, dimension=48):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,dimension,padding_idx=0)
        self.conv=nn.Conv1d(dimension,64,kernel_size=3,padding=1)
        self.clf=nn.Linear(64,1)
    def forward(self,x):
        mask=x.ne(0)
        z=torch.relu(self.conv(self.embed(x).transpose(1,2)))
        z=z.masked_fill(~mask.unsqueeze(1),-1e4)
        pooled=z.max(dim=2).values
        return self.clf(pooled).squeeze(1)

class BiLSTMClassifier(nn.Module):
    def __init__(self,vocab_size,dimension=48,hidden=32):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,dimension,padding_idx=0)
        self.lstm=nn.LSTM(dimension,hidden,batch_first=True,bidirectional=True)
        self.clf=nn.Linear(hidden*2,1)
    def forward(self,x):
        lengths=x.ne(0).sum(dim=1).clamp(min=1)
        emb=self.embed(x)
        packed=nn.utils.rnn.pack_padded_sequence(emb,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_) = self.lstm(packed)
        pooled=torch.cat([hidden[-2],hidden[-1]],dim=1)
        return self.clf(pooled).squeeze(1)

def train_neural(train,test,kind,epochs=6,seed=42,max_len=48):
    if kind not in ('cnn','bilstm'):raise ValueError('kind must be cnn or bilstm')
    if not 1<=int(epochs)<=100:raise ValueError('epochs must be 1-100')
    torch.manual_seed(int(seed));random.seed(int(seed));np.random.seed(int(seed))
    torch.set_num_threads(min(torch.get_num_threads(),4))
    vocab=EventVocabulary(train.event_list)
    xt=torch.tensor([vocab.encode(x,max_len) for x in train.event_list],dtype=torch.long)
    xv=torch.tensor([vocab.encode(x,max_len) for x in test.event_list],dtype=torch.long)
    yt=torch.tensor(train.target.to_numpy(),dtype=torch.float32)
    model=(CNNClassifier if kind=='cnn' else BiLSTMClassifier)(len(vocab.stoi))
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    loader=DataLoader(TensorDataset(xt,yt),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    losses=[]
    for _ in range(int(epochs)):
        model.train();running=0.0
        for x,y in loader:
            opt.zero_grad();out=model(x);loss=nn.functional.binary_cross_entropy_with_logits(out,y)
            loss.backward();opt.step();running+=loss.item()*len(x)
        losses.append(running/len(xt))
    model.eval()
    with torch.no_grad():probs=torch.sigmoid(model(xv)).cpu().numpy()
    return {'model':model,'vocab':vocab,'probability':probs,'losses':losses,'metrics':score(test.target.to_numpy(),probs)}

def compare_models(train,test,epochs=6,seed=42,group_key='sample_id',models=None):
    """Return serializable metrics and predictions; all models share identical holdout."""
    a,b,audit=inspect_partitions(train,test,group_key=group_key)
    models=['baseline','cnn','bilstm','transformer'] if models is None else list(models)
    allowed={'baseline','cnn','bilstm','transformer'}
    if not models or any(x not in allowed for x in models):raise ValueError('Unknown model selection')
    records=[];predictions=b[['sample_id','label','malware_family']].copy()
    for kind in models:
        if kind=='baseline':r=train_baseline(a,b)
        elif kind=='transformer':r=train_transformer(a,b,epochs=int(epochs),seed=int(seed))
        else:r=train_neural(a,b,kind,epochs=int(epochs),seed=int(seed))
        predictions[kind+'_p_malicious']=np.asarray(r['probability'],dtype=float)
        records.append({'model':kind,**{k:r['metrics'][k] for k in ('accuracy','precision','recall','f1')},'confusion_matrix':r['metrics']['confusion_matrix']})
    return records,predictions.reset_index(drop=True),audit
