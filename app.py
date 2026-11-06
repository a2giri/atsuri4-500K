from flask import Flask, request, jsonify
from flask_cors import CORS
import 【entity-torch¦canonical_name=torch】, sentencepiece as spm, os, re
import 【entity-torch¦canonical_name=torch】.nn as nn
import 【entity-torch¦canonical_name=torch】.nn.functional as F

DEVICE = 【entity-torch¦canonical_name=torch】.device("cpu")
SPM_PREFIX = "chat_v6_spm"
MODEL_PATH = "chat_lstm_fineweb_v6.pt"

# --- モデル定義 (v6のまま) ---
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.W1=nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W2=nn.Linear(hidden_dim, attention_dim, bias=False)
        self.V=nn.Linear(attention_dim, 1, bias=False)
    def forward(self, decoder_hidden, encoder_outputs, mask):
        query=decoder_hidden.unsqueeze(1)
        score=self.V(【entity-torch¦canonical_name=torch】.tanh(self.W1(encoder_outputs)+self.W2(query))).squeeze(-1)
        score=score.masked_fill(~mask, -1e9)
        attention=F.softmax(score, dim=1)
        context=【entity-torch¦canonical_name=torch】.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_dim, dropout):
        super().__init__()
        self.embedding=embedding
        self.lstm=nn.LSTM(embedding.embedding_dim, hidden_dim, batch_first=True)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        embedded=self.dropout(self.embedding(x))
        outputs,(hidden,cell)=self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, embedding, hidden_dim, attention_dim, dropout):
        super().__init__()
        self.embedding=embedding
        self.attention=BahdanauAttention(hidden_dim, attention_dim)
        self.lstm=nn.LSTM(embedding.embedding_dim+hidden_dim, hidden_dim, batch_first=True)
        self.proj=nn.Linear(hidden_dim+hidden_dim, embedding.embedding_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self, token, hidden, cell, encoder_outputs, mask):
        embedded=self.dropout(self.embedding(token).unsqueeze(1))
        context, attention=self.attention(hidden[-1], encoder_outputs, mask)
        lstm_in=【entity-torch¦canonical_name=torch】.cat([embedded, context.unsqueeze(1)], dim=2)
        output,(hidden,cell)=self.lstm(lstm_in, (hidden,cell))
        output=output.squeeze(1)
        emb_out=self.dropout(self.proj(torch.cat([output, context], dim=1)))
        logits=F.linear(emb_out, self.embedding.weight)
        return logits, hidden, cell, attention

class Seq2Seq(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim, attn_dim, pad_id, dropout):
        super().__init__()
        self.embedding=nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.encoder=Encoder(self.embedding, hidden_dim, dropout)
        self.decoder=Decoder(self.embedding, hidden_dim, attn_dim, dropout)

# --- 読み込み ---
sp = spm.SentencePieceProcessor()
sp.load(SPM_PREFIX + ".model")
PAD, BOS, EOS, UNK = sp.pad_id(), sp.bos_id(), sp.eos_id(), sp.unk_id()

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
config = ckpt.get("config", {"emb_dim":160,"hidden_dim":256,"attn_dim":64,"vocab":sp.get_piece_size(),"dropout":0.2})
VOCAB = config["vocab"]
model = Seq2Seq(VOCAB, config["emb_dim"], config["hidden_dim"], config["attn_dim"], PAD, config["dropout"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()

def encode_input(text):
    return sp.encode(text, out_type=int)[:32]

@【entity-torch¦canonical_name=torch】.no_grad()
def generate_response(user_text, temperature=0.7, top_k=20, repetition_penalty=1.2, max_len=40):
    src_ids=encode_input(user_text)
    if not src_ids: return ""
    src=torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    mask=src!=PAD
    enc_out, hidden, cell=model.encoder(src)
    cur=torch.tensor([BOS], dtype=torch.long, device=DEVICE)
    gen=[]
    for step in range(max_len):
        logits, hidden, cell,_=model.decoder(cur, hidden, cell, enc_out, mask)
        logits=logits[0].clone()
        logits[PAD]=-float("inf"); logits[BOS]=-float("inf")
        if step<2: logits[EOS]=-float("inf")
        if gen:
            for tid in set(gen):
                if logits[tid]>0: logits[tid]/=repetition_penalty
                else: logits[tid]*=repetition_penalty
        logits=logits/max(float(temperature),0.05)
        if top_k:
            k=min(int(top_k), logits.size(0)); vals,_=torch.topk(logits,k); logits[logits<vals[-1]]=-float("inf")
        probs=F.softmax(logits, dim=-1)
        tid=torch.multinomial(probs,1).item() if torch.isfinite(probs).all() else int(logits.argmax().item())
        if tid==EOS: break
        if tid in (PAD,BOS,UNK): continue
        gen.append(tid); cur=torch.tensor([tid], dtype=torch.long, device=DEVICE)
    return re.sub(r"\s+"," ", sp.decode(gen).strip())

# --- Flask ---
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "OK - /chat?text=こんにちは"

@app.route("/chat")
def chat():
    text = request.args.get("text","こんにちは")
    ans = generate_response(text)
    return jsonify({"input": text, "output": ans or "うまく答えられなかったよ"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)