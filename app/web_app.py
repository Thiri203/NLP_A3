from flask import Flask, request
import torch
import sentencepiece as spm
from pathlib import Path

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizers
app_dir = Path(__file__).parent
sp_en = spm.SentencePieceProcessor(model_file=str(app_dir / "spm_en.model"))
sp_my = spm.SentencePieceProcessor(model_file=str(app_dir / "spm_my.model"))

PAD_EN = sp_en.pad_id()
PAD_MY = sp_my.pad_id()

#model
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_EN)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return outputs, hidden

class AdditiveAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.v = nn.Linear(hid_dim, 1, bias=False)
        self.W = nn.Linear(hid_dim, hid_dim)
        self.U = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        enc = encoder_outputs.permute(1,0,2)
        energy = self.v(torch.tanh(self.W(hidden) + self.U(enc))).squeeze(2)
        energy = energy.masked_fill(mask, -1e10)
        return F.softmax(energy, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_MY)
        self.rnn = nn.GRU(hid_dim * 2 + emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, output_dim)

    def forward(self, input_tok, hidden, encoder_outputs, mask):
        input_tok = input_tok.unsqueeze(0)
        embedded = self.embedding(input_tok)
        attn = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        enc = encoder_outputs.permute(1,0,2)
        weighted = torch.bmm(attn, enc).permute(1,0,2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(0)
        hidden = hidden.squeeze(0)
        embedded = embedded.squeeze(0)
        weighted = weighted.squeeze(0)
        pred = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def create_mask(self, src):
        return (src == PAD_EN).permute(1,0)

#loading model
ckpt = torch.load(str(app_dir / "additive_model.pt"), map_location=device)

attention = AdditiveAttention(ckpt["hid_dim"])
encoder = Encoder(ckpt["vocab_en"], 128, ckpt["hid_dim"])
decoder = Decoder(ckpt["vocab_my"], 128, ckpt["hid_dim"], attention)
model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()


@torch.no_grad()
def translate(sentence, max_len=50):
    src_ids = [sp_en.bos_id()] + sp_en.encode(sentence, out_type=int) + [sp_en.eos_id()]
    src = torch.tensor(src_ids).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_ids)]).to(device)

    enc_out, hidden = model.encoder(src, src_len)
    mask = model.create_mask(src)

    input_tok = torch.tensor([sp_my.bos_id()]).to(device)
    out_ids = []

    for _ in range(max_len):
        pred, hidden = model.decoder(input_tok, hidden, enc_out, mask)
        top1 = pred.argmax(1).item()
        if top1 == sp_my.eos_id():
            break
        out_ids.append(top1)
        input_tok = torch.tensor([top1]).to(device)

    return sp_my.decode(out_ids)


@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    if request.method == "POST":
        output = translate(request.form["text"])
    return f"""
    <h2>English → Myanmar Translation</h2>
    <form method="post">
        <textarea name="text" rows="4" cols="60"></textarea><br><br>
        <button type="submit">Translate</button>
    </form>
    <p><b>Output:</b> {output}</p>
    """

if __name__ == "__main__":
    app.run(debug=True)
