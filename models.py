import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(embedding_dim, n_positions,num_heads, num_layers,args):
    if args.large_scale_ICL:
        model = TransformerModel_LS(
            n_positions=2 *n_positions,
            n_embd=embedding_dim,
            n_layer=num_layers,
            n_head=num_heads,
            args=args
        )
    else:
        model = TransformerModel(
            n_positions=2 * n_positions,
            n_embd=embedding_dim,
            n_layer=num_layers,
            n_head=num_heads,
            args=args
        )
    return model


class TransformerModel_LS(nn.Module):
    def __init__(self, n_positions, n_embd, n_layer, n_head,args):
        super(TransformerModel_LS, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self._read_in = nn.Linear(args.numofAnt*args.numofAP*2+int(args.modulationAware)+int(args.numberUEAware), n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 2)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        bsize_y, points_y, dim_y = ys_b.shape
        '''zero pad to match dimension'''
        ys_b=torch.cat((ys_b,torch.zeros(bsize_y,points_y,dim-dim_y).to(device)),dim=2)
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, ys_batch, xs_batch):
        task_token=ys_batch[:,:4,:]
        zs = self._combine(ys_batch[:,4:,:], xs_batch)
        zs = torch.cat((task_token,zs.to(torch.float32)),axis=1)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        '''Mapping to Constallation Symbol'''
        return prediction[:, 4::2, :]




class TransformerModel(nn.Module):
    def __init__(self, n_positions, n_embd, n_layer, n_head,args):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self._read_in = nn.Linear(args.numofAnt*args.numofAP*2+int(args.modulationAware)+int(args.numberUEAware), n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 2)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        bsize_y, points_y, dim_y = ys_b.shape
        '''zero pad to match dimension'''
        ys_b=torch.cat((ys_b,torch.zeros(bsize_y,points_y,dim-dim_y).to(device)),dim=2)
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, ys_batch, xs_batch):
        zs = self._combine(ys_batch, xs_batch)
        zs = zs.to(torch.float32)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        '''Mapping to Constallation Symbol'''
        return prediction[:, ::2, :]
