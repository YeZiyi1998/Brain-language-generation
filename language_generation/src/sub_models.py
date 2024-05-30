import torch
import torch.nn as nn
import random

class MLP(torch.nn.Module):
    def __init__(self,num_input,num_classes,position_index=False,num_layers = 2, args=None) :
        super(MLP,self).__init__()
        self.num_input=num_input
        self.num_classes=num_classes
        dropout = 0.5 if 'dropout' not in args.keys() else args['dropout']
        max_seq_len = 5
        embedding_size = self.num_input
        self.args = args
        if self.args['input_method'] == 'mask_input':
            self.position_embedding = nn.Parameter(torch.empty(max_seq_len, num_classes), requires_grad=True)
            nn.init.uniform_(self.position_embedding, -1, 1)
        elif position_index:   
            self.position_embedding = nn.Parameter(torch.empty(max_seq_len, embedding_size), requires_grad=True) # 
            nn.init.uniform_(self.position_embedding, -1, 1)
        net = nn.Sequential()
        num_layers = args['num_layers']
        for i in range(num_layers):
            if i==0:
                if args['pos']:
                    net.add_module(f'linear{i+1}',nn.Linear(self.num_input,num_input,bias=False, dtype=torch.float32))    
                else:
                    net.add_module(f'linear{i+1}',nn.Linear(self.num_input,num_input, dtype=torch.float32))
            else:
                net.add_module(f'linear{i+1}',nn.Linear(num_input,num_input,bias=False, dtype=torch.float32))
            if args['activation'] == 'relu':
                net.add_module(f'ReLu{i+1}',nn.ReLU())
            elif args['activation'] == 'relu6':
                net.add_module(f'ReLu{i+1}',nn.ReLU6())
            elif args['activation'] == 'sigmoid':
                net.add_module(f'Sig{i+1}',nn.Sigmoid())
            elif args['activation'] == 'tanh':
                net.add_module(f'tanh{i+1}',nn.Tanh())
            net.add_module(f'Dropout{i+1}',nn.Dropout(dropout))
        net.add_module(f'linear{num_layers+1}',nn.Linear(num_input,num_classes,bias=False, dtype=torch.float32))

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        net.apply(init_weights)
        self.net=net

    def forward(self,X, position_index = False):
        if position_index == False:
            return self.net(X) #X: batch_size * seqlength * dim
        else:
            return self.net(X+self.position_embedding[:X.shape[1],:])

class RNN(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(RNN, self).__init__()
        hidden_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, input_vec,position_index=False):
        batch_size = input_vec.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.rnn(input_vec, hidden) # b*seq*dim; 
        output = self.fc(output)
        return output

    def init_hidden(self, batch_size, half=True):
        if half:
            return torch.zeros(1, batch_size, self.hidden_size).half().to(self.device)
        else:
            return torch.zeros(1, batch_size, self.hidden_size).to(self.device)

class Linear(nn.Module):
    def __init__(self, input_size, output_size, args=None, seqlength=1):
        super(Linear, self).__init__()
        if args['pos']:
            self.linear = nn.Linear(input_size,output_size, bias=False,dtype=torch.float32)
        else:
            self.linear = nn.Linear(input_size,output_size, dtype=torch.float32)

    def forward(self, input, position_index = False):
        return self.linear(input)
         

class MultiMLP(nn.Module):
    def __init__(self,num_input,num_classes,position_index=False,num_layers = 2, args=None):
        super(MultiMLP, self).__init__()
        seq_len = 4
        self.mlps = nn.ModuleList([MLP(num_input, num_classes,position_index = False, args = args) for _ in range(seq_len)])
        self.num_classes = num_classes
        self.device = torch.device(f'cuda:{args["cuda"]}')
    def forward(self, x,position_index=False):
        batch_size, seq_len, dim = x.size()
        out = torch.zeros(batch_size, seq_len, self.num_classes, device=self.device)
        for i in range(seq_len):
            out[:, i, :] = self.mlps[i](x[:, i, :])
        return out

class BigMLP(nn.Module):
    def __init__(self,num_input,num_classes,position_index=False,num_layers = 2, args=None):
        super(BigMLP, self).__init__()
        seq_len = 4
        self.mlp = MLP(num_input * seq_len, args['word_embed_size'] * seq_len, position_index = False, args = args)
        self.seq_len = seq_len
    def forward(self,x,position_index = False):
        batch_size, seq_len, dim = x.size()
        # Reshape input to merge the sequence and feature dimensions
        x = x.view(batch_size, -1)
        # Pass through MLP
        x = self.mlp(x)
        # Reshape output to split sequence and feature dimensions
        x = x.view(batch_size, self.seq_len, -1)
        return x

class Encoding_model(nn.Module):
    def __init__(self, args,brain_embed_size=None,device=None):
        super(Encoding_model, self).__init__()
        if brain_embed_size is None:
            brain_embed_size = args['brain_embed_size']
        if args['brain_model'] == 'multi_mlp':
            self.model = MultiMLP(brain_embed_size,args['word_embed_size'],position_index = False, args = args)
        elif args['brain_model'] == 'big_mlp':
            self.model = BigMLP(brain_embed_size,args['word_embed_size'],position_index = False, args = args)
        elif args['brain_model'] == 'linear':
            self.model = Linear(brain_embed_size,args['word_embed_size'], args,)
        elif args['brain_model'] == 'mlp':
            self.model = MLP(brain_embed_size,args['word_embed_size'],position_index = args['pos'], args = args)
        elif args['brain_model'] == 'rnn':
            self.model = RNN(brain_embed_size,args['word_embed_size'], device)
            
    def forward(self, x, position_index = False):
        # x: batch_size * seq_len * dim
        return self.model(x, position_index = position_index)