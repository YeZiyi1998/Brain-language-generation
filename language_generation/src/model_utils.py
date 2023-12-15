import torch
import torch.nn as nn
from split_model import Split_gpt0, Split_gpt1
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
            nn.init.uniform_(self.position_embedding, -1, 1) # -0.02, 0.02
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
        if self.args['input_method'] == 'mask_input':
            return self.position_embedding[:X.shape[1],:].unsqueeze(0).tile(X.shape[0],1,1)
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

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(self.device)

class Linear(nn.Module):
    def __init__(self, input_size, output_size, args=None, seqlength=1):
        super(Linear, self).__init__()
        if args['pos']:
            self.linear = nn.Linear(input_size,output_size, bias=False,dtype=torch.float32)
        else:
            self.linear = nn.Linear(input_size,output_size, dtype=torch.float32)

    def forward(self, input, position_index = False):
        input = self.linear(input)
        return input

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

class Prompt_model(nn.Module):
    def __init__(self, args, model, tokenizer, device,new_tokens):
        super(Prompt_model, self).__init__()
        self.model = model
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        if args['model_split'] != -1 and 'gpt2' in args['model_name']:
            self.split_model0, self.split_model1 = Split_gpt0(self.model, layer = args['model_split'], model_name = args['model_name']), Split_gpt1(self.model, layer = args['model_split'])
            self.split_model0.to(device)
            self.split_model1.to(device)
            self.additional_prompt = Encoding_model(args)
            self.additional_prompt.to(device)
        self.mse_loss = nn.MSELoss()  
        tmp_weights = []
        for new_token in new_tokens:
            # 发现bug
            # new_token_id = self.tokenizer.convert_tokens_to_ids(f"[{new_token}]")
            new_token_id = self.tokenizer.convert_tokens_to_ids(f"{new_token}")
            if 'gpt2' in self.args['model_name']:
                tmp_weight = self.model.transformer.wte.weight[new_token_id]
            elif 'llama' in self.args['model_name']:
                tmp_weight = self.model.model.embed_tokens.weight[new_token_id]
            tmp_weights.append(tmp_weight)
        tmp_weights = torch.cat(tmp_weights, dim=0)
        self.token_weights = nn.Parameter(tmp_weights.clone().detach(), requires_grad=True)
    
    def init_encoding_model(self, ):
        if len(self.args['roi_selected']) > 1:
            self.encoding_model = []
            for k in range(len(self.args['roi_selected'])):
                self.encoding_model.append(Encoding_model(self.args, brain_embed_size=self.args['roi_size'][k]).to(self.device))
        else:
            self.encoding_model = Encoding_model(self.args, device = self.device)
            self.encoding_model.to(self.device)
        if self.args['model_name'] in ['llama-7b', 'vicuna-7b','llama-7b-old']:
            self.encoding_model.half()
        if self.args['load_check_point']:
            if type(self.encoding_model) == list:
                for i in range(len(self.encoding_model)):
                    self.encoding_model[i].load_state_dict(self.check_point['encoding_model'][i])
                    self.encoding_model[i].to(self.device)
            else:
                self.encoding_model.load_state_dict(self.check_point['encoding_model'])    
                self.encoding_model.to(self.device)   
        
    def words2embedding(self, input_ids):
        if self.args['model_name'] in ['llama-7b', 'vicuna-7b','llama-7b-old']:
            return self.model.get_input_embeddings()(input_ids)
        else:
            if type(input_ids) == list:
                re = []
                for item in input_ids:
                    re.append(self.model.transformer.wte(item))
                return re
            else:
                return self.model.transformer.wte(input_ids)

    def get_prev(self, additional_bs, content_prev_sep):
        if type(additional_bs) == list:
            re = []
            for k in range(len(additional_bs)):
                k_roi_toknizer = self.tokenizer.encode_plus([f'<roi{k}/>', f'<roi{k}/>'],return_tensors='pt')['input_ids'].to(self.device)
                k_roi_toknizer = self.words2embedding(k_roi_toknizer)
                re += [k_roi_toknizer[:,:1,:], additional_bs[k], k_roi_toknizer[:,1:,:],]    
            return re   
            # 
        else:
            if self.args['model_name'] in ['llama-7b','llama-7b-old']:
                return [content_prev_sep[:,:1,:], content_prev_sep[:,1:2,:], additional_bs, content_prev_sep[:,2:,:],]
            else:
                return [content_prev_sep[:,:1,:], additional_bs, content_prev_sep[:,1:,:],]

    def get_tokens(self, content_prev_sep):
        # batchsize * seqlength * shape
        content_prev_sep = self.words2embedding(content_prev_sep)
        content_prev_sep[:,-1] = self.token_weights[-1]
        content_prev_sep[:,-2] = self.token_weights[-2]
        return content_prev_sep

    # 目前代码还有一个bug是，对于without_input，输入还有可能是啥都没有，咋整呢，add special tokens吗？
    def tokenize(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=True,mode='train'):
        content_all = self.words2embedding(content_all)
        content_prev_sep = self.get_tokens(content_prev_sep)
        
        if random.random() > self.args['fake_input'] or use_fake == False:
            if type(additional_bs) == list:
                additional_bs_tokenized = []
                for k in range(len(additional_bs)):
                    additional_bs_tokenized.append(self.encoding_model[k](additional_bs[k]))
            else:
                additional_bs_tokenized = self.encoding_model(additional_bs, position_index = self.args['pos'])
        else:
            if type(additional_bs) == list:
                additional_bs_tokenized = []
                for k in range(len(additional_bs)):
                    additional_bs_tokenized[k] = self.words2embedding(content_all)
            else:
                additional_bs_tokenized = self.words2embedding(content_all)
        if self.args['without_input']:
            if self.args['model_name'] in ['llama-7b', 'llama-7b-old']:
                content_all_list = [self.get_prev(additional_bs_tokenized, content_prev_sep)[0]] + [content_all,]
                content_all_mask = torch.cat([additional_bs_mask[:,:1], content_all_mask], dim=-1)
            else:
                content_all_list = [content_all,]
        elif self.args['without_text'] and mode in ['test']:
            content_all_list = self.get_prev(additional_bs_tokenized, content_prev_sep)
            content_all_mask = torch.cat([additional_bs_mask, ], dim=-1)
        else:
            content_all_list = self.get_prev(additional_bs_tokenized, content_prev_sep) + [ content_all,]
            content_all_mask = torch.cat([additional_bs_mask, content_all_mask], dim=-1)
            if self.args['model_split'] != -1:
                content_all_mask[:,:len(additional_bs_mask)] = 0
        content_all = torch.cat(content_all_list, dim=-2)
        return content_all, content_all_mask

    def forward(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=True,mode='train'):
        content_all, content_all_mask = self.tokenize(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake,mode)
        if self.args['model_split'] != -1:
            args = self.split_model0(inputs_embeds=content_all, attention_mask = content_all_mask, return_dict=True)
            additional_bs = self.additional_prompt(additional_bs, self.args['pos'])
            args[0][:,1:additional_bs.shape[1]+1] = additional_bs
            if self.args['model_split'] != -1:
                args[4][:,:len(additional_bs_mask)] = 1
            output = self.split_model1(*args)
        else:
            output = self.model(inputs_embeds=content_all, attention_mask = content_all_mask)
        return output, content_all_mask
    
    def pad2left(self, content_prev, content_prev_mask):
        padding_counts = (content_prev_mask == 1).sum(dim=1)
        # 初始化新的张量
        front_padded_input_embeds = torch.zeros_like(content_prev)
        front_padded_mask = torch.zeros_like(content_prev_mask)

        for i in range(content_prev.size(0)):  # 遍历每个样本
            # 计算需要移动的位置数
            shift = padding_counts[i].item()
            # 为 input_embeds 和 mask 重新排列
            front_padded_input_embeds[i, content_prev.size(1) - shift:] = content_prev[i, :shift]
            front_padded_input_embeds[i, :content_prev.size(1) - shift] = content_prev[i, shift:]
            front_padded_mask[i, content_prev.size(1) - shift:] = content_prev_mask[i, :shift]
        return front_padded_input_embeds, front_padded_mask
    
    def get_perplexity(self, content_prev, content_prev_mask,  additional_bs, additional_bs_mask, content_prev_sep, candidate, ):
        content_prev, content_prev_mask = self.tokenize(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False, mode='test')
        content_prev, content_prev_mask = self.pad2left(content_prev, content_prev_mask)
        total_prob = 1
        for target_id in candidate[0]:
            # 使用模型进行预测，得到logits
            with torch.no_grad():
                outputs = self.model(inputs_embeds = content_prev, )
            logits = outputs.logits.squeeze(0)[-1]
            # 将logits转换为概率分布
            probs = torch.softmax(logits, dim=0)
            # 输出[MASK]被预测成"me"的概率
            prob = probs[target_id].item() / probs.sum().item()
            total_prob *= prob
            content_prev = torch.cat([content_prev, self.words2embedding(torch.tensor([[target_id]]).to(self.device))], dim=1)
        return total_prob
    
    def generate_more(self, content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, mode='test',max_new_tokens=32):
        content_prev, content_prev_mask = self.tokenize(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False, mode='test')
        content_prev, content_prev_mask = self.pad2left(content_prev, content_prev_mask)
        max_new_tokens = max(max_new_tokens, 4)
        if self.args['generation_method'] == 'greedy':
            seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=max_new_tokens,return_dict_in_generate=True,num_beams=1,do_sample=False, pad_token_id=self.tokenizer.eos_token_id,num_return_sequences=5,)
        elif self.args['generation_method'] == 'beam':
            if self.args['use_bad_words_ids']:
                seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=max_new_tokens,return_dict_in_generate=True,num_beams=5,do_sample=False, repetition_penalty=2.0,pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=5,bad_words_ids=self.bad_words_ids) 
            else:
                seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=max_new_tokens,return_dict_in_generate=True,num_beams=5,do_sample=False, repetition_penalty=2.0,pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=5,) 
        all_truncated_predictions = []
        for i in range(len(seq2seqLMoutput['sequences'])):
            predictions = seq2seqLMoutput['sequences'][i]
            truncated_prediction = []
            for t in predictions[1:]:
                if t != self.tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            all_truncated_predictions.append(truncated_prediction)
        return all_truncated_predictions
    
    def generate(self, content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, mode='test'):
        content_prev, content_prev_mask = self.tokenize(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False, mode='test')
        content_prev, content_prev_mask = self.pad2left(content_prev, content_prev_mask)

        if self.args['generation_method'] == 'greedy':
            seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=32,return_dict_in_generate=True,num_beams=1,do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        elif self.args['generation_method'] == 'beam':
            if self.args['use_bad_words_ids']:
                seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=32,return_dict_in_generate=True,num_beams=5,do_sample=False, repetition_penalty=2.0,pad_token_id=self.tokenizer.eos_token_id, bad_words_ids=self.bad_words_ids) 
            else:
                seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=32,return_dict_in_generate=True,num_beams=5,do_sample=False, repetition_penalty=2.0,pad_token_id=self.tokenizer.eos_token_id, ) 
        all_truncated_predictions = []
        for i in range(len(seq2seqLMoutput['sequences'])):
            predictions = seq2seqLMoutput['sequences'][i]
            truncated_prediction = []
            for t in predictions[1:]:
                if t != self.tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            all_truncated_predictions.append(truncated_prediction)
        return all_truncated_predictions

    def fuse(self, data, content_true_mask, fuse_len=4):
        # data is b * n * m, fuse to b * 4 * m, use a mean fusing for simplicity
        return torch.mean(data[:,:content_true_mask.shape[1],:], axis=1).unsqueeze(1).tile(1, fuse_len, 1)

    # 有个bug是在于 content_true和additional_bs的长度不一定一样
    def additional_loss(self, content_true, content_true_mask, additional_bs):
        fuse_len = additional_bs.shape[1]
        content_true = self.words2embedding(content_true)
        mean_content_true = self.fuse(content_true, content_true_mask, fuse_len)
        additional_bs = self.encoding_model(additional_bs)
        return self.mse_loss(additional_bs, mean_content_true)
    