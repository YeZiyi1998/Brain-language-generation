import torch
import numpy as np
from torch.nn.functional import softmax

class Top_model(GPT):
    def __init__(self, model, tokenizer, device = 'cpu', prompt_model=None):  
        self.device = device
        self.model = model
        self.prompt_model = prompt_model
        self.tokenizer = tokenizer
        self.word2id = tokenizer.get_vocab()
        self.vocab = [item[0] for item in sorted(self.word2id.items(), key=lambda v:v[1])]
        
    def encode(self, words):
        """map from words to ids
        """
        return self.tokenizer.encode(words)
    
    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs
    
    def get_probs_generation(self, content_all, additional_bs, additional_bs_mask, content_prev_sep):
        # 所有模型至少能接受长度为512的输入，先进行截断
        content_all = content_all[:,-500:] # beam_size * seq_length * embed_size
        content_all_mask = torch.ones(content_all.shape).int().to(self.device)
        content_all = content_all.to(self.device)

        content_all, content_all_mask = self.prompt_model.tokenize(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, mode='test')
        
        with torch.no_grad():
            output = self.model(inputs_embeds=content_all, attention_mask = content_all_mask) 
        
        probs = softmax(output.logits, dim = 2).detach().cpu().numpy()
    
        return probs

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

INIT = ['i', 'we', 'she', 'he', 'they', 'it']
STOPWORDS = {'is', 'does', 's', 'having', 'doing', 'these', 'shan', 'yourself', 'other', 'are', 'hasn', 'at', 'for', 'while', 'down', "hadn't", 'until', 'above', 'during', 'each', 'now', 'have', "won't", 'once', 'why', 'here', 'ourselves', 'to', 'over', 'into', 'who', 'that', 'myself', 'he', 'themselves', 'were', 'against', 'about', 'some', 'has', 'but', 'ma', 'their', 'this', 'there', 'with', "that'll", "shan't", "wouldn't", 'a', 'those', "you'll", 'll', 'few', 'couldn', 'an', 'd', "weren't", 'doesn', 'own', 'won', 'didn', 'what', 'when', 'in', 'below', 'where', "it's", 'most', 'just', "you're", 'yourselves', 'too', "don't", "she's", "didn't", "hasn't", 'isn', "mustn't", 'of', 'did', 'how', 'himself', 'aren', 'if', 'very', 'or', 'weren', 'it', 'be', 'itself', "doesn't", 'my', 'o', 'no', "isn't", 'before', 'after', 'off', 'was', 'can', 'the', 'been', 'her', 'him', "wasn't", 've', 'through', "needn't", 'because', 'nor', 'will', 'm', 't', 'out', 'on', 'she', 'all', 'then', 'than', "mightn't", 'hers', 'herself', 'only', 'should', 're', 'ain', 'wasn', "aren't", "couldn't", 'they', 'hadn', 'had', 'more', 'and', 'under', "shouldn't", 'any', 'y', 'don', 'from', 'so', 'whom', 'as', 'mustn', 'between', 'up', 'do', 'both', 'such', 'our', 'its', 'which', 'not', "haven't", 'needn', 'by', "should've", 'again', 'shouldn', 'his', 'me', 'further', 'yours', 'am', 'your', 'haven', 'wouldn', 'being', 'ours', 'you', 'i', 'theirs', 'mightn', 'same', 'we', "you've", 'them', "you'd"}

def get_nucleus(probs, nuc_mass, nuc_ratio):
    """identify words that constitute a given fraction of the probability mass
    """
    nuc_ids = np.where(probs >= np.max(probs) * nuc_ratio)[0]
    nuc_pairs = sorted(zip(nuc_ids, probs[nuc_ids]), key = lambda x : -x[1]) 
    sum_mass = np.cumsum([x[1] for x in nuc_pairs])
    cutoffs = np.where(sum_mass >= nuc_mass)[0]
    if len(cutoffs) > 0: nuc_pairs = nuc_pairs[:cutoffs[0]+1]
    nuc_ids = [x[0] for x in nuc_pairs]                     
    return nuc_ids

def in_context(word, context):
    """test whether [word] or a stem of [word] is in [context]
    """
    stem_context = [stemmer.stem(x) for x in context]
    stem_word = stemmer.stem(word)
    return (stem_word in stem_context or stem_word in context)

def context_filter(proposals, context):
    """filter out words that occur in a context to prevent repetitions
    """
    cut_words = []
    cut_words.extend([context[i+1] for i, word in enumerate(context[:-1]) if word == context[-1]]) # bigrams
    cut_words.extend([x for x in proposals if x not in STOPWORDS and in_context(x, context)]) # unigrams
    return [x for x in proposals if x not in cut_words]

class LanguageModel():
    """class for generating word sequences using a language model
    """
    def __init__(self, model, vocab, nuc_mass = 1.0, nuc_ratio = 0.0):        
        self.model = model
        self.ids = {i for word, i in self.model.word2id.items() if word in set(vocab)}
        self.nuc_mass, self.nuc_ratio = nuc_mass, nuc_ratio
        
    def ps(self, contexts,additional_bs=None,additional_bs_mask=None, content_prev_sep=None):
        """get probability distributions over the next words for each context
        """
        context_arr = self.model.get_context_array(contexts)
        if additional_bs is None:
            probs = self.model.get_probs(context_arr)
        else:
            probs = self.model.get_probs_generation(context_arr, additional_bs=additional_bs, content_prev_sep=content_prev_sep, additional_bs_mask=additional_bs_mask)
        return probs[:, - 1] 
    
    def beam_propose(self, beam, context_words, additional_bs=None,additional_bs_mask=None,content_prev_sep=None):
        """get possible extension words for each hypothesis in the decoder beam
        """
        if len(beam) == 1: 
            nuc_words = [w for w in INIT if self.model.word2id[w] in self.ids]
            nuc_logprobs = np.log(np.ones(len(nuc_words)) / len(nuc_words))
            return [(nuc_words, nuc_logprobs)]
        else:
            contexts = [hyp.words[-context_words:] for hyp in beam]
            beam_probs = self.ps(contexts, additional_bs=additional_bs,additional_bs_mask=additional_bs_mask,content_prev_sep=content_prev_sep)
            beam_nucs = []
            for context, probs in zip(contexts, beam_probs):
                nuc_ids = get_nucleus(probs, nuc_mass = self.nuc_mass, nuc_ratio = self.nuc_ratio)
                nuc_words = [self.model.vocab[i] for i in nuc_ids if i in self.ids]
                nuc_words = context_filter(nuc_words, context)
                nuc_logprobs = np.log([probs[self.model.word2id[w]] for w in nuc_words])
                beam_nucs.append((nuc_words, nuc_logprobs))
            return beam_nucs

# 淘汰掉的用法？
def generate_beam(model, tokenizer, beam_size: int = 5, embed=None, entry_length=32, temperature=1., stop_token: str = '.', bad_words_ids = None, context_ids = []):
    model.eval()
    # stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    # is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        generated = embed
        for i in range(entry_length):
            outputs = model(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                # logits[is_stopped] = -float(np.inf)
                # logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                # seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                # is_stopped = is_stopped[next_tokens_source]
                
            next_token_embed = model.get_input_embeddings()(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            # is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            # if is_stopped.all():
            #     break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    order = scores.argsort(descending=True)
    output_list = np.array([output_list[i] for i in order])
    return torch.tensor(output_list)[0]