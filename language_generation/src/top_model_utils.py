import torch
import numpy as np
from torch.nn.functional import softmax
try:
    from GPT import GPT
    from modules.config import INIT, STOPWORDS, INIT_ids, STOPWORDS_ids
except:
    from src.GPT import GPT
    from src.modules.config import INIT, STOPWORDS, INIT_ids, STOPWORDS_ids
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

class Top_model():
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
        return self.tokenizer.encode(words, add_special_tokens=False)
    
    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs
    
    def get_probs_generation(self, content_all, additional_bs, additional_bs_mask, content_prev_sep):
        content_all = content_all[:,-500:] # beam_size * seq_length
        content_all_mask = torch.ones(content_all.shape).int().to(self.device)
        content_all = content_all.to(self.device)
        content_all2, content_all_mask2 = self.prompt_model.tokenize(content_all, content_all_mask, additional_bs[:content_all.shape[0]], additional_bs_mask[:content_all.shape[0]], content_prev_sep[:content_all.shape[0]], use_fake=False, mode='test')

        with torch.no_grad():
            output = self.model(inputs_embeds=content_all2, attention_mask = content_all_mask2) 
        
        probs = softmax(output.logits, dim = 2).detach().cpu().numpy()
    
        return probs
    
    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

def get_nucleus(probs, nuc_mass, nuc_ratio, k=200):
    """identify words that constitute a given fraction of the probability mass
    """
    nuc_ids = np.argsort(probs)[-k:][::-1]
    # nuc_ids = np.where(probs >= np.max(probs) * nuc_ratio)[0]
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
        if type(vocab[0]) is int:    
            self.ids = vocab
            self.type_is_id = True
        else:
            self.ids = {i for word, i in self.model.word2id.items() if word in set(vocab)}
            self.type_is_id = False
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
    
    def beam_propose(self, beam, context_words, gcontext=None, additional_bs=None,additional_bs_mask=None,content_prev_sep=None):
        """get possible extension words for each hypothesis in the decoder beam
        """
        if len(beam[0].words) == 0: 
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

class TokenLanguageModel(LanguageModel):
    """class for generating word sequences using a language model
    """
    def __init__(self, model, vocab, nuc_mass = 1.0, nuc_ratio = 0.0, model_name='',task_name=''):
        if vocab is None:
            vocab = {}  
        super().__init__(model, vocab, )    
        self.model_name = model_name
        self.stop_word_ids = set(STOPWORDS_ids[self.model_name])     
        self.task_name = task_name   

    def ps(self, contexts,additional_bs=None,additional_bs_mask=None, content_prev_sep=None):
        """get probability distributions over the next words for each context
        """
        context_arr = torch.tensor(contexts)
        if additional_bs is None:
            probs = self.model.get_probs(context_arr)
        else:
            probs = self.model.get_probs_generation(context_arr, additional_bs=additional_bs, content_prev_sep=content_prev_sep, additional_bs_mask=additional_bs_mask)
        return probs[:, -1] 
    
    def context_filter(self, proposals, context, ):
        """filter out words that occur in a context to prevent repetitions
        """
        cut_words = []
        cut_words.extend([context[i+1] for i, word in enumerate(context[:-1]) if word == context[-1]]) # bigrams
        len_cut_words = len(cut_words)
        cut_words.extend([x for i, x in enumerate(proposals) if x not in self.stop_word_ids and x in context]) # unigrams
        cut_words.extend([x for i, x in enumerate(proposals) if x in self.stop_word_ids and x in context[-3:]]) # unigrams
        if self.model_name == 'gpt2-xl' and self.task_name == 'Huth_3':
            cut_words.extend([x for i, x in enumerate(proposals) if (context[-1] == 11752 and x == 616) or (context[-1] == 588 and x == 11752) or (context[-1] == 460 and x == 466) or (context[-1] == 373 and x == 588)]) # unigrams
            cut_words.extend([12008])
        return [x for x in proposals if x not in cut_words]

    def beam_propose(self, beam, context_words, gcontext, additional_bs=None,additional_bs_mask=None,content_prev_sep=None):
        """get possible extension words for each hypothesis in the decoder beam
        """
        if len(beam[0].words) == 0: 
            nuc_words = [w_id for w_id in INIT_ids[self.model_name]]
            nuc_logprobs = np.log(np.ones(len(nuc_words)) / len(nuc_words))
            return [(nuc_words, nuc_logprobs)]
        else:
            contexts = [hyp.words[-gcontext:] for hyp in beam]
            beam_probs = self.ps(contexts, additional_bs=additional_bs,additional_bs_mask=additional_bs_mask,content_prev_sep=content_prev_sep)
            beam_nucs = []
            for context, probs in zip(contexts, beam_probs):
                nuc_ids = get_nucleus(probs, nuc_mass = self.nuc_mass, nuc_ratio = self.nuc_ratio)
                nuc_ids = [nuc_id for nuc_id in nuc_ids if nuc_id in self.ids] if len(self.ids) > 0 else nuc_ids
                nuc_ids = self.context_filter(nuc_ids, context[-context_words:])
                nuc_logprobs = np.log([probs[nuc_id] for nuc_id in nuc_ids])
                beam_nucs.append((nuc_ids, nuc_logprobs))
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