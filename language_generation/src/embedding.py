import torch
import numpy as np
from openai_func import generate_embedding
import requests

def get_model_embedding(model, tokenizer, text, device):
    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs,output_hidden_states=True)
    embeddings = outputs.hidden_states[-1] 
    mean_embedding = np.mean(embeddings.cpu().detach().numpy(), axis=1)
    return np.squeeze(mean_embedding)

def get_model_embedding_part(model, tokenizer, text1, text2, device):
    inputs1 = tokenizer.encode(text1, return_tensors="pt", add_special_tokens=False)
    inputs2 = tokenizer.encode(text2, return_tensors="pt", add_special_tokens=False)
    inputs = torch.cat([inputs1,inputs2], axis = 1)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs,output_hidden_states=True)
    embeddings = outputs.hidden_states[-1][:,-len(inputs2[0]):,:] 
    mean_embedding = np.mean(embeddings.cpu().detach().numpy(), axis=1)
    return np.squeeze(mean_embedding)

def get_model_embedding_all(model, tokenizer, text, device):
    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs,output_hidden_states=True)
    embeddings = torch.cat(outputs.hidden_states)[1:,:,:]
    mean_embedding = np.mean(embeddings.cpu().detach().numpy(), axis=1)
    return np.squeeze(mean_embedding)

def get_model_embedding_old(model, tokenizer, text, device):
    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs,output_hidden_states=True)
    embeddings = outputs.hidden_states[-1] 
    cls_embedding = embeddings[:, 0, :].cpu().detach().numpy()
    return np.squeeze(cls_embedding)

def get_api_embedding(text, model='text-embedding-ada-002'):
    return generate_embedding(text, model)

def get_api2d_embedding(text, model='text-embedding-ada-002'):
    url = "https://openai.api2d.net/v1/embeddings"

    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer fk199699-lBMWIxPsVjqyOjFafaU2Bu2jVFrExedD' # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
    }

    data = {
    "model": model,
    "input": text
    }

    response = requests.post(url, headers=headers, json=data)

    return np.array(response.json()['data'][0]['embedding'])
