import os
if os.path.exists('/home/yzy/'):
    server_name = 'faster'
elif os.path.exists('/data/home/scv6830/'):
    server_name = 'bingxing'
elif os.path.exists('/home/whs145/'):
    server_name = 'hendrix'
else:
    server_name = 'bingxing2'

llama_7b_dict = {'hendrix': '/home/whs145/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc', 'faster': '/home/yzy/.cache/huggingface/hub/decapoda-research--llama-7b-hf.main.5f98eefcc80e437ef68d457ad7bf167c2c6a1348', 'bingxing': '/data/home/scv6830/.cache/huggingface/hub/decapoda-research--llama-7b-hf.main.5f98eefcc80e437ef68d457ad7bf167c2c6a1348', 'bingxing2':'/home/bingxing2/home/scx7140/.cache/huggingface/hub/decapoda-research--llama-7b-hf.main.5f98eefcc80e437ef68d457ad7bf167c2c6a1348'}
llama_7b_2_dict = {'hendrix': '/home/whs145/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc', 'faster': '/home/yzy/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc', 'bingxing': '', 'bingxing2':'/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852'}
gpt2_xl_dict = {'faster': '/home/yzy/.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8', 'bingxing': '/data/home/scv6830/run/.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8', 'bingxing2':'/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8'}
gpt2_dict = {'faster': '/home/yzy/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8', 'hendirx':'/home/whs145/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10','bingxing': '/data/home/scv6830/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8', 'bingxing2':'/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8'}
gpt2_large_dict = {'faster': '/home/yzy/.cache/huggingface/hub/models--gpt2-large/snapshots/97935fc1a406f447320c3db70fe9e9875dca2595', 'bingxing': '/data/home/scv6830/.cache/huggingface/hub/models--gpt2-large/snapshots/97935fc1a406f447320c3db70fe9e9875dca2595', 'bingxing2':'/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--gpt2-large/snapshots/97935fc1a406f447320c3db70fe9e9875dca2595'}
gpt2_medium_dict = {'faster': '/home/yzy/.cache/huggingface/hub/models--gpt2-medium/snapshots/f65d4965d1221eff2bcf34f53a2ba12120e18f24', 'bingxing': '/data/home/scv6830/.cache/huggingface/hub/models--gpt2-medium/snapshots/f65d4965d1221eff2bcf34f53a2ba12120e18f24', 'bingxing2':'/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--gpt2-medium/snapshots/f65d4965d1221eff2bcf34f53a2ba12120e18f24'}

model_name2path = {
                    'vicuna-7b': '/work/czm/LLMs/FastChat/vicuna-7b',
                    'gpt2-large':gpt2_large_dict[server_name],
                    'gpt2-medium':gpt2_medium_dict[server_name],
                    'llama-7b-old': llama_7b_dict[server_name],
                    'llama-7b': llama_7b_2_dict[server_name],
                    'gpt2':gpt2_dict[server_name],
                    'gpt2-xl':gpt2_xl_dict[server_name],
                }