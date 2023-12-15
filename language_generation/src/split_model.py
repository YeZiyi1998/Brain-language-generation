import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions 
import torch
from torch.nn import CrossEntropyLoss
from model_config import model_name2path

class Split_gpt0(GPT2LMHeadModel):
    def __init__(self, model,layer=6, model_name='gpt2-xl'):
        self.config = GPT2Config.from_pretrained(model_name2path[model_name])
        super(Split_gpt0, self).__init__(self.config)
        self.model = model
        self.h = self.model.transformer.h[:layer]
    
    def transformer_forward(self, 
        input_ids= None,
        past_key_values= None,
        attention_mask= None,
        token_type_ids = None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        encoder_hidden_states= None,
        encoder_attention_mask= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict = None,
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]

            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.model.transformer.wte(input_ids)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.model.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.model.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.model.transformer.gradient_checkpointing and self.model.transformer.training:
            if use_cache:
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.model.transformer.gradient_checkpointing and self.model.transformer.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model.transformer.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.model.transformer.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        return hidden_states, output_shape, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions, output_hidden_states, return_dict

    def forward(self, 
        input_ids= None,
        past_key_values= None,
        attention_mask= None,
        token_type_ids = None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        encoder_hidden_states= None,
        encoder_attention_mask= None,
        labels = None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict = None,
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs, output_shape, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions, output_hidden_states, return_dict = self.transformer_forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return transformer_outputs, output_shape, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache,  output_attentions, output_hidden_states, return_dict

class Split_gpt1(GPT2LMHeadModel):
    def __init__(self, model,layer=6, model_name='gpt2-xl'):
        self.config = GPT2Config.from_pretrained(model_name2path[model_name])
        super(Split_gpt1, self).__init__(self.config)
        self.model = model
        self.h = self.model.transformer.h[layer:]
    
    def transformer_forward(self, 
        hidden_states, 
        output_shape,
        input_ids= None,
        past_key_values= None,
        attention_mask= None,
        token_type_ids = None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        encoder_hidden_states= None,
        encoder_attention_mask= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict = None,
        ):
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model.transformer.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.model.transformer.gradient_checkpointing and self.model.transformer.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model.transformer.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.model.transformer.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.model.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


    # transformer_outputs, output_shape, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache,  output_attentions, output_hidden_states, return_dict
    def forward(self, 
            transformer_outputs,
            output_shape,
            input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache,  output_attentions, output_hidden_states, return_dict, labels=None
        ):

        transformer_outputs = self.transformer_forward(transformer_outputs,output_shape,
            input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache,  output_attentions, output_hidden_states,return_dict=return_dict,)

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model.model_parallel:
            torch.cuda.set_device(self.model.transformer.first_device)
            hidden_states = hidden_states.to(self.model.lm_head.weight.device)

        lm_logits = self.model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

if __name__ == '__main__':
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # 加载GPT-2模型和tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 要生成文本的输入IDs
    input_ids = [40, 120, 11, 389, 314, 318, 287, 318, 290, 168, 287, 11, 314, 318, 2628, 314, 262, 318, 290, 168, 287, 11, 314, 318, 2628, 314, 262, 318, 290, 168, 287, 11, 314, 318, 2628, 314, 262, 318, 290, 168, 287, 11, 314, 318, 2628, 314, 262, 318, 290, 168]

    # 将输入IDs转换为PyTorch张量
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    # 生成文本
    output = model(input_ids=input_ids, return_dict=True)

    split_model0, split_model1 = Split_gpt0(model, model_name='gpt2'), Split_gpt1(model)

    args = split_model0(input_ids=input_ids, return_dict=True)
    output_split = split_model1(*args)
    # output2 = model(input_ids=input_ids)
    # jiayudebug snippet start----------
    inputs = ''
    while inputs != 'continue':
        try:
            print(eval(inputs))
        except Exception as e:
            print(e)
        inputs = input()
    # jiayudebug snippet end-------------
