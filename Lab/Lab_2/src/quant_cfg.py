from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
# Score: 18.72741394042967
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q2_config = BaseQuantizeConfig(nbits=4, group_size=64)
    
    quant_config['head'] = q2_config
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q2_config
        quant_config[f'blocks.{i}.attn.proj'] = q2_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q2_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q2_config
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    print(model)
    
    n_layers = model.config.num_hidden_layers
    
    # Define quantization configurations for different parts of the model
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)  # 4-bit quantization
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)  # 8-bit quantization
    
    # Apply 4-bit quantization to the attention and MLP projections
    for i in range(n_layers):
        # Self-attention projections (Q, K, V, O)
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_config
        
        # MLP projections (Gate, Up, Down)
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_config
    
    # Apply 8-bit quantization to the output layer (lm_head)
    quant_config['lm_head'] = q8_config
        
    return quant_config