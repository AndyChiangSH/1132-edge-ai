from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
# Score: 19.091586303710983
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=32)
    
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q4_config
        quant_config[f'blocks.{i}.attn.proj'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_config
        
    quant_config['head'] = q4_config
    
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q8_config = BaseQuantizeConfig(nbits=8, group_size=128)  # 8-bit quantization
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_config
    
    return quant_config