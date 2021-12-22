import h5py
import numpy as np
import torch
from swin import SwinTransformer


torch_weights = "weights/swin_tiny_patch4_window7_224.pth"
# torch_weights = "weights/swin_base_patch4_window7_224_22k.pth"
# torch_weights = "weights/swin_base_patch4_window12_384_22kto1k.pth"
n_stages = 4
emb_dim = 96             # T-96, B-128
num_layers = [2,2,6,2]   # T-[2,2,6,2] , B-[2,2,18,2]
num_heads = [3,6,12,24]  # T-[3,6,12,24], B-[4,8,16,32]
window_size = 7          # 12
n_classes = 1000         # 21k-21841, 1k-1000

state_dict = torch.load(torch_weights, map_location='cpu')
model_weights = state_dict['model']

# aggregate
print('------------- patch embedding --------------')
patchEmb_weights = {k:v for k,v in model_weights.items() if 'patch_embed' in k}
print({k:v.shape for k,v in patchEmb_weights.items()})    # conv-bias, norm-bias


for i in range(n_stages):
    print('------------- layer %d --------------' % i)
    for b in range(num_layers[i]):
        print('------------- block %d --------------' % b)
        block_weights = {k:v for k,v in model_weights.items() if 'layers.%d.blocks.%d' % (i,b) in k}
        print({k:v.shape for k,v in block_weights.items()})
    print('------------- patchmerging --------------')
    block_weights = {k:v for k,v in model_weights.items() if 'layers.%d.downsample' % (i) in k}
    print({k:v.shape for k,v in block_weights.items()})


print('------------ last norm --------------')
Norm_weights = {k:v for k,v in model_weights.items() if 'norm.' in k and 'layers' not in k and 'patch' not in k}
print({k:v.shape for k,v in Norm_weights.items()})


print('------------ head ----------------')
Head_weights = {k:v for k,v in model_weights.items() if 'head' in k}
print({k:v.shape for k,v in Head_weights.items()})


### special varaibles
# attn.relative_position_index [49, 49]    # fixed, (7x7,7x7), range [0,12]
# attn_mask [16, 49, 49]   # nW 7x7 7x7
# attn.relative_position_bias_table [169, 6]    # trainable, (13x13,n_heads)
###


stage_ref = {}
for stage_idx, n_blocks in enumerate(num_layers):
    for i in range(n_blocks):
        STB_idx = (sum(num_layers[:stage_idx])+i) // 2
        stage_ref[STB_idx] = stage_idx
print(stage_ref)


# keras_model_swin = SwinTransformer(patch_size=4, emb_dim=96, n_classes=1000, num_layers=[2,2,6,2],
#                                     num_heads=[3,6,12,24], window_size=7, qkv_bias=True, qk_scale=None,
#                                     mlp_ratio=4, attn_drop=0., ffn_drop=0., residual_drop=0.2)
keras_model_swin = SwinTransformer(patch_size=4, emb_dim=emb_dim, n_classes=n_classes, num_layers=num_layers,
                                    num_heads=num_heads, window_size=window_size, residual_drop=0.5)
for layer in keras_model_swin.layers:
    if not layer.get_weights():
        continue
    print('------------- layer name: %s -------------' % layer.name)
    # for w in layer.get_weights():
    #     print(w.shape)
    if layer.name == 'conv2d_1':
        conv_weights = np.transpose(patchEmb_weights['patch_embed.proj.weight'], (2,3,1,0))   # [out,in,k,k] -> [k,k,in,out]
        conv_bias = patchEmb_weights['patch_embed.proj.bias']
        layer.set_weights([conv_weights, conv_bias])
    elif layer.name == 'layer_normalization_1':
        norm_weights = patchEmb_weights['patch_embed.norm.weight']
        norm_bias = patchEmb_weights['patch_embed.norm.bias']
        layer.set_weights([norm_weights, norm_bias])
    elif 'STB' in layer.name:   # a pair of  WSA-block  +  SWSA-block
        block_idx = int(layer.name.split('_')[-1])
        stage_id = stage_ref[block_idx]
        block_id = block_idx*2 - sum(num_layers[:stage_id])
        print('stage & block', stage_id, [block_id, block_id+1])
        # block0: WSA
        block_weights = {k:v for k,v in model_weights.items() if 'layers.%d.blocks.%d' % (stage_id,block_id) in k and 'index' not in k and 'mask' not in k}
        block_weights = [np.transpose(v,(1,0)) if 'weight' in k and 'norm' not in k else v
                                                for k,v in block_weights.items()]     # [out,in] -> [in,out]
        block_weights.insert(6, block_weights[-1])
        block_weights = block_weights[:-1]
        # block1: SWSA
        block1_weights = {k:v for k,v in model_weights.items() if 'layers.%d.blocks.%d' % (stage_id,block_id+1) in k and 'index' not in k and 'mask' not in k}
        block1_weights = [np.transpose(v,(1,0)) if 'weight' in k and 'norm' not in k else v
                                                for k,v in block1_weights.items()]
        block1_weights.insert(6, block1_weights[-1])
        block1_weights = block1_weights[:-1]
        block_weights += block1_weights
        layer.set_weights(block_weights)

        # for sub_l in layer.layers:
        #     print(sub_l.name)
            # LN: gamma,bias
            # WMH: Dense,Dense
            # LN: gamma,bias
            # FFN: Dense, Dense
            # LN
            # SWMH
            # LN
            # FFN

    elif 'PatchMerging' in layer.name:
        # LN: gamma, bias
        # Dense: weight
        stage_id = int(layer.name.split('_')[-1])
        block_weights = {k:v for k,v in model_weights.items() if 'layers.%d.downsample' % (stage_id) in k}
        block_weights = [np.transpose(v,(1,0)) if 'reduction' in k else v for k,v in block_weights.items()] 
        # print([w.shape for w in block1_weights])
        # print([l.shape for l in layer.get_weights()])
        layer.set_weights(block_weights)

    elif 'layer_norm' in layer.name:
        # last layernorm
        norm_weights = Norm_weights['norm.weight']
        norm_bias = Norm_weights['norm.bias']
        layer.set_weights([norm_weights, norm_bias])

    elif 'dense' in layer.name:
        # head
        dense_weights = np.transpose(Head_weights['head.weight'], (1,0))
        dense_bias = Head_weights['head.bias']
        layer.set_weights([dense_weights, dense_bias])

keras_model_swin.save_weights('weights/swin_tiny_patch4_window7_224.h5')








