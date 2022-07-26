import torch
import numpy as np


model = torch.load("weights/RepVGG-B1g4-train.pth", map_location='cpu')   # ordered dict
# print(model.keys())


# stages
for i in range(5):
    print("================= stage %d =================" % i)
    stage_weights = {k:v for k,v in model.items() if 'stage%d' % i in k}
    print([[k,v.shape] for k,v in stage_weights.items()])

# # head
# head_weights = {k:v for k,v in model.items() if 'linear.' in k}
# print(head_weights.keys)


from repvgg import RepVGG, g4_map
# num_blocks = [2,4,14,1]
# width_multiplier = [0.75,0.75,0.75,2.5]
num_blocks = [4,6,16,1]
width_multiplier = [2,2,2,4]
groups = g4_map
keras_model = RepVGG((224,224,3), num_classes=1000, use_se=False, test_mode=False, groups=groups,
                     num_blocks=num_blocks, width_multiplier=width_multiplier)

layer_idx = 0
for layer in keras_model.layers:
    if not layer.weights:
        continue
    print(layer.name)

    if layer.name == 'repvggblock_1':
        # stem: single block, branch3x3 & branch1x1
        # stage_weights = {k:v for k,v in model.items() if 'stage0' in k}
        # bn: [gamma, beta, mean, std]
        g = groups.get(layer_idx, 1)
        b3_weights = [v for k,v in model.items() if 'stage0.rbr_dense' in k and 'num_batches_tracked' not in k]
        for idx,v in enumerate(b3_weights):
            if len(v.shape)>1:  # conv
                out, g_in, k, k = v.shape
                v = np.reshape(v, (g, out//g, g_in, k, k))
                v = np.transpose(v, axes=(0,3,4,2,1))
                b3_weights[idx] = v
        b1_weights = [v for k,v in model.items() if 'stage0.rbr_1x1' in k and 'num_batches_tracked' not in k]
        for idx,v in enumerate(b1_weights):
            if len(v.shape)>1:  # conv
                out, g_in, k, k = v.shape
                v = np.reshape(v, (g, out//g, g_in, k, k))
                v = np.transpose(v, axes=(0,3,4,2,1))
                b1_weights[idx] = v

        # torch_weights = [b3_weights, b1_weights]
        # idx = 0
        # for sub_l in layer.layers:
        #     print(sub_l.name, [[i.name, i.shape] for i in sub_l.weights])
        #     if sub_l.weights:
        #         sub_l.set_weights(torch_weights[idx])
        #         idx += 1
        layer.set_weights(b3_weights + b1_weights)
        layer_idx += 1

    elif 'repvggstage' in layer.name:
        # stage blocks
        stage_id = int(layer.name.split('_')[-1])
        n_blocks = num_blocks[stage_id-1]
        sub_layers = layer.layers
        for block_id, sub_l in zip(range(n_blocks), sub_layers):
            # convbn+convbn / bn+convbn+convbn
            print(sub_l.name, [i.shape for i in sub_l.weights])
            if block_id!=0:
                bn_weights = [v for k,v in model.items() if 'stage%d.%d.rbr_identity' % (stage_id, block_id) in k and 'num_batches_tracked' not in k]
            else:
                bn_weights = []
            g = groups.get(layer_idx, 1)
            b3_weights = [v for k,v in model.items() if 'stage%d.%d.rbr_dense' % (stage_id, block_id) in k and 'num_batches_tracked' not in k]
            for idx,v in enumerate(b3_weights):
                if len(v.shape)>1:  # conv
                    out, g_in, k, k = v.shape
                    v = np.reshape(v, (g, out//g, g_in, k, k))
                    v = np.transpose(v, axes=(0,3,4,2,1))
                    b3_weights[idx] = v
            b1_weights = [v for k,v in model.items() if 'stage%d.%d.rbr_1x1' % (stage_id, block_id) in k and 'num_batches_tracked' not in k]
            for idx,v in enumerate(b1_weights):
                if len(v.shape)>1:  # conv
                    out, g_in, k, k = v.shape
                    v = np.reshape(v, (g, out//g, g_in, k, k))
                    v = np.transpose(v, axes=(0,3,4,2,1))
                    b1_weights[idx] = v

            # torch_weights = [bn_weights,b3_weights,b1_weights]
            # idx = 0
            # while not torch_weights[idx]:
            #     idx += 1
            # for subsub_l in sub_l.layers:
            #     print(subsub_l.name, [i.shape for i in subsub_l.weights])
            #     if subsub_l.weights:
            #         subsub_l.set_weights(torch_weights[idx])
            #         idx += 1
            sub_l.set_weights(bn_weights + b3_weights + b1_weights)
            layer_idx += 1

    else:
        # dense head
        dense_weights = [np.transpose(v) if len(v.shape)>1 else v for k,v in model.items() if 'linear' in k]
        layer.set_weights(dense_weights)

keras_model.save_weights("weights/RepVGG-B1g4-train.h5")



