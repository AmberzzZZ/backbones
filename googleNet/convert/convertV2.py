import pandas as pd
import pickle


def aggregate_by_field(weights, idx):
    weights_by_field = {}
    for key, value in weights.items():
        loc = key.split('/')[idx]
        if loc not in weights_by_field.keys():
            weights_by_field[loc] = {}
        weights_by_field[loc][key] = value
    return weights_by_field


if __name__ == '__main__':

    weights = pd.read_pickle("official_effv2s.pkl")   # {name:weight} dict
    print(len(weights))
    print(weights.keys())

    # aggregate by location
    weights_by_loc = aggregate_by_field(weights, 1)
    # print(weights_by_loc.keys())

    # aggregate by stages
    weights_overall = {}
    n_blocks = [2,4,4,6,9,5]

    # stem: conv-bn
    print("====================================stem====================================")
    weights_by_layer = aggregate_by_field(weights_by_loc['stem'], 2)
    print(weights_by_layer.keys())
    weights_keras = []
    for layer_name, layer_params in weights_by_layer.items():
        print(layer_name)
        print([[k,v.shape] for k,v in layer_params.items()])
        weights_keras.append(list(layer_params.values()))
    print(len(weights_keras))
    weights_overall['stem'] = weights_keras

    # stage1: fused-MB, 3x3conv-bn
    # stage23: fused-MB, 3x3conv-bn-1x1conv-bn
    for s in range(0,3):
        print("===============================stage: %d==================================" % s)
        start = sum(n_blocks[:s])
        end = sum(n_blocks[:s+1])
        blocks = ['blocks_%d' % i for i in range(start,end)]
        for b in blocks:
            weights_keras = []
            weights_by_layer = aggregate_by_field(weights_by_loc[b], 2)
            print(weights_by_layer.keys())
            for layer_name, layer_params in weights_by_layer.items():
                print(layer_name)
                print([[k,v.shape] for k,v in layer_params.items()])
                weights_keras.append(list(layer_params.values()))
            print(len(weights_keras))
            weights_overall['stage_%d_%s' % (s,b)] = weights_keras

    # stage456: MB, conv-bn-dconv-bn-se-pconv-bn
    # se: conv-bias-conv-bias
    for s in range(3,6):
        print("===============================stage: %d==================================" % s)
        start = sum(n_blocks[:s])
        end = sum(n_blocks[:s+1])
        blocks = ['blocks_%d' % i for i in range(start,end)]
        for b in blocks:
            weights_keras = []
            weights_by_layer = aggregate_by_field(weights_by_loc[b], 2)
            print(weights_by_layer.keys())
            for layer_name, layer_params in weights_by_layer.items():
                print(layer_name)
                print([[k,v.shape] for k,v in layer_params.items()])
                if layer_name=='se':
                    conv_layers = aggregate_by_field(layer_params, 3)
                    for layer in conv_layers.values():
                        weights_keras.append(list(layer.values()))
                else:
                    weights_keras.append(list(layer_params.values()))
            print(len(weights_keras))
            weights_overall['stage_%d_%s' % (s,b)] = weights_keras

    # top: conv-bn
    print("====================================head====================================")
    weights_by_layer = aggregate_by_field(weights_by_loc['head'], 2)
    print(weights_by_layer.keys())
    weights_keras = []
    for layer_name, layer_params in weights_by_layer.items():
        print(layer_name)
        print([[k,v.shape] for k,v in layer_params.items()])
        weights_keras.append(list(layer_params.values()))
    print(len(weights_keras))
    weights_overall['stem'] = weights_keras

    with open("serialized_weights_by_layer.pkl", 'wb') as f:
        pickle.dump(weights_overall, f)



















