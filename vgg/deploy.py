from repvgg import RepVGG_B1g4, RepVGGStage, RepVGGBlock
import h5py

weights = h5py.File("weights/RepVGG-B1g4-train.h5", 'r')
# for k,v in weights.items():
#     print(k,v)


model = RepVGG_B1g4(input_shape=(224,224,3), num_classes=1000, test_mode=True)
# model.summary()
# model.load_weights("weights/RepVGG-B1g4-train.h5", skip_mismatch=True, by_name=True)
# Skipping loading of weights for layer repvggblock_1 due to mismatch in number of weights (2 vs 10)
# Skipping loading of weights for layer repvggstage_1 due to mismatch in number of weights (8 vs 52).
# Skipping loading of weights for layer repvggstage_2 due to mismatch in number of weights (12 vs 80).
# Skipping loading of weights for layer repvggstage_3 due to mismatch in number of weights (32 vs 220).
# Skipping loading of weights for layer repvggstage_4 due to mismatch in number of weights (2 vs 10).
model.load_weights("weights/RepVGG-B1g4-deploy.h5")


num_blocks = [4,6,16,1]
width_multiplier = [2,2,2,4]

for layer in model.layers:

    if not layer.weights:
        continue

    print('deploy entry: ', layer.name)

    if isinstance(layer, RepVGGStage):
        layer.switch_to_deploy(weights[layer.name][layer.name])

    elif isinstance(layer, RepVGGBlock):
        layer.switch_to_deploy(weights[layer.name][layer.name])
        print(layer.weights)
    else:
        # dense
        w = weights[layer.name][layer.name]
        weight = w['kernel:0']
        bias = w['bias:0']
        layer.set_weights([weight, bias])

model.save_weights("weights/RepVGG-B1g4-deploy.h5")



