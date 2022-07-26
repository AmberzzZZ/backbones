import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import PIL
from repvgg import RepVGGBlock
from keras.models import Model
from keras.layers import Input
from repvgg import RepVGG_B1g4, g4_map, RepVGGStage


trans = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
    ])

img = PIL.Image.open("/Users/amber/Downloads/cat.jpeg")
img = trans(img)     # tensor
print(img.shape)

img_arr = img.numpy()
print(img_arr[:,0,0])


def tmp_model():
    inpt = Input((224,224,3))
    # stem: single RepVGG block
    x = RepVGGBlock(64, kernel_size=3, stride=2, groups=1, use_se=False, test_mode=True)(inpt)
    # stage1
    x = RepVGGStage(4, strides=2, n_filters=128, group_dict=g4_map, layer_idx=1, use_se=False, test_mode=True)(x)

    model = Model(inpt, x)
    model.load_weights("weights/RepVGG-B1g4-deploy.h5", by_name=True, skip_mismatch=True)
    # model.load_weights("weights/RepVGG-B1g4-train.h5", by_name=True, skip_mismatch=True)

    return model


inpt = np.transpose(img_arr, (1,2,0))
inpt = np.expand_dims(inpt, axis=0)

# model = tmp_model()
# preds = model.predict(inpt)[0]     # [h,w,64]
# print(np.max(preds), np.min(preds), np.argmax(preds))
# for i in range(64):
#     cv2.imwrite('tmp/tmp%i.png' % i, preds[:,:,i]*25)

# model = RepVGG_B1g4(input_shape=(224,224,3), num_classes=1000, test_mode=True)
# model.load_weights("weights/RepVGG-B1g4-deploy.h5")
model = RepVGG_B1g4(input_shape=(224,224,3), num_classes=1000, test_mode=False)
model.load_weights("weights/RepVGG-B1g4-train.h5")
probs = model.predict(inpt)[0]     # [1000,]
print(np.argmax(probs), np.max(probs))




