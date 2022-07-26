import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import PIL
from repvgg import RepVGGBlock
from keras.models import Model
from keras.layers import Input
from repvgg import RepVGG_B1g4


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
    x = RepVGGBlock(64, kernel_size=3, stride=2, groups=1, use_se=False, test_mode=False)(inpt)

    model = Model(inpt, x)
    model.load_weights("weights/RepVGG-B1g4-train.h5", by_name=True, skip_mismatch=True)

    return model


model = tmp_model()
# model = RepVGG_B1g4()
# model.load_weights("weights/RepVGG-B1g4-train.h5")

inpt = np.transpose(img_arr, (1,2,0))
inpt = np.expand_dims(inpt, axis=0)
preds = model.predict(inpt)[0]     # [h,w,64]
print(np.max(preds), np.min(preds), np.argmax(preds))
for i in range(64):
    cv2.imwrite('tmp/tmp%i.png' % i, preds[:,:,i]*25)



