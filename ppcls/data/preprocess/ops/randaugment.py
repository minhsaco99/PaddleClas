# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is based on https://github.com/heartInsert/randaugment
# reference: https://arxiv.org/abs/1909.13719

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import imgaug as ia
from imgaug import parameters as iap
import imgaug.augmenters as iaa
import random
import cv2
import gc
import os

class RandAugment(object):
    def __init__(self, num_layers=2, magnitude=5, fillcolor=(128, 128, 128)):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.max_level = 10

        abso_level = self.magnitude / self.max_level
        self.level_map = {
            "shearX": 0.3 * abso_level,
            "shearY": 0.3 * abso_level,
            "translateX": 150.0 / 331 * abso_level,
            "translateY": 150.0 / 331 * abso_level,
            "rotate": 30 * abso_level,
            "color": 0.9 * abso_level,
            "posterize": int(4.0 * abso_level),
            "solarize": 256.0 * abso_level,
            "contrast": 0.9 * abso_level,
            "sharpness": 0.9 * abso_level,
            "brightness": 0.9 * abso_level,
            "autocontrast": 0,
            "equalize": 0,
            "invert": 0
        }

        # from https://stackoverflow.com/questions/5252170/
        # specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot,
                                   Image.new("RGBA", rot.size, (128, ) * 4),
                                   rot).convert(img.mode)

        rnd_ch_op = random.choice

        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * rnd_ch_op([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * rnd_ch_op([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * rnd_ch_op([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * rnd_ch_op([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * rnd_ch_op([-1, 1])),
            "posterize": lambda img, magnitude:
                ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude:
                ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude:
                ImageEnhance.Contrast(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "sharpness": lambda img, magnitude:
                ImageEnhance.Sharpness(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "brightness": lambda img, magnitude:
                ImageEnhance.Brightness(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "autocontrast": lambda img, magnitude:
                ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def __call__(self, img):
        avaiable_op_names = list(self.level_map.keys())
        for layer_num in range(self.num_layers):
            op_name = np.random.choice(avaiable_op_names)
            img = self.func[op_name](img, self.level_map[op_name])
        return img

class CustomCLSAug(object):
    def __init__(self, debug=False, save_img_path=None, n_save_imgs=2000, **kwargs) -> None:
        self.augmentation_pipeline = iaa.Sequential([
                    # # Augment color and brightness
                    iaa.Sometimes(0.8,
                        iaa.OneOf([
                            iaa.Add((-30, 50)),
                            iaa.AddToHue((-100, 100)),
                            iaa.AddToBrightness((-30, 30)),
                            iaa.AddToSaturation((-50, 50)),
                            iaa.AddToHueAndSaturation((-40, 40)),
                            iaa.ChangeColorTemperature((4000, 11000)),
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            iaa.ChannelShuffle(1),
                            iaa.Invert(0.1, per_channel=True),
                            iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-30, 30]), 20)), start_at=(0, 0.25), end_at=(0.75, 1)),
                            iaa.BlendAlphaHorizontalLinearGradient(iaa.Add(iap.Normal(iap.Choice([-30, 30]), 20)), start_at=(0.75, 1), end_at=(0, 0.25)),
                            iaa.MultiplyBrightness((0.75, 1.25)),
                            iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-20, 20)),
                            iaa.Multiply((0.85, 1.10)),
                            # Change contrast
                            iaa.SigmoidContrast(gain= (3, 7), cutoff=(0.3, 0.6)),
                            iaa.LinearContrast((0.7, 1.3)),
                            iaa.GammaContrast((0.7, 1.5)),
                            iaa.LogContrast(gain=(0.7, 1.3)),
                            iaa.pillike.Autocontrast((2, 5)),
                            iaa.Emboss(alpha=(0.1, 0.5), strength=(0.8, 1.2)),
                            ]),
                        ), 
                    # # # Noise and change background
                    iaa.Sometimes(0.6,
                        iaa.OneOf([
                            iaa.pillike.FilterSmoothMore(),
                            iaa.imgcorruptlike.Spatter(severity=(1,3)),
                            iaa.pillike.EnhanceSharpness(),
                            iaa.AdditiveGaussianNoise(scale=(0.02 * 255, 0.05 * 255)),
                            iaa.AdditiveGaussianNoise(scale=(0.02 * 255, 0.05 * 255), per_channel=True),
                            iaa.SaltAndPepper(p=(0.001, 0.01)),
                            iaa.Sharpen(alpha=(0.1, 0.5)),
                            iaa.MultiplyElementwise((0.9, 1.1), per_channel=0.5),
                            iaa.GaussianBlur(sigma=(0.5, 2)),
                            iaa.AverageBlur(k=(3, 7)),
                            iaa.MotionBlur(k=(3, 9), angle=(-180, 180)),
                            iaa.Dropout((0.001, 0.01), per_channel=True),
                            iaa.ElasticTransformation(alpha=(1, 10), sigma=(2, 4)),
                            iaa.CoarseDropout(0.02, size_percent=(0.01, 0.3), per_channel=True),
                            ])
                        ),
                    # # # Transform
                    iaa.Sometimes(0.4,
                        iaa.OneOf([
                            iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                            iaa.Rotate((-2, 2)),
                            iaa.Rotate((-3, 3), cval=(0,255), mode=ia.ALL),
                            # iaa.Crop((1,3)),
                            iaa.ShearX((-5, 5), mode=ia.ALL),
                            iaa.ShearX((-9, 9), mode=ia.ALL),
                            iaa.ShearY((-2, 2), mode=ia.ALL, cval=(0, 255)),
                            iaa.ShearY((-4, 4), mode=ia.ALL, cval=(0, 255)),
                            iaa.Affine(translate_px=(-2,4), mode=ia.ALL),
                            iaa.Affine(translate_px=(-4,4), mode=ia.ALL),
                        ])
                    ),
                    
                    # # compress image
                    iaa.Sometimes(0.03,
                        iaa.OneOf([
                            iaa.JpegCompression(compression=(50, 80)),
                            iaa.imgcorruptlike.Pixelate(severity=(1)),
                            iaa.UniformColorQuantization((10, 120)),
                        ])
                    )
                ])
        self.debug = True
        self.save_img_path = save_img_path
        self.n_save_imgs = n_save_imgs
        self.save_img_count = 0
        self.gc_count = 0
        if self.debug:
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            assert save_img_path != None, "Use debug must pass the save_img_path parameter"

    def __call__(self, img):
        h, w, _ = img.shape
        if random.random() <= 0.85:
            img = self.augmentation_pipeline.augment(image=img.astype(np.uint8))
        if self.debug and self.save_img_count < self.n_save_imgs:
            cv2.imwrite(os.path.join(self.save_img_path, str(self.save_img_count) + '.jpg'), img.astype(np.uint8)[:,:,::-1])
            self.save_img_count += 1
        self.gc_count += 1
        if self.gc_count % 10000 == 0:
            gc.collect()
            self.gc_count = 0
        return img