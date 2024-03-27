#!/usr/bin/bash
## Set model_dir to save_inference_dir in export_model.py config file
python tools/export_model.py -c=ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_small_x0_75.yaml

paddle2onnx --model_dir ./output/ocr_game/v11_224x224_add_new_game/inference \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file output/ocr_game/v11_224x224_add_new_game/onnx/model.onnx \
            --opset_version 16 \
            --input_shape_dict="{'x':[-1,3,-1,-1]}" \
            --enable_onnx_checker True 