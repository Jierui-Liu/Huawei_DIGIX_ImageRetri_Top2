# -*- coding: utf-8 -*-
"""
Created on 2020.06.11

@author: LWS
"""
import tensorrt as trt
import os

def ONNX2TRT(args, calib=None):
    ''' convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''

    assert args.mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(G_LOGGER) as builder, builder.create_network() as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:

        builder.max_batch_size = args.batch_size
        builder.max_workspace_size = 1 << 32
        # builder.max_workspace_size = 1 << 30
        if args.mode == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
        elif args.mode == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True

        if not os.path.exists(args.onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(args.onnx_file_path))
            exit(0)

        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('============================something wrong============================')
                print('network.num_layers',network.num_layers)
                last_layer = network.get_layer(network.num_layers - 4)
                print(last_layer,last_layer.get_output(0).shape)
                last_layer = network.get_layer(network.num_layers - 3)
                print(last_layer,last_layer.get_output(0).shape)
                last_layer = network.get_layer(network.num_layers - 2)
                print(last_layer,last_layer.get_output(0).shape)
                last_layer = network.get_layer(network.num_layers - 1)
                print(last_layer,last_layer.get_output(0).shape)

                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(0)
            # parser.parse(model.read())
            print('network.num_layers',network.num_layers)
            last_layer = network.get_layer(network.num_layers - 4)
            print(last_layer,last_layer.get_output(0).shape)
            last_layer = network.get_layer(network.num_layers - 3)
            print(last_layer,last_layer.get_output(0).shape)
            last_layer = network.get_layer(network.num_layers - 2)
            print(last_layer,last_layer.get_output(0).shape)
            last_layer = network.get_layer(network.num_layers - 1)
            print(last_layer,last_layer.get_output(0).shape)
            # network.mark_output(last_layer.get_output(0))
        print('Completed parsing of ONNX file')
        # print(network.get_input(0).shape,'================================')
        # print(network.get_output(0).shape,'================================')

        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print(type(engine),'================================')
        
        print("Created engine success! ")

        # 保存计划文件
        print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
        with open(args.engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print('Engine file has already saved to {}!'.format(args.engine_file_path))
        return engine


def loadEngine2TensorRT(filepath):
    '''
    通过加载计划文件，构建TensorRT运行引擎
    '''
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
