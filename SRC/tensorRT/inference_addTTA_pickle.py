'''
Author: your name
Date: 2020-08-11 05:02:51
LastEditTime: 2020-08-16 13:19:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/SRC/eval_model/extract_embedding_addTTA_pickle.py
'''

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms
from PIL import Image
import time
import argparse

def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

def do_inference(context, batch_size, input, output, d_input,d_output):
    # print('1================================')

    # output_shape=(len(input),output_shape[0])
    # output = np.empty(output_shape, dtype=np.float32)
    # print('2================================')

    # d_input = cuda.mem_alloc(args.batch_size * input.size * input.dtype.itemsize)
    # d_output = cuda.mem_alloc(args.batch_size * output.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]
    # print(type(engine),type(context))
    # print('3================================')

    # pycuda操作缓冲区
    stream = cuda.Stream()
    # 将输入数据放入device
    cuda.memcpy_htod_async(d_input, input, stream)

    # start = time.time()
    # 执行模型
    context.execute_async(batch_size, bindings, stream.handle, None)
    # 将预测结果从从缓冲区取出
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # end = time.time()

    # 线程同步
    stream.synchronize()
    # cuda.free(d_input)
    # cuda.free(d_input)
    # d_input.free()
    # d_output.free()
    # #
    # print("\nTensorRT {} test:".format(engine_path.split('/')[-1].split('.')[0]))
    # print("output:", output)
    # print("time cost:", end - start)
    # print(output.shape)
    return output

def get_shape(engine):
    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
        else:
            output_shape = engine.get_binding_shape(binding)
    return input_shape, output_shape


def extract_embedding(args,dataloader,engine,context):
    # parser = argparse.ArgumentParser(description = "TensorRT do inference")
    # parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    # parser.add_argument("--img_path", type=str, default='test_image/1.jpg', help='cache_file')
    # parser.add_argument("--engine_file_path", type=str, default='my_files/test.engine', help='engine_file_path')
    # args = parser.parse_args()

    # engine_path = args.engine_file_path
    # engine = loadEngine2TensorRT(engine_path)
    # context = engine.create_execution_context()


    
    features_list = []*len(dataloader)
    filename_list = []*len(dataloader)
    data_dict={}
    input_shape, output_shape = get_shape(engine)
    output_shape=(args.batch_size,output_shape[0])
    output = np.empty(output_shape, dtype=np.float32)
    full=True
    start=True
    for image, filename in tqdm(dataloader):
        # img = Image.open(args.img_path)
        if(len(image)==args.batch_size):
            if start:
                input=image.numpy()
                # 分配内存
                d_input = cuda.mem_alloc(args.batch_size * input.size * input.dtype.itemsize)
                d_output = cuda.mem_alloc(args.batch_size * output.size * output.dtype.itemsize)
                start=False
            features_1=do_inference(context, args.batch_size, image.numpy(),output,d_input,d_output)

            # 水平
            features_2=do_inference(context, args.batch_size, torch.flip(image,[-1]).numpy(), output,d_input,d_output)

            # 垂直
            features_3=do_inference(context, args.batch_size, torch.flip(image,[-2]).numpy(), output,d_input,d_output)
            
            image_t=image.transpose(2,3)
            # 顺时针90
            features_4=do_inference(context, args.batch_size, torch.flip(image_t,[-1]).numpy(), output,d_input,d_output)

            # 逆时针90
            features_5=do_inference(context, args.batch_size, torch.flip(image_t,[-2]).numpy(), output,d_input,d_output)

            features = features_1+features_2+features_3+features_4+features_5

            # print('1=======================')
            # print(image.shape)
            # print(features.shape)
            features_list.append(features)
            filename_list.append(filename)
        else:
            batch=len(image)
            tenor_tmp=image[:1,...].clone()
            for i in range(args.batch_size-batch):
                image=torch.cat((image,tenor_tmp),dim=0)
            if start:
                input=image.numpy()
                # 分配内存
                d_input = cuda.mem_alloc(args.batch_size * input.size * input.dtype.itemsize)
                d_output = cuda.mem_alloc(args.batch_size * output.size * output.dtype.itemsize)
                start=False

            features_1=do_inference(context, args.batch_size, image.numpy(), output, d_input,d_output)

            # 水平
            features_2=do_inference(context, args.batch_size, torch.flip(image,[-1]).numpy(), output,d_input,d_output)

            # 垂直
            features_3=do_inference(context, args.batch_size, torch.flip(image,[-2]).numpy(), output,d_input,d_output)
            
            image_t=image.transpose(2,3)
            # 顺时针90
            features_4=do_inference(context, args.batch_size, torch.flip(image_t,[-1]).numpy(), output,d_input,d_output)

            # 逆时针90
            features_5=do_inference(context, args.batch_size, torch.flip(image_t,[-2]).numpy(), output,d_input,d_output)

            features = features_1+features_2+features_3+features_4+features_5

            # print('2=======================')
            # print(batch)
            # print(image.shape)
            # print(features.shape)
            features_list.append(features[:batch])
            filename_list.append(filename)

    np_filename = np.concatenate(filename_list)
    np_features = np.concatenate(features_list).astype(np.float32)
    data_dict["fname"]=np_filename
    data_dict["data"]=np_features

    # model.eval()
    # features_list = []*len(dataloader)
    # filename_list = []*len(dataloader)
    # data_dict={}
    # with torch.no_grad():
    #     for image, filename in tqdm(dataloader):

    #         image = image.cuda()

    #         features_1 = model(image.clone(), extract_embedding=True)
    #         features_1 = features_1.cpu().data.numpy().astype(np.float32)

    #         # 水平
    #         features_2 = model(torch.flip(image.clone(),[-1]), extract_embedding=True)
    #         features_2 = features_2.cpu().data.numpy().astype(np.float32)

    #         # 垂直
    #         features_3 = model(torch.flip(image.clone(),[-2]), extract_embedding=True)
    #         features_3 = features_3.cpu().data.numpy().astype(np.float32)
            
    #         # 顺时针90
    #         features_4 = model(torch.flip(image.clone().transpose(2,3),[-1]), extract_embedding=True)
    #         features_4 = features_4.cpu().data.numpy().astype(np.float32)

    #         # 逆时针90
    #         features_5 = model(torch.flip(image.clone().transpose(2,3),[-2]), extract_embedding=True)
    #         features_5 = features_5.cpu().data.numpy().astype(np.float32)

    #         features = features_1+features_2+features_3+features_4+features_5
    #         # import pdb; pdb.set_trace()
    #         features_list.append(features)
    #         filename_list.append(filename)
    #         # data_dict[filename]=list(features)
            
    #     np_filename = np.concatenate(filename_list)
    #     np_features = np.concatenate(features_list).astype(np.float32)
    # data_dict["fname"]=np_filename
    # data_dict["data"]=np_features



    return data_dict