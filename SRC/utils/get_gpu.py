
import GPUtil
import os
from random import shuffle


def get_gpu(num_of_gpu):
    # 获取GPU，并且设置环境变量，并且返回可以使用的GPU编号
    # 注意，最好在运行这个函数之后立刻占用获取的GPU，否则GPU可能被其他程序占用，导致获取失败。

    gpu_ids_avail = GPUtil.getAvailable(maxMemory=0.02, limit=8)

    if len(gpu_ids_avail) < num_of_gpu:
        #如果正确的提交了任务，不应该获取不到足够的GPU
        #queue.pl -q GPU_QUEUE --num-threads 4  #需要占用4个GPU
        print("not enough GPU")
        return []

    shuffle(gpu_ids_avail)
    CUDA_VISIBLE_DEVICES=""
    for gpu in gpu_ids_avail[:num_of_gpu]:
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES+str(gpu)+","

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES[:-1]
    print("using GPU:",CUDA_VISIBLE_DEVICES[:-1])

    return list(range(num_of_gpu))


if __name__ == '__main__':
    aa=get_gpu(4)
    print(os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(aa)


