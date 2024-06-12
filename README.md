# 下载obs
1. linux AMD
```
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz
```
2. 在软件包所在目录，执行以下解压命令。
```
tar -xzvf obsutil_linux_amd64.tar.gz
```
3. 进入obsutil所在目录，执行以下命令，为obsutil增加可执行权限。
```
chmod 755 obsutil
```
4. 继续在目录中执行以下命令，如果能顺利返回obsutil版本号，说明安装成功。
```
./obsutil version
```
5. 使用AK、SK、endpoint进行初始化配置（微信给你）：
```
./obsutil config -i=ak -k=sk -e=endpoint
```
6. 下载docker镜像，数据，模型
```

./obsutil cp -f obs://lxy-obs/screenshots_16w.zip /path/to/data/

./obsutil cp -f obs://lxy-obs/train_reverse_html_qa_imgtoken_164600_minicpm.json /path/to/data/

./obsutil cp -f obs://lxy-obs/train_reverse_html_qa_imgtoken_1000_minicpm_test.json /path/to/data/

./obsutil cp -r -f obs://huawei-b127/MiniCPM-Llama3-V-2_5/ /path/to/data/
```
7. 解压screenshots数据于/path/to/data/
```
unzip screenshots_16w.zip
```
8. docker镜像
复制Dockerfile
```
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list &&\
    apt-get update && apt-get install -y \
    git \
    wget \
    bzip2 

RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda3/miniconda.sh

ENV PATH /root/miniconda3/bin:$PATH

RUN conda create -n MiniCPM-V python=3.10 -y
SHELL ["conda", "run", "-n", "MiniCPM-V", "/bin/bash", "-c"]

RUN git clone https://github.com/OpenBMB/MiniCPM-V.git /root/MiniCPM-V
WORKDIR /root/MiniCPM-V
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install deepspeed
RUN pip install peft
RUN pip install tensorboard

ENV CONDA_DEFAULT_ENV=MiniCPM-V
ENV PATH /root/miniconda3/envs/MiniCPM-V/bin:$PATH

CMD [ "bash" ]
```
构建镜像
```
docker build -t minicmp-v:latest .
```
9. 启动容器
```
docker run --gpus all -dit --shm-size=[内存大小]g --net=host -v /path/to/data/:/data  --name=minicpm 镜像id
```
# 训练
1. 进入容器/root/MiniCPM-V/finetune
2. 覆盖finetune_lora.sh为
可根据卡数调整GPUS_PER_NODE=8
可根据显存大小调整per_device_train_batch_size

若是8卡a100 80G，则建议直接复制下面代码
```
#!/bin/bash

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/data/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/data/train_reverse_html_qa_imgtoken_164600_minicpm.json"
EVAL_DATA="/data/train_reverse_html_qa_imgtoken_1000_minicpm_test.json"
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj)" \
    --model_max_length 4096 \
    --max_slice_nums 9 \
    --max_steps 26000 \
    --eval_steps 1000 \
    --output_dir output/output_minicpmv2_lora \
    --logging_dir output/output_minicpmv2_lora \
    --logging_strategy "steps" \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard" # wandb
```
2. 在/data/MiniCPM-V/finetune下面运行shell脚本
```
sh finetune_lora.sh
```
