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
8. 编写Dockerfile
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

WORKDIR /root/MiniCPM-V/finetune
RUN rm -f finetune_lora.sh
RUN printf '#!/bin/bash\n\n\
GPUS_PER_NODE=8\n\
NNODES=1\n\
NODE_RANK=0\n\
MASTER_ADDR=localhost\n\
MASTER_PORT=6001\n\n\
MODEL="/data/MiniCPM-Llama3-V-2_5"\n\
DATA="/data/train_reverse_html_qa_imgtoken_164600_minicpm.json"\n\
EVAL_DATA="/data/train_reverse_html_qa_imgtoken_1000_minicpm_test.json"\n\
LLM_TYPE="llama3"\n\n\
DISTRIBUTED_ARGS="\n\
    --nproc_per_node $GPUS_PER_NODE \\\n\
    --nnodes $NNODES \\\n\
    --node_rank $NODE_RANK \\\n\
    --master_addr $MASTER_ADDR \\\n\
    --master_port $MASTER_PORT\n\
"\n\
torchrun $DISTRIBUTED_ARGS finetune.py  \\\n\
    --model_name_or_path $MODEL \\\n\
    --llm_type $LLM_TYPE \\\n\
    --data_path $DATA \\\n\
    --eval_data_path $EVAL_DATA \\\n\
    --remove_unused_columns false \\\n\
    --label_names "labels" \\\n\
    --prediction_loss_only false \\\n\
    --bf16 false \\\n\
    --bf16_full_eval false \\\n\
    --fp16 true \\\n\
    --fp16_full_eval true \\\n\
    --do_train \\\n\
    --do_eval \\\n\
    --tune_vision true \\\n\
    --tune_llm false \\\n\
    --use_lora true \\\n\
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj)" \\\n\
    --model_max_length 4096 \\\n\
    --max_slice_nums 9 \\\n\
    --max_steps 26000 \\\n\
    --eval_steps 1000 \\\n\
    --output_dir output/output_minicpmv2_lora \\\n\
    --logging_dir output/output_minicpmv2_lora \\\n\
    --logging_strategy "steps" \\\n\
    --per_device_train_batch_size 6 \\\n\
    --per_device_eval_batch_size 1 \\\n\
    --gradient_accumulation_steps 1 \\\n\
    --evaluation_strategy "steps" \\\n\
    --save_strategy "steps" \\\n\
    --save_steps 1000 \\\n\
    --save_total_limit 10 \\\n\
    --learning_rate 1e-6 \\\n\
    --weight_decay 0.1 \\\n\
    --adam_beta2 0.95 \\\n\
    --warmup_ratio 0.01 \\\n\
    --lr_scheduler_type "cosine" \\\n\
    --logging_steps 1 \\\n\
    --gradient_checkpointing true \\\n\
    --deepspeed ds_config_zero2.json \\\n\
    --report_to "tensorboard"' > finetune_lora.sh


ENV CONDA_DEFAULT_ENV=MiniCPM-V
ENV PATH /root/miniconda3/envs/MiniCPM-V/bin:$PATH

CMD [ "bash" ]
```

9. 构建镜像
```
docker build -t minicmp-v:latest .
```
10. 启动容器
```
docker run --gpus all -dit --shm-size=[内存大小]g --net=host -v /path/to/data/:/data  --name=minicpm 镜像id
```
# 训练
1. 进入容器/root/MiniCPM-V/finetune
2. 修改finetune_lora.sh参数
- 可根据卡数调整GPUS_PER_NODE
- 可根据显存大小调整per_device_train_batch_size

- 若是8卡a100 80G，则建议直接运行下面代码
- 若是8卡a100 40G，则建议修改per_device_train_batch_size=3，max_steps=54000

3. 在/root/MiniCPM-V/finetune下面运行shell脚本
```
sh finetune_lora.sh
```
