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

# 测试
1. 测试数据下载
```
./obsutil cp -r -f obs://huawei-b127/Design2code/Design2Code/ /path/to/data/

./obsutil cp -r -f obs://lxy-obs/test_d2code/ /path/to/data/
```

2. 测试代码
- /root/MiniCPM-V/infer_web.py
```
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from chat import img2base64
import json
import torch
from PIL import Image
import os
import tqdm

def gen_html(img_path, output_dir, model, tokenizer):
    image = Image.open(img_path).convert('RGB')
    question = 'Write the HTML code.'
    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        max_new_tokens=4096,
        # system_prompt='' # pass system_prompt if needed
    )

    with open(os.path.join(output_dir, os.path.basename(img_path).split(".png")[0] + ".html"), "w") as f:
        f.write(res)
    # print(answer)
#processbar
# for _ in tqdm.tqdm(imgs):
#     gen_html(_)

if __name__ == "__main__":
    #增加参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_file", type=str, default="/root/MiniCPM-V/img1.txt")
    parser.add_argument("--output_dir", type=str, default="/root/MiniCPM-V/rs_cpm_finetune/")
    parser.add_argument("--path_to_checkpoint", type=str, default="/root/MiniCPM-V/finetune/output/output_minicpmv2_lora/checkpoint-7000")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.img_file, "r") as f:
        img_files = f.readlines()
        img_files = [img_file.strip() for img_file in img_files]

    path_to_adapter=args.path_to_checkpoint

    model = AutoPeftModelForCausalLM.from_pretrained(
        # path to the output directory
        path_to_adapter,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    vpm_resampler_embedtokens_weight = torch.load(f"{path_to_adapter}/vpm_resampler_embedtokens.pt")
    tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)

    msg = model.load_state_dict(vpm_resampler_embedtokens_weight, strict=False)

    #processbar
    for img_file in tqdm.tqdm(img_files):
        gen_html(img_file, args.output_dir, model, tokenizer)
```
- /root/MiniCPM-V/infer_d2code.sh
  - OUTPUT_DIR="/root/MiniCPM-V/rs_cpm_finetune_sampling_tune_visua_16_1epoch"  修改为输出路径
  - CHECKPOINT="/root/MiniCPM-V/output/output_minicpmv2_lora/checkpoint-xxx"  修改为对应的权重
  - 做以下三个测试，在infer_d2code.sh，修改OUTPUT_DIR，CHECKPOINT
     - 1epoch：checkpoint-xxx
     - 2epoch: checkpoint-xxx
     - 3epoch: checkpoint-xxx
     - 可能得看日志对应起来，checkpoint取1000的倍数
```

IMAGE_FILE1="/data/test_d2code/img1.txt"
IMAGE_FILE2="/data/test_d2code/img2.txt"
IMAGE_FILE3="/data/test_d2code/img3.txt"
IMAGE_FILE4="/data/test_d2code/img4.txt"
IMAGE_FILE5="/data/test_d2code/img5.txt"
IMAGE_FILE6="/data/test_d2code/img6.txt"
IMAGE_FILE7="/data/test_d2code/img7.txt"
IMAGE_FILE8="/data/test_d2code/img8.txt"
OUTPUT_DIR="/root/MiniCPM-V/rs_cpm_finetune_sampling_tune_visua_16_1epoch"
CHECKPOINT="/root/MiniCPM-V/output/output_minicpmv2_lora/checkpoint-xxx"
cd /root/MiniCPM-V

CUDA_VISIBLE_DEVICES=0 nohup python infer_web.py \
    --img_file $IMAGE_FILE1 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &

CUDA_VISIBLE_DEVICES=1 nohup python infer_web.py \
    --img_file $IMAGE_FILE2 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &

CUDA_VISIBLE_DEVICES=2 nohup python infer_web.py \
    --img_file $IMAGE_FILE3 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &

CUDA_VISIBLE_DEVICES=3 nohup python infer_web.py \
    --img_file $IMAGE_FILE4 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &

CUDA_VISIBLE_DEVICES=4 nohup python infer_web.py \
    --img_file $IMAGE_FILE5 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &

CUDA_VISIBLE_DEVICES=5 nohup python infer_web.py \
    --img_file $IMAGE_FILE6 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &

CUDA_VISIBLE_DEVICES=6 nohup python infer_web.py \
    --img_file $IMAGE_FILE7 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &

CUDA_VISIBLE_DEVICES=7 nohup python infer_web.py \
    --img_file $IMAGE_FILE8 \
    --path_to_checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR &
```
3. 运行测试
```
sh /root/MiniCPM-V/infer_d2code.sh
```
