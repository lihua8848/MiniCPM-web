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
./obsutil cp -f obs://lxy-obs/minicpm_docker.tar /path/to/docker/

./obsutil cp -f obs://lxy-obs/screenshots_16w.zip /path/to/data/

./obsutil cp -f obs://lxy-obs/train_reverse_html_qa_imgtoken_164600_minicpm.json /path/to/data/

./obsutil cp -f obs://lxy-obs/train_reverse_html_qa_imgtoken_1000_minicpm_test.json /path/to/data/

./obsutil cp -f obs://huawei-b127/MiniCPM-Llama3-V-2_5/ /path/to/data/
```
7. 下载代码到/path/to/data/
```
git clone https://github.com/lihua8848/MiniCPM-web.git
```
8. 安装镜像
```
docker load -i /path/to/docker/minicpm_docker.tar
```
9. 启动容器
```
docker run --gpus all -dit --shm-size=[内存大小]g --net=host -v /path/to/data/:/data  --name=minicpm 容器id
```

# 训练
1. conda 环境
```
conda activate MiniCPM-V
```
2. 在/data/MiniCPM-V/finetune下面运行shell脚本
```
sh finetune_lora.sh
```
