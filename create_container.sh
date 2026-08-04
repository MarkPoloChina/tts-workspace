docker run  -d \
--name xzw_vllm \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-e ASCEND_RUNTIME_OPTIONS=NODRV \
--privileged=true \
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
-v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
-v /home:/home \
-v /data:/data \
-v /mnt:/mnt \
-v /usr/local/sbin/:/usr/local/sbin \
-v /etc/ascend_install.info:/etc/ascend_install.info \
--shm-size=1000g \
--net=host \
-it quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler  \
/bin/bash
