# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1


clear && \
pkill -9 python && \
pkill -9 ray


CUDA_VISIBLE_DEVICES=0,1 \
nsys profile -t cuda,nvtx \
python ./ray_pipeline_test.py