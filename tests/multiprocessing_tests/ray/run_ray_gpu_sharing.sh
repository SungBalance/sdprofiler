export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=0,1

NSYS_DIR="./nsys"
if [ ! -d "${NSYS_DIR}" ]; then
    mkdir -p ${NSYS_DIR}
fi

# nsys profile --trace=cuda,nvtx -o ${NSYS_DIR}/ray_gpu_sharing --force-overwrite true \
    
python ray_gpu_sharing.py > ray.log 2>&1
