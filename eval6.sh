MODEL="training/AdaptiveMRL_qwen3_cls_projection_only/checkpoint-epoch-1"
python eval_mmeb.py \
    --model_name "${MODEL}" \
    --encode_output_path ./MMEB-evaloutputs/AdaptiveMRL_qwen3_cls_projection_only/ \
    --pooling eos \
    --model_backbone "qwen3_vl" \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split test \
    --per_device_eval_batch_size 4 \
    --image_resolution mid \
    --image_dir "eval_images" \
    --tgt_prefix_mod \
    --nested_dims 64 128 256 512 768 1024 2048
