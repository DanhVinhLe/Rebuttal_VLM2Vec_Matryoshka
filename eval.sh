MODEL="training/AdaptiveMRL_fastVLM_cls3/checkpoint-epoch-0"
python eval_mmeb.py \
    --model_name "${MODEL}" \
    --encode_output_path ./MMEB-evaloutputs/fastvlm_mrl_cls_1/ \
    --pooling eos \
    --model_backbone "llava_qwen2" \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split test \
    --per_device_eval_batch_size 4 \
    --image_resolution mid \
    --image_dir "/home/gdi-user/enguyen/research_vllm/test/VLM_Embed/eval_images" \
    --tgt_prefix_mod \
    --nested_dims 64 128 256 512 768 896
