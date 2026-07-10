# VLMEmbed
## Set up env
```bash
apt-get update
apt-get upgrade -y
cd VLM_Embed_Matryoshka
python -m venv vlm
source vlm/bin/activate
```
## Set up
```
pip install -r requirements.txt
```
## Download dataset
1. Download the eval image file zip from huggingface (`optional`) 
```bash
cd VLM_Embed_Matryoshka
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```
2. Download train image, it can take > 1 hour to download
```bash
cd VLM_Embed_Matryoshka
bash download_traindata.sh
bash download_traindata_2.sh
```
3. Fix some line code 

Because of the error of code in **Transformers library**, run the following script to find the error and comment some lines: 

Just comment the following code, from line 139 to 143 in file **/vlm/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py**: 
```python
if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
    raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
else:
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
```
Or run `fix_lib.py` to fix: 
```python 
python fix_lib.py
```

## Training

Just run the 8 scripts in folder `script_train`
- For run training: 
```bash
bash fastvlm_adaptive_mrl_stage1_cls_0.5_0.1.sh
bash fastvlm_adaptive_mrl_stage1_cls_0.5_0.3.sh
bash fastvlm_adaptive_mrl_stage1_cls_0.5_0.5.sh
bash fastvlm_adaptive_mrl_stage1_cls_0.7_0.1.sh
bash fastvlm_adaptive_mrl_stage1_cls_0.7_0.3.sh
bash fastvlm_adaptive_mrl_stage1_cls_0.7_0.5.sh
bash fastvlm_adaptive_mrl_stage1_cls_1.0_0.1.sh
bash fastvlm_adaptive_mrl_stage1_cls_1.0_0.5.sh
```
## Inference & Evaluation
1. To evaluate our model on an MMEB dataset (e.g., MSCOCO_i2t), run (use correct training point):
```bash 
bash eval.sh
bash eval2.sh
bash eval3.sh
bash eval4.sh
bash eval5.sh
bash eval6.sh
bash eval7.sh
bash eval8.sh
```
Finally, zip json folder ./MMEB-evaloutputs

## Acknowledgement
- We have adapted code from [VLM2Vec]([https://github.com/TIGER-AI-Lab/VLM2Vec]) and [B3](https://github.com/raghavlite/B3)
