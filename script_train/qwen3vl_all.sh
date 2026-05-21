#!/bin/bash
set -e
bash ./qwen3vl_adaptive_mrl_stage1_cls.sh
bash ./qwen3vl_adaptive_mrl_stage1_vqa.sh
bash ./qwen3vl_base_mrl_cls.sh
bash ./qwen3vl_base_mrl_vqa.sh