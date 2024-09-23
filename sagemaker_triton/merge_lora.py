import argparse
import logging
import torch
from transformers import AutoModelForSpeechSeq2Seq
from peft import PeftModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_lora(model_name_or_path, output_path, lora_path):
    logger.info('Loading model...')
    # 加载基础模型
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    logger.info(f'Base model dtype: {base_model.dtype}')

    # 加载 LoRA 权重并合并
    logger.info(f'Loading LoRA weights from {lora_path}')
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = peft_model.merge_and_unload()
    logger.info(f'Merged model dtype: {merged_model.dtype}')

    logger.info(f"Saving the target model to {output_path}")
    # 保存整个模型（包括架构）
    torch.save(merged_model, output_path)

    logger.info("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply LoRA to a Whisper model and save the merged result.")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v3", help="Path to the base model or its name on Hugging Face.")
    parser.add_argument("--output_path", type=str, default="full_merged_whisper_lora.pt", help="Path to save the merged model.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA weights.")

    args = parser.parse_args()

    apply_lora(args.model_name_or_path, args.output_path, args.lora_path)
