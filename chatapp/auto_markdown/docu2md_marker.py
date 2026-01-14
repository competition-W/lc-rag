from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
import argparse
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


config = {
    "output_format": "markdown",
    # "page_range": "0-15",
    "disable_image_extraction": True, ###需要同时启用
    "output_dir": "/mnt/chatchat/auto_markdown/output",
    # 指定使用 marker 自带的 OpenAIService 类
    "use_llm": True, ##图像会被描述文本代替，所以还需要视觉语言模型！意外地实现了图片理解，图像将被描述文本代替
    "llm_service": "marker.services.openai.OpenAIService",  # ✅ 指定路径
    "openai_api_key": "sk-b285c03d1bf0401a977132de909f89ed",  # ✅ 替换为你自己的 key
    "openai_model": "qwen-vl-max",     # 
    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"  # ✅ 或自托管 OpenAI-compatible endpoint
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

def parse_args():
    p = argparse.ArgumentParser(description="convert many files to markdown.")
    p.add_argument("--source-file", required=True, help="源文件的地址")
    p.add_argument("--target-file-name",     #
                   required=True,
                   help="目标文件的名称，默认存储在 /mnt/chatchat/source_DATA_md/")
    return p.parse_args()

### 可以使用本地模型替代，使用Ollama
if __name__ == "__main__":
    args = parse_args()
    target_dir = "/mnt/chatchat/source_DATA_md/"
    target_file = os.path.join(target_dir, args.target_file_name)

    rendered = converter(args.source_file)
    markdown, metadata, images = text_from_rendered(rendered)
    # 将Markdown文本保存为 .md 文件
    with open(target_file, 'w') as f:
        f.write(markdown)
    logger.error("源文件\n %s\n 已转换成 %s", args.source_file, target_file)

