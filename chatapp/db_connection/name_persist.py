from typing import Dict
import json
from pathlib import Path

def load_map_reduce_from_registry(registry_path: str = None) -> Dict[str, str]:
    """从注册表动态加载所有ready状态的语料库"""
    if registry_path is None:
        registry_path = "/mnt/ai/corpus_manager_persist/data/storage/registry/corpus_registry.json"
    
    registry_file = Path(registry_path)
    if not registry_file.exists():
        # 回退到空配置（完全依赖新系统）
        return {}
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    map_reduce = {}
    for corpus_name, corpus_info in registry.items():
        if corpus_info.get("status") == "ready":
            persist_dir = corpus_info.get("persist_dir")
            if persist_dir and Path(persist_dir).exists():
                map_reduce[corpus_name] = persist_dir
    
    return map_reduce

# 动态加载所有ready状态的语料库
map_reduce = load_map_reduce_from_registry()

# 调试信息（启动时显示）
if __name__ == "__main__":
    if map_reduce:
        print(f"✅ 从注册表加载了 {len(map_reduce)} 个语料库")
        for name in sorted(map_reduce.keys()):
            print(f"   - {name}")
    else:
        print("⚠️  未加载到任何语料库")
