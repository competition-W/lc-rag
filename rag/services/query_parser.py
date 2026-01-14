"""
    意图识别

"""
import json
import logging
import os
import re 
from typing import Tuple, Dict, Any, List
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from utils.auth import AuthContext

logger = logging.getLogger(__name__)

# =========================================================
# 🟢 1. 备用数据：您之前提供的详细组织列表
# 当 JSON 文件里读不到值时，使用这个列表兜底
# =========================================================
FALLBACK_TISSUES = [
            "背最长肌", "心脏", "T细胞", "背膘", "癌旁组织", "疤痕皮肤", "半脑", "海马", "颞叶", 
            "瘢痕皮肤", "骨痂细胞悬液", "瓣膜组织", "肠癌", "膀胱", "膀胱癌", "鲍温病皮肤", 
            "肺", "扁桃体癌", "背肌", "黑质", "棕色脂肪", "扁桃体癌旁", "扁桃体鳞癌", 
            "髌下脂肪", "肠", "大脑", "全血", "外周血", "PBMC", "淋巴瘤", "肠癌肠镜活检", 
            "肠癌和癌旁", "肠癌活检", "肠癌淋巴结", "肠癌组织和淋巴结", "肠粘膜", "肠组织", 
            "垂体瘤", "冻存细胞", "心肌组织", "椎体突出", "眼球", "大肠", "肠道", 
            "肠系膜淋巴结", "脂肪组织", "尘肺", "皮肤", "大肠癌", "大肠穿刺", "动脉夹层", 
            "动脉硬化血管组织", "非小细胞肺癌", "肺癌", "细胞悬液", "骨髓细胞", "肺癌癌旁", 
            "大脑皮层", "肺癌穿刺", "多裂肌", "肺粘液腺癌", "肺组织", "多组织", "胶质瘤", 
            "淋巴瘤细胞", "肺癌和癌旁", "心肌", "肺部肉芽", "额叶", "耳朵皮肤", "肺穿刺", 
            "肺黏液腺癌", "肺泡灌洗液", "脾脏来源细胞", "腓肠", "腓肠肌", "肺腺癌", 
            "肺转移瘤", "肌肉", "皮肤样本冻存细胞", "肺组织来源细胞", "心脏分选细胞", 
            "肺来源细胞", "腹部皮肤", "腹水", "肝转移灶", "干细胞", "心脏来源细胞", 
            "胃癌细胞", "干细胞诱导内皮细胞", "宫颈", "宫颈癌", "股骨", "骨肉瘤", 
            "骨髓血", "肝癌", "肝癌癌旁", "胰岛来源细胞", "膀胱壁", "大脑PFC", "大脑NAC", 
            "肌肉神经组织来源细胞", "骨和骨髓", "膀胱癌细胞", "肌腱组织", "骨纤维结构不良组织", 
            "骨转移癌", "骨组织", "关节突", "颌骨", "颌骨和神经", "化疗后乳腺癌", 
            "肺癌骨转移瘤", "胃癌冻存细胞", "肺癌脑转移", "肺癌脑转移瘤", "肺损伤组织", 
            "化疗后乳腺癌转移", "化疗后胰腺癌", "肺叶", "脊柱肌肉", "骨髓", "肺原位瘤", 
            "肺转移脑瘤", "回肠远端肠组织", "结肠", "伏隔核", "肾脏", "脑", 
            "结肠癌肝转移肠正常组织", "结肠癌肝转移肠肿瘤组织", "结肠黏膜", "结直肠", 
            "回肠", "结肠癌", "胰腺胰岛", "子宫来源细胞", "结直肠癌", "分选细胞", 
            "颈动脉斑块", "淋巴", "淋巴穿刺", "淋巴结", "卵巢组织", "脑部生殖瘤", 
            "脑胶质瘤", "脑皮层", "淋巴瘤骨髓血", "淋巴瘤细胞复苏", "皮肤组织", 
            "肝类器官", "基质组分", "鳞癌", "视网膜纤维血管膜", "肝脏", "胎盘", 
            "肝脏原位瘤", "颅骨", "膈肌", "肱三头肌", "卵巢", "毛囊", "骨髓PBMC冻存细胞", 
            "脑复发胶质瘤", "股骨骨髓", "骨", "脾脏", "脾脏细胞", "纤维化肺冻存细胞", 
            "前列腺", "PBMC细胞", "空肠", "瘤胃", "脑胶质瘤穿刺", "乳腺癌", "骨髓和骨", 
            "脑膜", "脑生殖瘤", "脑生殖瘤组织", "脑肿瘤", "脑组织", "骨髓", "关节", 
            "海马体", "海马组织", "肾癌", "肾癌旁", "胚胎头骨", "脑核团", "胃癌", 
            "皮肤肿瘤", "肌肉组织", "肝分选细胞", "血管", "前列腺癌", "角膜", 
            "巧克力囊肿壁", "丘脑", "肠癌和淋巴结", "肾解离细胞", "脾脏冻存细胞", 
            "胃来源细胞", "骨髓MSC细胞", "胫骨接种肿瘤细胞", "皮下瘤", "肠道来源细胞", 
            "流式分选脑细胞", "背根神经节", "老化皮肤", "贲门", "流式分选造血干细胞", 
            "胃癌来源细胞", "肠系膜脂肪", "流式分选B细胞", "胰腺细胞", "颅骨缺损", 
            "股骨颈", "骨骼肌", "下丘脑", "卵巢癌", "卵巢皮质", "血管瘤", 
            "大网膜内脏脂肪", "脾脏分选细胞", "脑白质", "脑灰质", "脑胶质母细胞瘤", 
            "核团", "脑胶质瘤细胞", "腹膜后脂肪肉瘤", "中脑", "肩胛间棕色组织", 
            "皮层", "颈部皮肤", "胰腺癌", "皮下肺癌肿瘤", "胰腺癌肝转移", "胃", 
            "气道肺组织", "气管", "胰腺纤维悬液", "脑区", "气管内疤痕", "肠癌来源细胞", 
            "全脑", "缺损颅骨", "乳腺", "绒毛", "乳头皮肤肿瘤", "乳腺癌转移淋巴组织", 
            "乳腺癌癌旁", "乳腺癌穿刺", "乳腺癌化疗后组织", "乳腺癌脑转移", "乳腺癌卫星灶", 
            "乳腺癌原发灶", "软骨骨痂", "烧伤皮肤组织", "皮肤创面", "肾", "胰岛", 
            "前额叶皮层", "中缝背核", "乳腺癌治疗后", "乳腺癌组织", "乳腺穿刺", 
            "上臂皮肤", "舌癌", "神经胶质瘤组织", "肾皮质", "肾和肾癌来源细胞", 
            "脑细胞核", "乳头状肾细胞癌", "睑板腺癌", "肾损伤组织", "乳腺癌骨转移", 
            "肾癌细胞", "肾癌胰腺转移", "肾穿刺", "多能干细胞", "视网膜", 
            "视网膜脉络膜", "肺菌灌胃处理", "肾上腺皮质癌", "肾上腺皮质癌旁", 
            "肾上腺皮质癌细胞", "肾上腺醛固酮腺瘤", "肾上腺正常组织", "肾脏穿刺", 
            "食管癌", "食管癌组织", "松质骨", "胎儿肾", "透明隔", "腿部皮肤", 
            "蜕膜", "流式分选细胞悬液", "肌肉来源细胞", "下颌骨", "小肠", "主动脉", 
            "小脑", "心脏瓣膜", "肾癌来源细胞", "肾来源细胞", "胃癌癌旁", "胃贲门", 
            "胃肠黏膜冻存细胞", "细胞系", "腺鳞癌", "腺瘤周围正常肾上腺组织", 
            "小肠癌", "小腿皮肤", "心脏组织", "胎儿肠", "胎盘组织", "乳腺癌来源细胞", 
            "新生小鼠皮肤", "牙", "牙和牙冠", "牙龈颌骨", "牙龈上颌骨", "胰腺", 
            "心脏动脉", "心脏动脉瓣膜", "心脏动脉瘤", "心脏主动脉", "心脏血管脂肪", 
            "皮肤分选细胞", "星状神经节", "牙龈", "胰腺癌肺转移", "胰腺癌原位瘤", 
            "乳腺癌冻存细胞", "胰腺肿瘤", "胰腺癌近端", "胰腺癌淋巴", "骨髓分选细胞", 
            "胰腺癌旁", "长骨", "体液", "胰腺癌远端", "胰腺穿刺", "胰腺来源细胞", 
            "原发灶乳腺癌", "早期鳞癌", "真皮", "肺粘液癌", "直肠黏膜活检", 
            "中枢神经细胞瘤", "子宫内膜", "子宫内膜癌", "主动脉壁", "转移淋巴结", 
            "子宫", "子宫平滑肌", "子宫滋养细胞瘤", "左心房心肌", "右心房心肌", 
            "左大腿皮肤", "左心室"
        ]

# FALLBACK_SPECIES = ["人", "小鼠", "大鼠", "猴", "斑马鱼", "猪", "牛"]

# # 允许筛选的字段
# ALLOWED_FILTER_KEYS = [
#     "col_物种", 
#     "col_样本详细类型", 
#     "col_实验平台", 
#     "col_样本大类", 
#     "col_样本保存方案",
#     "col_是否裂红",
#     "col_是否去死"
# ]

# # 🔥 键名映射表：防止 LLM 记不住复杂的 Key，做一层容错映射
# KEY_MAPPING = {
#     "物种": "col_物种",
#     "组织": "col_样本详细类型",
#     "组织类型": "col_样本详细类型",
#     "样本类型": "col_样本详细类型",
#     "部位": "col_样本详细类型",
#     "平台": "col_实验平台",
#     "保存方案": "col_样本保存方案"
# }

FALLBACK_SPECIES = ["人", "小鼠", "大鼠", "猕猴", "绵羊","山羊","羊", "猪", "牛","蝙蝠","人+大鼠"]

# ==========================================
# 1. 允许筛选的字段 (必须与 Milvus 里的英文 Key 一致)
# ==========================================
ALLOWED_FILTER_KEYS = [
    "species",          # 对应原来的 col_物种
    "tissue",           # 对应 col_样本详细类型
    "platform",         # 对应 col_实验平台
    "category",         # 对应 col_样本大类
    "storage_method",   # 对应 col_样本保存方案
    "is_lysis",         # 对应 col_是否裂红
    "is_dead_removal",  # 对应 col_是否去死
    "rin_score",        # 对应 col_核酸质量
    "project_id",
    "sample_date"
]

# ==========================================
# 2. 键名映射表 (User Input -> Standard DB Key)
# 作用：把用户口语词、或者 LLM 可能输出的旧 col_ 写法，统一转成标准英文
# ==========================================
KEY_MAPPING = {
    # === 物种 ===
    "物种": "species",
    "col_物种": "species", # 防止 LLM 还是习惯性输出旧 Key

    # === 组织/样本类型 ===
    "组织": "tissue",
    "组织类型": "tissue",
    "样本类型": "tissue",
    "样本详细类型": "tissue",
    "部位": "tissue",
    "col_样本详细类型": "tissue",

    # === 平台 ===
    "平台": "platform",
    "实验平台": "platform",
    "col_实验平台": "platform",

    # === 大类 ===
    "大类": "category",
    "样本大类": "category",
    "col_样本大类": "category",

    # === 保存方案 ===
    "保存方案": "storage_method",
    "样本保存方案": "storage_method",
    "col_样本保存方案": "storage_method",

    # === 其他处理 ===
    "裂红": "is_lysis",
    "是否裂红": "is_lysis",
    "col_是否裂红": "is_lysis",

    "去死": "is_dead_removal",
    "是否去死": "is_dead_removal",
    "col_是否去死": "is_dead_removal"
}

def load_dynamic_schema(json_path: str = None) -> str:
    """
    加载 Schema，如果 JSON 里是空的，自动使用 fallback 数据
    """
    if json_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "..", "data", "schema_registry.json")
        json_path = os.path.normpath(json_path)

    registry = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            logger.info(f"✅ 成功加载 Schema 文件: {json_path}")
        except Exception as e:
            logger.error(f"❌ 读取 Schema JSON 出错: {e}")
    
    columns = registry.get("market", [])
    schema_lines = []

    # 这里的 Set 用于去重，防止 Prompt 过长
    added_keys = set()

    # 1. 优先处理 JSON 文件里的定义
    for col in columns:
        key = col.get("key")
        if key not in ALLOWED_FILTER_KEYS:
            continue
            
        name = col.get("name", key)
        valid_values = col.get("valid_values", [])

        # 🔧 补丁：如果样本类型为空，注入备用
        if key == "col_样本详细类型":
            if not valid_values or (len(valid_values) == 1 and not valid_values[0]):
                valid_values = FALLBACK_TISSUES
        
        # 🔧 补丁：如果物种为空，注入备用
        if key == "col_物种":
            if not valid_values:
                valid_values = FALLBACK_SPECIES

        clean_values = [str(v) for v in valid_values if v and str(v).strip() not in ["/", "nan"]]
        
        if clean_values:
            # 限制显示数量，防止 Token 溢出
            values_str = ", ".join(clean_values[:80]) 
            schema_lines.append(f"- 字段: `{key}` ({name})\n  可选值: [{values_str}]")
            added_keys.add(key)

    # 2. 如果 JSON 完全挂了，启用强制兜底
    if "col_样本详细类型" not in added_keys:
        logger.warning("⚠️ Schema 中缺少样本类型，启用强制兜底")
        schema_lines.append(f"- 字段: `col_样本详细类型` (组织)\n  可选值: {str(FALLBACK_TISSUES[:80])}")
        
    if "col_物种" not in added_keys:
        schema_lines.append(f"- 字段: `col_物种` (物种)\n  可选值: {str(FALLBACK_SPECIES)}")

    return "\n".join(schema_lines)

# 全局加载
SCHEMA_CONTEXT_CACHE = load_dynamic_schema()

async def parse_user_query(user_text: str, auth: AuthContext = None) -> Tuple[str, Dict[str, Any]]:
    # 2. 构建 Prompt
    system_prompt = (
        "你是一个生物数据库查询专家。请将用户的自然语言转换为精确的数据库筛选条件。\n\n"
        "### 数据库字段定义 (Schema)\n"
        f"{SCHEMA_CONTEXT_CACHE}\n\n"
        "### 提取规则\n"
        "1. **必须严格匹配**：提取出的 filter 值必须出现在上述【可选值】列表中。\n"
        "   - 用户说 'Mouse' -> 输出 '小鼠'\n"
        "   - 用户说 'Heart' -> 输出 '心脏'\n"
        "2. **输出格式**：只输出标准的 JSON 字符串，不要包含 '```json' 或其他废话。\n"  
        "3. **Key要求**：必须使用 `col_` 开头的标准字段名（例如 `col_物种`）。\n\n"  
        "### 示例\n"  
        "用户: 小鼠心脏的冻存方案\n"  
        "助手: " + json.dumps({  
            "filters": {  
                "col_物种": "小鼠",   
                "col_样本详细类型": "心脏"  
            },  
            "search_term": "冻存方案"  
        }, ensure_ascii=False)  
    )  

    try:  
        if not Settings.llm:  
            return user_text, {}  

        # 调用 LLM  
        response = await Settings.llm.achat(  
            messages=[  
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),  
                ChatMessage(role=MessageRole.USER, content=user_text)  
            ]  
        )  
        
        raw_content = response.message.content  
        logger.info(f"🧠 [LLM Raw Output]: {raw_content}")  

        # =========================================================  
        # 🔥 核心增强 1: 使用正则提取 JSON，无视 Markdown 和废话  
        # =========================================================  
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)  
        if not json_match:  
            logger.warning("❌ 无法从 LLM 输出中提取 JSON")  
            return user_text, {}  
            
        json_str = json_match.group()  
        parsed_result = json.loads(json_str)  
        
        filters = parsed_result.get("filters", {})  
        search_term = parsed_result.get("search_term", user_text)  

        # =========================================================  
        # 🔥 核心增强 2: 键名自动修正 (Key Correction)  
        # 防止 LLM 输出 "组织": "心脏" 这种非标准 Key  
        # =========================================================  
        final_filters = {}  
        for k, v in filters.items():  
            # 1. 如果是标准 Key，直接用  
            if k in ALLOWED_FILTER_KEYS:  
                final_filters[k] = v  
            # 2. 如果是别名（如 "组织"），查表映射回 "col_样本详细类型"  
            elif k in KEY_MAPPING:  
                correct_key = KEY_MAPPING[k]  
                final_filters[correct_key] = v  
                logger.info(f"🔧 自动修正 Key: {k} -> {correct_key}")  
            else:  
                logger.warning(f"⚠️ 丢弃未知字段: {k}")  

        return search_term, final_filters  

    except Exception as e:  
        logger.error(f"❌ Parser 解析异常: {e}")  
        # 降级：只返回原文本，不做过滤  
        return user_text, {}  