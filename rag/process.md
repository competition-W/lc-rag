好的，我为你提供一个完整的、详细的数据预处理方案。这个方案将确保你的数据结构化，便于Milvus进行精确过滤，并为后续的LLM上下文构建提供清晰、可控的输入。

### **数据预处理总览**

目标是将原始的两张表数据转换为统一的JSONL格式，每个JSON对象代表一个可检索的文档。每个文档将包含以下核心字段：

*   `id` (string): 唯一标识符。
*   `text` (string): 文档的自然语言描述，用于Embedding模型的输入和LLM的上下文。
*   `metadata` (dict): 包含所有结构化信息，用于Milvus的过滤和LLM的结构化提取。

#### **核心原则：**

1.  **扁平化与标准化：** 消除多级表头和嵌套结构，将字段名标准化为统一、易于识别的英文或拼音。
2.  **类型转换：** 将数值、百分比等字段转换为适当的数值类型（`int`/`float`）。
3.  **信息提取与丰富：** 从复杂文本字段（如“人工注释结果”、“备注”）中提取更细粒度的信息。
4.  **原子化文档：** 尤其针对第二张表，将一个原始记录拆分成多个具有明确含义的独立文档。
5.  **Milvus兼容性：** `metadata`中的字段应易于在Milvus中创建索引和执行过滤查询。

---

### **详细预处理方案**

我们将为每张表分别制定预处理步骤。

### **A. 第一张表：实验数据 (实验数据表)**

**原始数据示例：**

```json
{
  "项目建库类型": "10X单细胞3‘转录本-抽核(V3试剂)",
  "物种": "小鼠",
  "样本类型": "冻存组织",
  "样本详细类型": "心肌组织",
  "实验方案（解离/抽核）": "抽核",
  "到样温度（℃）": "-80",
  "细胞总量(万)": "10",
  "结团率(%)": "1.2",
  "细胞活率(%)": "0",
  "有核率(%)": "99",
  "捕获细胞数": "14,466",
  "reads/cell": "22,517",
  "基因中位数": "1,259",
  "人工注释结果": "Endothelial(38.6%),Fibroblast(26.34%),Adipocyte(12.52%),Macrophage(10.09%),Pericyte(9.1%),Lymphatic endothelial cell(1.22%),Cardiomyocyte(0.61%),T cell(0.6%),Schwann cell(0.39%),DC(0.34%),B cell(0.18%)",
  "组织消化方案": null, // 示例中为空
  "相关组织用户文章链接": null // 示例中为空
}
```

**预处理步骤：**

1.  **为每条记录生成唯一 `id`：** 可以使用一个序列号，例如 `exp_001`, `exp_002`...
    *   `id`: `string`
2.  **标准化字段名：**
    *   `project_library_type`: `项目建库类型` (`string`)
    *   `species`: `物种` (`string`)
    *   `sample_type`: `样本类型` (`string`)
    *   `sample_detailed_type`: `样本详细类型` (`string`)
    *   `prep_method`: `实验方案（解离/抽核）` (`string`, 标准化为 "解离" 或 "抽核")
    *   `arrival_temp_celsius`: `到样温度（℃）` (`float`)
    *   `total_cells_10k`: `细胞总量(万)` (`float`)
    *   `clumping_rate_percent`: `结团率(%)` (`float`)
    *   `cell_viability_percent`: `细胞活率(%)` (`float`)
    *   `nucleated_rate_percent`: `有核率(%)` (`float`)
    *   `captured_cells`: `捕获细胞数` (`int`)
    *   `reads_per_cell`: `reads/cell` (`int`)
    *   `median_genes`: `基因中位数` (`int`)
    *   `annotation_results_full`: `人工注释结果` (`string`, 完整原始文本)
    *   `tissue_digestion_protocol`: `组织消化方案` (`string`)
    *   `tissue_digestion_summary`: `组织消化方案概述` (`string`)
    *   `related_article_link`: `相关组织用户文章链接` (`string`)
    *   `feishu_doc_link`: `飞书文档链接` (`string`)
    *   `video_link`: `视频直播链接` (`string`)
3.  **类型转换与数据清洗：**
    *   **数值字段：** 将所有数值字段从字符串转换为 `float` 或 `int`。移除可能存在的逗号（例如 "14,466" -> 14466）。如果原始数据是百分比，直接转换为 `float` (例如 "1.2%" -> 1.2)。
    *   **`prep_method`：** 确保这个字段的值只有 "解离" 或 "抽核" 两种，去除括号等冗余信息。
    *   **缺失值：** 对于空值，统一处理为 `None` (Python) 或 `null` (JSON)。
4.  **信息提取与丰富 (`人工注释结果`)：**
    *   **`annotation_cell_types_list`:** 从 `annotation_results_full` 字段中解析出所有识别到的细胞类型，存储为一个字符串列表。
        *   例如："Endothelial(38.6%),Fibroblast(26.34%)" -> `["Endothelial", "Fibroblast", "Adipocyte", ...]`
        *   **作用：** 支持Milvus的`$contains`操作，用于查询“有没有鉴定到xx细胞”。
    *   **`annotation_cell_type_details`:** 存储为一个字典，键是细胞类型，值是其百分比。
        *   例如：`{"Endothelial": 38.6, "Fibroblast": 26.34, ...}`
        *   **作用：** 方便LLM在回答时引用特定细胞类型的百分比，也支持精确查询。
5.  **构建 `text` 字段：** 结合所有核心信息，形成一个连贯的自然语言描述。
    *   **示例：**
        ```text
        该实验记录了单细胞平台类型为"{project_library_type}"的样本。
        样本信息：物种为"{species}"，样本类型为"{sample_type}"，详细类型是"{sample_detailed_type}"。
        实验方案采用"{prep_method}"，到样温度为{arrival_temp_celsius}℃。
        实验指标：细胞总量{total_cells_10k}万，结团率{clumping_rate_percent}%，细胞活率{cell_viability_percent}%，有核率{nucleated_rate_percent}%。
        数据指标：捕获细胞数{captured_cells}，平均reads/cell为{reads_per_cell}，基因中位数为{median_genes}。
        人工注释结果显示主要细胞类型及其占比为：{annotation_results_full}。
        ```
6.  **构建 `metadata` 字段：**
    *   包含所有标准化后的字段 (`project_library_type`, `species`, `sample_type`, ...)。
    *   包含丰富后的字段 (`annotation_cell_types_list`, `annotation_cell_type_details`)。
    *   添加一个 `source_table: "experiment_data"` 标签，方便后续识别来源。

**输出示例 (First table document):**

```json
{
  "id": "exp_001",
  "text": "该实验记录了单细胞平台类型为\"10X单细胞3‘转录本-抽核(V3试剂)\"的样本。样本信息：物种为\"小鼠\"，样本类型为\"冻存组织\"，详细类型是\"心肌组织\"。实验方案采用\"抽核\"，到样温度为-80.0℃。实验指标：细胞总量10.0万，结团率1.2%，细胞活率0.0%，有核率99.0%。数据指标：捕获细胞数14466，平均reads/cell为22517，基因中位数为1259。人工注释结果显示主要细胞类型及其占比为：Endothelial(38.6%),Fibroblast(26.34%),Adipocyte(12.52%),Macrophage(10.09%),Pericyte(9.1%),Lymphatic endothelial cell(1.22%),Cardiomyocyte(0.61%),T cell(0.6%),Schwann cell(0.39%),DC(0.34%),B cell(0.18%)。",
  "metadata": {
    "source_table": "experiment_data",
    "project_library_type": "10X单细胞3‘转录本-抽核(V3试剂)",
    "species": "小鼠",
    "sample_type": "冻存组织",
    "sample_detailed_type": "心肌组织",
    "prep_method": "抽核",
    "arrival_temp_celsius": -80.0,
    "total_cells_10k": 10.0,
    "clumping_rate_percent": 1.2,
    "cell_viability_percent": 0.0,
    "nucleated_rate_percent": 99.0,
    "captured_cells": 14466,
    "reads_per_cell": 22517,
    "median_genes": 1259,
    "annotation_results_full": "Endothelial(38.6%),Fibroblast(26.34%),Adipocyte(12.52%),Macrophage(10.09%),Pericyte(9.1%),Lymphatic endothelial cell(1.22%),Cardiomyocyte(0.61%),T cell(0.6%),Schwann cell(0.39%),DC(0.34%),B cell(0.18%)",
    "annotation_cell_types_list": ["Endothelial", "Fibroblast", "Adipocyte", "Macrophage", "Pericyte", "Lymphatic endothelial cell", "Cardiomyocyte", "T cell", "Schwann cell", "DC", "B cell"],
    "annotation_cell_type_details": {"Endothelial": 38.6, "Fibroblast": 26.34, "Adipocyte": 12.52, "Macrophage": 10.09, "Pericyte": 9.1, "Lymphatic endothelial cell": 1.22, "Cardiomyocyte": 0.61, "T cell": 0.6, "Schwann cell": 0.39, "DC": 0.34, "B cell": 0.18},
    "tissue_digestion_protocol": null,
    "tissue_digestion_summary": null,
    "related_article_link": null,
    "feishu_doc_link": null,
    "video_link": null
  }
]
```

---

### **B. 第二张表：样本制备指南 (sample_preparation)**

**原始数据示例：**

```json
{
  "product": {
    "产品一级目录": "单细胞空间组学",
    "产品二级目录": "单细胞测序",
    "产品三级目录": "单细胞转录组"
  },
  "sample_preparation": {
    "样本大类": "实体组织类样本",
    "样本类型": "新鲜实体组织",
    "组织类型": "肿瘤",
    "样本处理方式": "解离",
    "recommended_amount": {
      "风险": "约20mg；",
      "合格": ["约50mg；", "约250mg；"]
    },
    "qualitative_description": {
      "风险": "大米大小；",
      "合格": ["绿豆大小；", "黄豆大小；"]
    },
    "methods_and_notes": {
      "样本准备方法": "实体组织送样方法sop",
      "取样送样的注意事项": "",
      "备注": "风险：可以尝试，细胞量接近上机捕获极限，坏死类组织需根据坏死部位占比来提高送样量；\n合格：可以尝试，细胞量一般可满足上机要求，坏死类组织需根据坏死部位占比来提高送样量；\n合格：细胞量完全足够，满足上机捕获要求，坏死类组织需根据坏死部位占比来提高送样量；"
    }
  }
}
```

**预处理步骤：**

这张表的关键在于将嵌套的`recommended_amount`, `qualitative_description`, 和 `备注` 字段**展开**成多条独立的文档，每条文档代表一个具体的“风险等级-送样量-定性描述”组合。

1.  **解析 `备注` 字段：**
    *   将 `methods_and_notes.备注` 字段按分号 `;` 分割成独立的备注条目。
    *   例如："风险：...；合格：...；合格：..." -> `["风险：...", "合格：...", "合格：..."]`
    *   **核心关联：** 假定 `recommended_amount` 的 `风险` 对应 `备注` 的第一个 `风险` 条目，`合格` 的第一个条目对应 `备注` 的第二个 `合格` 条目，依此类推。这需要你在数据源保持这种隐式顺序。
2.  **遍历并展开记录：**
    *   为每个 `recommended_amount` 和 `qualitative_description` 的组合（包括“风险”和“合格”下的多个值），创建一个新的文档。
    *   例如，原始一条记录将生成3条预处理后的文档：
        *   (风险, 20mg, 大米大小, 对应第一条备注)
        *   (合格, 50mg, 绿豆大小, 对应第二条备注)
        *   (合格, 250mg, 黄豆大小, 对应第三条备注)
3.  **为每条展开的记录生成唯一 `id`：** 例如 `prep_001_risk_20mg`, `prep_001_qualified_50mg`。
    *   `id`: `string`
4.  **标准化字段名：**
    *   `product_level1`: `product.产品一级目录` (`string`)
    *   `product_level2`: `product.产品二级目录` (`string`)
    *   `product_level3`: `product.产品三级目录` (`string`)
    *   `sample_category`: `sample_preparation.样本大类` (`string`)
    *   `sample_type`: `sample_preparation.样本类型` (`string`)
    *   `tissue_type`: `sample_preparation.组织类型` (`string`)
    *   `prep_method`: `sample_preparation.样本处理方式` (`string`)
    *   `preparation_method_doc`: `methods_and_notes.样本准备方法` (`string`)
    *   `handling_notes`: `methods_and_notes.取样送样的注意事项` (`string`)
    *   `risk_level`: (`string`, "风险" 或 "合格") - 新字段，来自展开
    *   `recommended_amount_mg`: (`float`) - 新字段，从 "约20mg；" 中提取数值
    *   `qualitative_description_text`: (`string`) - 新字段，从 "大米大小；" 中提取
    *   `notes_full`: (`string`) - 新字段，对应步骤1中解析出的具体备注条目
5.  **类型转换与数据清洗：**
    *   **`recommended_amount_mg`：** 从字符串中提取数值并转换为 `float` (例如 "约20mg；" -> 20.0)。
    *   **缺失值：** 同上，统一处理为 `None` / `null`。
6.  **构建 `text` 字段：** 结合所有信息，形成一个描述特定制备场景的自然语言描述。
    *   **示例 (针对“风险”20mg的场景):**
        ```text
        这是一条关于"{product_level1}"下"{product_level3}"产品的样本制备指南。
        样本属于"{sample_category}"中的"{sample_type}"，具体为"{tissue_type}"组织。
        样本处理方式为"{prep_method}"。
        如果建议送样量为{recommended_amount_mg}mg（约"{qualitative_description_text}"），则属于"{risk_level}"级别。
        具体说明：{notes_full}
        样本准备方法请参考文档："{preparation_method_doc}"。
        ```
7.  **构建 `metadata` 字段：**
    *   包含所有标准化后的字段 (`product_level1`, `sample_category`, ...)。
    *   包含展开后新增的字段 (`risk_level`, `recommended_amount_mg`, `qualitative_description_text`, `notes_full`)。
    *   添加一个 `source_table: "preparation_guidelines"` 标签。

**输出示例 (Second table documents - expanded):**

```json
[
  {
    "id": "prep_001_risk_20mg",
    "text": "这是一条关于\"单细胞空间组学\"下\"单细胞转录组\"产品的样本制备指南。样本属于\"实体组织类样本\"中的\"新鲜实体组织\"，具体为\"肿瘤\"组织。样本处理方式为\"解离\"。如果建议送样量为20.0mg（约\"大米大小\"），则属于\"风险\"级别。具体说明：风险：可以尝试，细胞量接近上机捕获极限，坏死类组织需根据坏死部位占比来提高送样量；样本准备方法请参考文档：\"实体组织送样方法sop\"。",
    "metadata": {
      "source_table": "preparation_guidelines",
      "product_level1": "单细胞空间组学",
      "product_level2": "单细胞测序",
      "product_level3": "单细胞转录组",
      "sample_category": "实体组织类样本",
      "sample_type": "新鲜实体组织",
      "tissue_type": "肿瘤",
      "prep_method": "解离",
      "preparation_method_doc": "实体组织送样方法sop",
      "handling_notes": null,
      "risk_level": "风险",
      "recommended_amount_mg": 20.0,
      "qualitative_description_text": "大米大小",
      "notes_full": "风险：可以尝试，细胞量接近上机捕获极限，坏死类组织需根据坏死部位占比来提高送样量；"
    }
  },
  {
    "id": "prep_001_qualified_50mg",
    "text": "这是一条关于\"单细胞空间组学\"下\"单细胞转录组\"产品的样本制备指南。样本属于\"实体组织类样本\"中的\"新鲜实体组织\"，具体为\"肿瘤\"组织。样本处理方式为\"解离\"。如果建议送样量为50.0mg（约\"绿豆大小\"），则属于\"合格\"级别。具体说明：合格：可以尝试，细胞量一般可满足上机要求，坏死类组织需根据坏死部位占比来提高送样量；样本准备方法请参考文档：\"实体组织送样方法sop\"。",
    "metadata": {
      "source_table": "preparation_guidelines",
      "product_level1": "单细胞空间组学",
      "product_level2": "单细胞测序",
      "product_level3": "单细胞转录组",
      "sample_category": "实体组织类样本",
      "sample_type": "新鲜实体组织",
      "tissue_type": "肿瘤",
      "prep_method": "解离",
      "preparation_method_doc": "实体组织送样方法sop",
      "handling_notes": null,
      "risk_level": "合格",
      "recommended_amount_mg": 50.0,
      "qualitative_description_text": "绿豆大小",
      "notes_full": "合格：可以尝试，细胞量一般可满足上机要求，坏死类组织需根据坏死部位占比来提高送样量；"
    }
  },
  {
    "id": "prep_001_qualified_250mg",
    "text": "这是一条关于\"单细胞空间组学\"下\"单细胞转录组\"产品的样本制备指南。样本属于\"实体组织类样本\"中的\"新鲜实体组织\"，具体为\"肿瘤\"组织。样本处理方式为\"解离\"。如果建议送样量为250.0mg（约\"黄豆大小\"），则属于\"合格\"级别。具体说明：合格：细胞量完全足够，满足上机捕获要求，坏死类组织需根据坏死部位占比来提高送样量；样本准备方法请参考文档：\"实体组织送样方法sop\"。",
    "metadata": {
      "source_table": "preparation_guidelines",
      "product_level1": "单细胞空间组学",
      "product_level2": "单细胞测序",
      "product_level3": "单细胞转录组",
      "sample_category": "实体组织类样本",
      "sample_type": "新鲜实体组织",
      "tissue_type": "肿瘤",
      "prep_method": "解离",
      "preparation_method_doc": "实体组织送样方法sop",
      "handling_notes": null,
      "risk_level": "合格",
      "recommended_amount_mg": 250.0,
      "qualitative_description_text": "黄豆大小",
      "notes_full": "合格：细胞量完全足够，满足上机捕获要求，坏死类组织需根据坏死部位占比来提高送样量；"
    }
  }
]
```

---

### **Python 实现建议**

可以使用 `pandas` 进行数据的加载和大部分的清洗、类型转换。然后通过循环处理每行（或展开后的每个逻辑文档），构建 `text` 和 `metadata` 字段，最后保存为JSONL文件。

```python
import pandas as pd
import json
import re

def clean_and_convert_value(value, target_type):
    """通用清洗和类型转换函数"""
    if pd.isna(value) or value is None or value == "":
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip() # 移除逗号和多余空格

        if target_type == int:
            return int(float(value)) # 先转float处理小数，再转int
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return bool(value)
        else: # target_type == str
            return str(value)
    except (ValueError, TypeError):
        return None # 转换失败返回None

def preprocess_experiment_data(raw_data_dict, record_id_prefix="exp_"):
    """预处理第一张表（实验数据）"""
    processed_documents = []
    
    # 模拟从字典中获取数据，实际可能来自CSV或数据库
    # 假设raw_data_dict是单条记录
    item = raw_data_dict 

    # 1. 生成ID
    doc_id = f"{record_id_prefix}{len(processed_documents) + 1:03d}" # 示例ID生成

    # 2. 标准化字段名并转换类型
    metadata = {
        "source_table": "experiment_data",
        "project_library_type": clean_and_convert_value(item.get("项目建库类型"), str),
        "species": clean_and_convert_value(item.get("物种"), str),
        "sample_type": clean_and_convert_value(item.get("样本类型"), str),
        "sample_detailed_type": clean_and_convert_value(item.get("样本详细类型"), str),
        "prep_method": clean_and_convert_value(item.get("实验方案（解离/抽核"), str).replace('（解离/抽核）', '').strip() if item.get("实验方案（解离/抽核）") else None,
        "arrival_temp_celsius": clean_and_convert_value(item.get("到样温度（℃）"), float),
        "total_cells_10k": clean_and_convert_value(item.get("细胞总量(万)"), float),
        "clumping_rate_percent": clean_and_convert_value(item.get("结团率(%)"), float),
        "cell_viability_percent": clean_and_convert_value(item.get("细胞活率(%)"), float),
        "nucleated_rate_percent": clean_and_convert_value(item.get("有核率(%)"), float),
        "captured_cells": clean_and_convert_value(item.get("捕获细胞数"), int),
        "reads_per_cell": clean_and_convert_value(item.get("reads/cell"), int),
        "median_genes": clean_and_convert_value(item.get("基因中位数"), int),
        "annotation_results_full": clean_and_convert_value(item.get("人工注释结果"), str),
        "tissue_digestion_protocol": clean_and_convert_value(item.get("组织消化方案"), str),
        "tissue_digestion_summary": clean_and_convert_value(item.get("组织消化方案概述"), str),
        "related_article_link": clean_and_convert_value(item.get("相关组织用户文章链接"), str),
        "feishu_doc_link": clean_and_convert_value(item.get("飞书文档链接"), str),
        "video_link": clean_and_convert_value(item.get("视频直播链接"), str),
    }

    # 4. 信息提取与丰富 (人工注释结果)
    annotation_cell_types_list = []
    annotation_cell_type_details = {}
    if metadata["annotation_results_full"]:
        # 正则表达式匹配 'CellType(Percentage%)'
        matches = re.findall(r'([a-zA-Z\s]+)\((\d+\.?\d*)%\)', metadata["annotation_results_full"])
        for cell_type, percentage in matches:
            annotation_cell_types_list.append(cell_type.strip())
            annotation_cell_type_details[cell_type.strip()] = float(percentage)
    metadata["annotation_cell_types_list"] = annotation_cell_types_list
    metadata["annotation_cell_type_details"] = annotation_cell_type_details

    # 5. 构建 text 字段
    text_template = (
        "该实验记录了单细胞平台类型为\"{project_library_type}\"的样本。\n"
        "样本信息：物种为\"{species}\"，样本类型为\"{sample_type}\"，详细类型是\"{sample_detailed_type}\"。\n"
        "实验方案采用\"{prep_method}\"，到样温度为{arrival_temp_celsius}℃。\n"
        "实验指标：细胞总量{total_cells_10k}万，结团率{clumping_rate_percent}%，细胞活率{cell_viability_percent}%，有核率{nucleated_rate_percent}%。\n"
        "数据指标：捕获细胞数{captured_cells}，平均reads/cell为{reads_per_cell}，基因中位数为{median_genes}。\n"
        "人工注释结果显示主要细胞类型及其占比为：{annotation_results_full}。"
    )
    # 使用 .get() 并在 format_map 中处理 None 值以避免 KeyError
    safe_metadata = {k: v if v is not None else "未知" for k, v in metadata.items()}
    text = text_template.format_map(safe_metadata)

    processed_documents.append({"id": doc_id, "text": text, "metadata": metadata})
    return processed_documents

def preprocess_preparation_data(raw_data_dict, record_id_prefix="prep_"):
    """预处理第二张表（样本制备指南）"""
    processed_documents = []
    
    product_info = raw_data_dict.get("product", {})
    sample_prep_info = raw_data_dict.get("sample_preparation", {})
    methods_notes_info = sample_prep_info.get("methods_and_notes", {})

    # 1. 解析备注字段 (根据分号分割并清理)
    notes_parts = [n.strip() for n in methods_notes_info.get("备注", "").split('；') if n.strip()]

    # 2. 展开记录
    recommended_amounts = sample_prep_info.get("recommended_amount", {})
    qualitative_descriptions = sample_prep_info.get("qualitative_description", {})

    idx = 0 # 用于同步notes_parts
    
    # Process "风险" category
    risk_amounts = recommended_amounts.get("风险", [])
    risk_descriptions = qualitative_descriptions.get("风险", [])
    if not isinstance(risk_amounts, list): risk_amounts = [risk_amounts]
    if not isinstance(risk_descriptions, list): risk_descriptions = [risk_descriptions]
    
    for i in range(max(len(risk_amounts), len(risk_descriptions))):
        amount_str = risk_amounts[i] if i < len(risk_amounts) else None
        desc_str = risk_descriptions[i] if i < len(risk_descriptions) else None
        notes_str = notes_parts[idx] if idx < len(notes_parts) else None
        idx += 1

        if amount_str or desc_str or notes_str:
            doc_id = f"{record_id_prefix}{len(processed_documents) + 1:03d}_risk"
            
            recommended_amount_mg = clean_and_convert_value(re.search(r'(\d+)', amount_str).group(1), float) if amount_str and re.search(r'(\d+)', amount_str) else None

            metadata = {
                "source_table": "preparation_guidelines",
                "product_level1": clean_and_convert_value(product_info.get("产品一级目录"), str),
                "product_level2": clean_and_convert_value(product_info.get("产品二级目录"), str),
                "product_level3": clean_and_convert_value(product_info.get("产品三级目录"), str),
                "sample_category": clean_and_convert_value(sample_prep_info.get("样本大类"), str),
                "sample_type": clean_and_convert_value(sample_prep_info.get("样本类型"), str),
                "tissue_type": clean_and_convert_value(sample_prep_info.get("组织类型"), str),
                "prep_method": clean_and_convert_value(sample_prep_info.get("样本处理方式"), str),
                "preparation_method_doc": clean_and_convert_value(methods_notes_info.get("样本准备方法"), str),
                "handling_notes": clean_and_convert_value(methods_notes_info.get("取样送样的注意事项"), str),
                "risk_level": "风险",
                "recommended_amount_mg": recommended_amount_mg,
                "qualitative_description_text": clean_and_convert_value(desc_str.replace('；', '').strip(), str) if desc_str else None,
                "notes_full": clean_and_convert_value(notes_str, str),
            }

            text_template = (
                "这是一条关于\"{product_level1}\"下\"{product_level3}\"产品的样本制备指南。\n"
                "样本属于\"{sample_category}\"中的\"{sample_type}\"，具体为\"{tissue_type}\"组织。\n"
                "样本处理方式为\"{prep_method}\"。\n"
                "如果建议送样量为{recommended_amount_mg}mg（约\"{qualitative_description_text}\"），则属于\"{risk_level}\"级别。\n"
                "具体说明：{notes_full}\n"
                "样本准备方法请参考文档：\"{preparation_method_doc}\"。\n"
            )
            safe_metadata = {k: v if v is not None else "未知" for k, v in metadata.items()}
            text = text_template.format_map(safe_metadata)
            
            processed_documents.append({"id": doc_id, "text": text, "metadata": metadata})

    # Process "合格" category
    qualified_amounts = recommended_amounts.get("合格", [])
    qualified_descriptions = qualitative_descriptions.get("合格", [])
    if not isinstance(qualified_amounts, list): qualified_amounts = [qualified_amounts]
    if not isinstance(qualified_descriptions, list): qualified_descriptions = [qualified_descriptions]
    
    for i in range(max(len(qualified_amounts), len(qualified_descriptions))):
        amount_str = qualified_amounts[i] if i < len(qualified_amounts) else None
        desc_str = qualified_descriptions[i] if i < len(qualified_descriptions) else None
        notes_str = notes_parts[idx] if idx < len(notes_parts) else None
        idx += 1

        if amount_str or desc_str or notes_str:
            doc_id = f"{record_id_prefix}{len(processed_documents) + 1:03d}_qualified"

            recommended_amount_mg = clean_and_convert_value(re.search(r'(\d+)', amount_str).group(1), float) if amount_str and re.search(r'(\d+)', amount_str) else None

            metadata = {
                "source_table": "preparation_guidelines",
                "product_level1": clean_and_convert_value(product_info.get("产品一级目录"), str),
                "product_level2": clean_and_convert_value(product_info.get("产品二级目录"), str),
                "product_level3": clean_and_convert_value(product_info.get("产品三级目录"), str),
                "sample_category": clean_and_convert_value(sample_prep_info.get("样本大类"), str),
                "sample_type": clean_and_convert_value(sample_prep_info.get("样本类型"), str),
                "tissue_type": clean_and_convert_value(sample_prep_info.get("组织类型"), str),
                "prep_method": clean_and_convert_value(sample_prep_info.get("样本处理方式"), str),
                "preparation_method_doc": clean_and_convert_value(methods_notes_info.get("样本准备方法"), str),
                "handling_notes": clean_and_convert_value(methods_notes_info.get("取样送样的注意事项"), str),
                "risk_level": "合格",
                "recommended_amount_mg": recommended_amount_mg,
                "qualitative_description_text": clean_and_convert_value(desc_str.replace('；', '').strip(), str) if desc_str else None,
                "notes_full": clean_and_convert_value(notes_str, str),
            }
            
            text_template = (
                "这是一条关于\"{product_level1}\"下\"{product_level3}\"产品的样本制备指南。\n"
                "样本属于\"{sample_category}\"中的\"{sample_type}\"，具体为\"{tissue_type}\"组织。\n"
                "样本处理方式为\"{prep_method}\"。\n"
                "如果建议送样量为{recommended_amount_mg}mg（约\"{qualitative_description_text}\"），则属于\"{risk_level}\"级别。\n"
                "具体说明：{notes_full}\n"
                "样本准备方法请参考文档：\"{preparation_method_doc}\"。\n"
            )
            safe_metadata = {k: v if v is not None else "未知" for k, v in metadata.items()}
            text = text_template.format_map(safe_metadata)

            processed_documents.append({"id": doc_id, "text": text, "metadata": metadata})

    return processed_documents


# 示例数据
sample_data_exp = {
  "项目建库类型": "10X单细胞3‘转录本-抽核(V3试剂)",
  "物种": "小鼠",
  "样本类型": "冻存组织",
  "样本详细类型": "心肌组织",
  "实验方案（解离/抽核）": "抽核",
  "到样温度（℃）": "-80",
  "细胞总量(万)": "10",
  "结团率(%)": "1.2",
  "细胞活率(%)": "0",
  "有核率(%)": "99",
  "捕获细胞数": "14,466",
  "reads/cell": "22,517",
  "基因中位数": "1,259",
  "人工注释结果": "Endothelial(38.6%),Fibroblast(26.34%),Adipocyte(12.52%),Macrophage(10.09%),Pericyte(9.1%),Lymphatic endothelial cell(1.22%),Cardiomyocyte(0.61%),T cell(0.6%),Schwann cell(0.39%),DC(0.34%),B cell(0.18%)",
  "组织消化方案": "温和机械解离",
  "组织消化方案概述": "使用剪刀剪碎后，温和吹打",
  "相关组织用户文章链接": "http://example.com/article1",
  "飞书文档链接": "http://example.com/feishu1",
  "视频直播链接": "http://example.com/video1"
}

sample_data_prep = {
  "product": {
    "产品一级目录": "单细胞空间组学",
    "产品二级目录": "单细胞测序",
    "产品三级目录": "单细胞转录组"
  },
  "sample_preparation": {
    "样本大类": "实体组织类样本",
    "样本类型": "新鲜实体组织",
    "组织类型": "肿瘤",
    "样本处理方式": "解离",
    "recommended_amount": {
      "风险": "约20mg；",
      "合格": ["约50mg；", "约250mg；"]
    },
    "qualitative_description": {
      "风险": "大米大小；",
      "合格": ["绿豆大小；", "黄豆大小；"]
    },
    "methods_and_notes": {
      "样本准备方法": "实体组织送样方法sop",
      "取样送样的注意事项": "避免重复冻融；尽快送样",
      "备注": "风险：可以尝试，细胞量接近上机捕获极限，坏死类组织需根据坏死部位占比来提高送样量；\n合格：可以尝试，细胞量一般可满足上机要求，坏死类组织需根据坏死部位占比来提高送样量；\n合格：细胞量完全足够，满足上机捕获要求，坏死类组织需根据坏死部位占比来提高送样量；"
    }
  }
}

# 运行预处理
processed_exp_docs = preprocess_experiment_data(sample_data_exp)
processed_prep_docs = preprocess_preparation_data(sample_data_prep)

all_processed_docs = processed_exp_docs + processed_prep_docs

# 将结果保存到 JSONL 文件
output_filepath = "processed_data.jsonl"
with open(output_filepath, 'w', encoding='utf-8') as f:
    for doc in all_processed_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')

print(f"预处理完成，数据已保存到 {output_filepath}")
for doc in all_processed_docs:
    print(json.dumps(doc, indent=2, ensure_ascii=False))
```

---

### **Milvus Schema 定义 (示例)**

在Milvus中创建collection时，你需要根据上述预处理后的`metadata`字段来定义schema。

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

# 定义 Milvus 字段
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768), # 假设使用768维的Embedding模型

    # 公共字段
    FieldSchema(name="source_table", dtype=DataType.VARCHAR, max_length=64),

    # 实验数据表 (experiment_data) 字段
    FieldSchema(name="project_library_type", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="species", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="sample_type", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="sample_detailed_type", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="prep_method", dtype=DataType.VARCHAR, max_length=64), # 解离/抽核
    FieldSchema(name="arrival_temp_celsius", dtype=DataType.FLOAT),
    FieldSchema(name="total_cells_10k", dtype=DataType.FLOAT),
    FieldSchema(name="clumping_rate_percent", dtype=DataType.FLOAT),
    FieldSchema(name="cell_viability_percent", dtype=DataType.FLOAT),
    FieldSchema(name="nucleated_rate_percent", dtype=DataType.FLOAT),
    FieldSchema(name="captured_cells", dtype=DataType.INT64),
    FieldSchema(name="reads_per_cell", dtype=DataType.INT64),
    FieldSchema(name="median_genes", dtype=DataType.INT64),
    FieldSchema(name="annotation_results_full", dtype=DataType.VARCHAR, max_length=4096),
    # 使用JSON类型来存储列表和字典，Milvus 2.2+支持
    FieldSchema(name="annotation_cell_types_list", dtype=DataType.JSON), 
    FieldSchema(name="annotation_cell_type_details", dtype=DataType.JSON),
    FieldSchema(name="tissue_digestion_protocol", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="tissue_digestion_summary", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="related_article_link", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="feishu_doc_link", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="video_link", dtype=DataType.VARCHAR, max_length=512),

    # 样本制备指南表 (preparation_guidelines) 字段
    FieldSchema(name="product_level1", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="product_level2", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="product_level3", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="sample_category", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="tissue_type", dtype=DataType.VARCHAR, max_length=128),
    # prep_method 与实验数据表同名，可以复用或区分
    # FieldSchema(name="prep_method_guideline", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="preparation_method_doc", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="handling_notes", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="risk_level", dtype=DataType.VARCHAR, max_length=64), # 风险/合格
    FieldSchema(name="recommended_amount_mg", dtype=DataType.FLOAT),
    FieldSchema(name="qualitative_description_text", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="notes_full", dtype=DataType.VARCHAR, max_length=4096),
]

# 创建 CollectionSchema
schema = CollectionSchema(
    fields, 
    "single_cell_rag_collection", 
    "Collection for single cell RAG system"
)

# 创建 Collection
# collection = Collection("single_cell_rag_collection", schema)

# 创建索引 (用于加速搜索和过滤)
# collection.create_index(
#     field_name="embedding", 
#     index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
# )
# collection.create_index(field_name="source_table")
# collection.create_index(field_name="species")
# collection.create_index(field_name="sample_detailed_type")
# collection.create_index(field_name="prep_method")
# collection.create_index(field_name="annotation_cell_types_list") # 对JSON字段的数组元素创建索引
# collection.create_index(field_name="risk_level")
# collection.create_index(field_name="recommended_amount_mg")
# # ... 为所有需要进行过滤的metadata字段创建索引

# collection.load() # 加载 collection 到内存
```

**注意事项：**

*   **`max_length`：** `VARCHAR` 字段的 `max_length` 需要根据你的实际数据调整，确保不会截断数据。对于长文本，可以设置为较大的值。
*   **JSON 字段：** Milvus 2.2+ 对 `DataType.JSON` 有很好的支持，可以直接对其内部元素创建索引，这对于 `annotation_cell_types_list` 和 `annotation_cell_type_details` 非常有用，可以直接在查询中使用类似 `annotation_cell_types_list array_contains "Macrophage"` 的表达式。
*   **重复字段名：** 如果两张表的某些字段（如 `prep_method`）在预处理后名称相同但含义略有不同，或者它们的值域完全互斥，可以合并到一个字段中。如果它们的含义或值域可能重叠并导致歧义，最好在字段名上做出区分（例如 `prep_method_exp` 和 `prep_method_prep`）。在这个方案中，我假设它们是兼容的。
*   **Robustness：** 实际生产环境的预处理脚本需要更健壮的错误处理和数据验证逻辑。

这个详细的预处理方案应该能很好地支持你的Milvus检索和RAG构建。
