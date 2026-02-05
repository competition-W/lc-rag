#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共字段映射模块
定义系统中使用的字段映射关系
定义的是mivlus中真正的schema字段
"""

# 核心字段映射：原始表格列名到数据库字段名的映射
COLUMN_MAPPING = {
    # ---项目经验查询字段映射定义（单细胞项目经验查询（人工细胞注释20260115）.xlsx）---
    # === 实验平台 ===
    "实验平台": "platform_type",
    
    # === 样本信息 ===
    "物种": "species",
    "样本类型": "sample_type_exp",
    "样本详细类型": "sample_detailed_type",
    "实验方案": "experiment_protocol",
    "组织重量": "tissue_weight",
    "组织重量单位": "tissue_weight_unit",
    "定性描述": "qualitative_description",
    "核酸质量RIN值": "rin_score",
    "是否流式": "is_streaming",
    "抗体信息": "antibody_info",
    "流式方案": "streaming_protocol",
    "是否裂红": "is_lysis",
    "是否去死": "is_dead_removal",
    "到样温度": "arrival_temp_celsius",

    # === 实验指标 ===
    "细胞总量": "total_cells_10k",
    "结团率": "clumping_rate_percent",
    "细胞活率": "cell_viability_percent",
    "有核率": "nucleated_rate_percent",

    # === 数据指标 ===
    "捕获细胞数": "captured_cells",
    "reads/cell": "reads_per_cell",
    "基因中位数": "median_genes",
    "人工细胞注释": "annotation_results",
    "人工细胞注释结果": "annotation_results",

    # === 参考资料 ===
    "组织消化方案": "digestion_protocol",
    "相关组织用户文章链接": "related_articles",
    "飞书文档链接": "feishu_doc_link",
    "视频直播链接": "video_live_link",

    
    # ---样本准备字段映射定义（单细胞样本类型细分保存方式.xlsx）---
    # === 产品信息相关 ===
    "产品一级目录": "product_level1",
    "产品二级目录": "product_level2",
    "产品三级目录": "product_level3",
    
    # === 样本制备信息 ===
    "样本大类": "sample_category_prep",
    "样本类型": "sample_type_prep",
    "组织类型": "tissue_type_prep",
    "样本处理方式": "sample_prep_method",
    
    # === 送样要求 ===
    "建议送样量1": "recommended_amount_1",
    "建议送样量2": "recommended_amount_2",
    "建议送样量3": "recommended_amount_3",
    
    "定性描述1": "qualitative_description_1",
    "定性描述2": "qualitative_description_2",
    "定性描述3": "qualitative_description_3",

    # === 质检标准 ===
    "质检标准_RIN": "rin_score",
    "质检标准_细胞活率": "cell_viability_percent",
    "质检标准_细胞总量": "cell_count",
    "质检标准_活细胞浓度": "live_cell_concentration",
    "质检标准_结团率": "clustering_rate",
    "质检标准_有核率": "nucleated_rate_percent",

    # === 方法和注意事项 ===
    "样本准备方法": "sample_preparation_method_doc",
    "取样送样的注意事项": "sampling_notes",
    "备注": "notes_full_text"
}

# 扩展字段映射：中文关键词到数据库字段名的映射
# 用于意图识别和实体抽取
EXTENDED_FIELD_MAPPING = {
    # 物种相关扩展
    "小鼠": "species",
    "人": "species",
    "大鼠": "species",
    "猪": "species",
    "牛": "species",
    "绵羊": "species",
    "羊": "species",
    "蝙蝠": "species",
    "山羊": "species",
    "猕猴": "species",
    
    # 组织类型扩展
    "心脏": "tissue_type_prep",
    "心肌": "tissue_type_prep",
    "肿瘤": "tissue_type_prep",
    "肝脏": "tissue_type_prep",
    "肺": "tissue_type_prep",
    "脑": "tissue_type_prep",
    "脾脏": "tissue_type_prep",
    "肾": "tissue_type_prep",
    "皮肤": "tissue_type_prep",
    "肠道": "tissue_type_prep",
    "肌肉": "tissue_type_prep",
    "全血": "tissue_type_prep",
    "骨髓": "tissue_type_prep",
    "外周血": "tissue_type_prep",
    "PBMC": "tissue_type_prep",
    
    # 样本类型扩展
    "新鲜实体组织": "sample_detailed_type",
    "液体类样本": "sample_detailed_type",
    "细胞类样本": "sample_detailed_type",
    "类器官类样本": "sample_detailed_type",
    "冻存组织": "sample_detailed_type",
    "石蜡包埋组织": "sample_detailed_type",
    
    # 实验方案扩展
    "解离": "experiment_protocol",
    "抽核": "experiment_protocol",
    
    # 样本处理方式扩展
    "解离": "sample_prep_method",
    "抽核": "sample_prep_method",
    
    # 实验平台扩展
    "10X": "platform_type",
    "华大": "platform_type",
    "墨卓": "platform_type",
    
    # 产品类型扩展
    "单细胞转录组": "product_level3",
    "单细胞免疫组": "product_level3",
    "单细胞ATAC": "product_level3",
    "单细胞Flex": "product_level3",
    
    # 样本大类扩展
    "实体组织类样本": "sample_category_prep",
    "液体类样本": "sample_category_prep",
    "细胞类样本": "sample_category_prep",
    "类器官类样本": "sample_category_prep",
    "植物类样本": "sample_category_prep"
}

# 合并所有字段映射
def get_combined_field_mapping():
    """
    获取合并后的字段映射
    包含核心映射和扩展映射
    """
    combined_mapping = COLUMN_MAPPING.copy()
    combined_mapping.update(EXTENDED_FIELD_MAPPING)
    return combined_mapping

# 构建完整的字段映射，包括多对一映射
def build_complete_field_mapping():
    """
    构建完整的字段映射
    包括核心映射、扩展映射和更多的中文关键词映射
    """
    mapping = get_combined_field_mapping()
    
    # 添加更多的中文关键词映射
    additional_mappings = {
        # 组织相关
        "组织": "tissue_type_prep",
        "组织类型": "tissue_type_prep",
        "脑组织": "tissue_type_prep",
        "肺组织": "tissue_type_prep",
        "肝组织": "tissue_type_prep",
        "骨髓血": "tissue_type_prep",
        
        # 样本类型相关
        "样本类型": "sample_type_exp",
        "样本类型": "sample_type_prep",
        "类器官": "sample_detailed_type",
        "石蜡": "sample_detailed_type",
        
        # 实验方案相关
        "实验方案": "experiment_protocol",
        "样本处理方式": "sample_prep_method",
        
        # 实验平台相关
        "实验平台": "platform_type",
        "10X单细胞": "platform_type",
        "华大单细胞": "platform_type",
        "墨卓单细胞": "platform_type",
        
        # 产品类型相关
        "产品类型": "product_level3",
        "单细胞 ATAC": "product_level3",
        "单细胞 Flex": "product_level3",
        
        # 样本大类相关
        "样本大类": "sample_category_prep",
        "实体组织": "sample_category_prep",
        "组织样本": "sample_category_prep",
        "液体": "sample_category_prep",
        "血液": "sample_category_prep",
        "腹水": "sample_category_prep",
        "胸水": "sample_category_prep",
        "细胞": "sample_category_prep",
        "类器官": "sample_category_prep",
        "植物": "sample_category_prep",
        
        # 质检标准相关
        "质检标准": "rin_score",
        "RIN": "rin_score",
        "细胞活率": "cell_viability_percent",
        "结团率": "clustering_rate",
        
        # 数据源表
        "source_table": "source_table"
    }
    
    mapping.update(additional_mappings)
    return mapping
