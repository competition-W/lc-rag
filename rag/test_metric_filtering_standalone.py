#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试数据指标和实验指标的过滤功能
"""

def _aggregate_numeric_metrics(context_nodes, query_text=""):
    """
    聚合处理数字指标，计算平均值、中位数等统计信息
    根据查询文本区分数据指标和实验指标
    """
    if not context_nodes:
        return ""
    
    # 定义指标类型分类
    experimental_metrics = [
        "col_zu_zhi_zhong_liang_shu_zhi", "col_xbzl\n（w）", "col_jie_tuan_lv", 
        "col_xi_bao_huo_lv", "col_you_he_lv", "col_bu_huo_xi_bao_shu", 
        "col_hszl\n（rinz）"
    ]
    
    data_metrics = [
        "col_shu_ju_liang", "col_ji_yin_zhong_liang_shu", "col_ji_yin_zhong_wei_shu", 
        "col_zsjg_zztyxzs"
    ]
    
    # 定义指标名称映射，将英文列名转换为友好的中文名称
    metric_name_mapping = {
        "col_zu_zhi_zhong_liang_shu_zhi": "组织重量",
        "col_shu_ju_liang": "数据量",
        "col_ji_yin_zhong_wei_shu": "基因中位数",
        "col_xbzl\n（w）": "细胞总量",
        "col_jie_tuan_lv": "结团率",
        "col_xi_bao_huo_lv": "细胞活率",
        "col_you_he_lv": "有效核率",
        "col_bu_huo_xi_bao_shu": "捕获细胞数",
        "col_hszl\n（rinz）": "核碎片率",
        "col_zsjg_zztyxzs": "注释结果-组织特异性指标"
    }
    
    # 分析查询类型
    lower_query = query_text.lower() if query_text else ""
    need_data_metrics = "数据指标" in lower_query
    need_experimental_metrics = "实验指标" in lower_query
    
    # 如果没有明确指定，默认聚合所有数字指标
    if not need_data_metrics and not need_experimental_metrics:
        need_data_metrics = True
        need_experimental_metrics = True
    
    # 收集指定类型的数字指标
    numeric_metrics = {}
    
    for node in context_nodes:
        meta = node.get("metadata", {})
        for key, value in meta.items():
            # 根据查询类型过滤指标
            is_data_metric = key in data_metrics
            is_experimental_metric = key in experimental_metrics
            
            # 检查是否需要当前指标类型
            should_process = False
            if need_data_metrics and is_data_metric:
                should_process = True
            elif need_experimental_metrics and is_experimental_metric:
                should_process = True
            elif not is_data_metric and not is_experimental_metric:  # 未分类的指标默认处理
                should_process = True
            
            if should_process:
                # 尝试转换为数字
                try:
                    # 清理字符串，去除空格和特殊字符
                    clean_value = str(value).strip()
                    if clean_value and clean_value != "/" and clean_value != "(空)":
                        # 处理带千分位的数字，如12,033
                        clean_value = clean_value.replace(",", "")
                        # 转换为浮点数
                        num_value = float(clean_value)
                        if key not in numeric_metrics:
                            numeric_metrics[key] = []
                        numeric_metrics[key].append(num_value)
                except (ValueError, TypeError):
                    # 不是数字，跳过
                    continue
    
    if not numeric_metrics:
        return ""
    
    # 生成聚合结果
    aggregate_result = "\n\n**数字指标统计汇总**\n"
    aggregate_result += "| 指标名称 | 样本数量 | 平均值 | 最小值 | 最大值 | 中位数 |\n"
    aggregate_result += "|----------|----------|--------|--------|--------|--------|\n"
    
    for metric_name, values in numeric_metrics.items():
        # 计算统计值
        sample_count = len(values)
        avg_value = sum(values) / sample_count
        min_value = min(values)
        max_value = max(values)
        # 计算中位数
        sorted_values = sorted(values)
        mid_index = sample_count // 2
        median_value = sorted_values[mid_index] if sample_count % 2 == 1 else (sorted_values[mid_index - 1] + sorted_values[mid_index]) / 2
        
        # 使用友好的中文名称，如果没有映射则使用原名称
        display_name = metric_name_mapping.get(metric_name, metric_name)
        
        # 添加到结果表格
        aggregate_result += f"| {display_name} | {sample_count} | {avg_value:.2f} | {min_value:.2f} | {max_value:.2f} | {median_value:.2f} |\n"
    
    return aggregate_result

def test_metric_filtering():
    """
    测试指标过滤功能
    """
    # 创建模拟数据
    mock_data = [
        {
            "metadata": {
                "col_shu_ju_liang": "10000",  # 数据指标
                "col_ji_yin_zhong_wei_shu": "5000",  # 数据指标
                "col_xbzl\n（w）": "1000",  # 实验指标
                "col_jie_tuan_lv": "5",  # 实验指标
                "col_xi_bao_huo_lv": "90"  # 实验指标
            }
        },
        {
            "metadata": {
                "col_shu_ju_liang": "20000",  # 数据指标
                "col_ji_yin_zhong_wei_shu": "6000",  # 数据指标
                "col_xbzl\n（w）": "2000",  # 实验指标
                "col_jie_tuan_lv": "3",  # 实验指标
                "col_xi_bao_huo_lv": "95"  # 实验指标
            }
        }
    ]
    
    print("=== 测试指标过滤功能 ===")
    
    # 测试1：数据指标查询
    print("\n1. 数据指标查询：")
    result = _aggregate_numeric_metrics(mock_data, "数据指标")
    print(result)
    
    # 检查是否只包含数据指标
    if "数据量" in result and "基因中位数" in result and "细胞总量" not in result and "结团率" not in result:
        print("✅ 数据指标查询测试通过")
    else:
        print("❌ 数据指标查询测试失败")
    
    # 测试2：实验指标查询
    print("\n2. 实验指标查询：")
    result = _aggregate_numeric_metrics(mock_data, "实验指标")
    print(result)
    
    # 检查是否只包含实验指标
    if "细胞总量" in result and "结团率" in result and "细胞活率" in result and "数据量" not in result and "基因中位数" not in result:
        print("✅ 实验指标查询测试通过")
    else:
        print("❌ 实验指标查询测试失败")
    
    # 测试3：默认查询（包含所有指标）
    print("\n3. 默认查询：")
    result = _aggregate_numeric_metrics(mock_data, "")
    print(result)
    
    # 检查是否包含所有指标
    if "数据量" in result and "基因中位数" in result and "细胞总量" in result and "结团率" in result and "细胞活率" in result:
        print("✅ 默认查询测试通过")
    else:
        print("❌ 默认查询测试失败")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_metric_filtering()
