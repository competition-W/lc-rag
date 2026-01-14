import ast

def get_variable_values_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 执行代码，获取所有全局变量（注意：必须信任文件！）
    local_vars = {}
    exec(code, {}, local_vars)

    result = {}
    tree = ast.parse(code, filename=file_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    if var_name in local_vars:
                        result[var_name] = local_vars[var_name]
    return result

# import ast

# def get_variable_names_from_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         tree = ast.parse(f.read(), filename=file_path)

#     var_names = []
#     for node in ast.walk(tree):
#         if isinstance(node, ast.Assign):
#             for target in node.targets:
#                 if isinstance(target, ast.Name):
#                     var_names.append(target.id)

#     return var_names

# # 用法
# file_path = 'your_file.py'  # 替换为你的文件路径
# variables = get_variable_names_from_file(file_path)
# print(variables)
