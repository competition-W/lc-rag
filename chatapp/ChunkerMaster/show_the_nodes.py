'''打印nodes的内容，方便调试和查看切块结果'''
def show_the_nodes(nodes):
    for i in range(0,3):
        # 简单查看
        print(f"*********************THIS IS {i}*********************")
        print(nodes[i].text[:500])