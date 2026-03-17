import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_data
from dataset.builder import build_dataset
import matplotlib as mpl
print(mpl.get_cachedir())   # 查看缓存目录

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Noto Serif CJK SC']   # 使用正确的宋体名称
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题  

base_dir = r"D:\work\实时对话\TET\train\outputs\cot0104"
dataset_config = "./configs/dataset.yml"

def summary(counter):
    total = sum(counter.values())
    for k, v in counter.items():
        print("{}: {} ({:.2f}%)".format(k, v, v / total * 100))
    
# 2. 定义分类函数（利用关键词命中）
def classify_question(question):
    q = question.lower() # 转小写
    
    # 按照优先级匹配：推理 > 时序动作 > 空间 > 属性感知
    if any(word in q for word in ['why', 'how', 'reason', 'cause', 'purpose']):
        return 'Causality & Reasoning (因果推理)'
    
    elif any(word in q for word in ['doing', 'happen', 'before', 'after', 'next', 'action']):
        return 'Action & Temporal (动作与时序)'
        
    elif any(word in q for word in ['where', 'left', 'right', 'position']):
        return 'Spatial (空间位置)'
        
    elif any(word in q for word in ['color', 'how many', 'who', 'which']):
        return 'Object & Attribute (物体与属性)'
        
    else:
        # 如果啥都没匹配上，兜底算作基础物体识别（因为大部分 What 开头的都是问物体）
        return 'Object & Attribute (物体与属性)'

categories_all = []
total = []
for name in ["Egoschema_subset", "IntentQA_test", "MLVU_test", "MVBench", "NextQA_test", "VideoMME_short"]:
    # 3. 开始统计
    categories = []
    if name == "NextQA_test":
        path = "nextmc_test"
    else:
        path = name.lower()
    data = build_dataset(dataset_config, path, is_training=False)
    for item in tqdm(data, desc=f"Processing {name}"):
        cat = classify_question(item['question'])
        categories.append(cat)
        categories_all.append(cat)
        total.append(name)    
    counts = Counter(categories)
    print(f"{name}分类统计结果:")
    summary(counts)

counts = Counter(categories_all)
print(f"总体分类统计结果:")
summary(counts)

counts2 = Counter(total)
print(f"每个数据集的数量统计结果:")
summary(counts2)
    
# 4. 直接画饼图！(保存为 categories_pie.png)
labels = counts.keys()
sizes = counts.values()
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'] # 选几个好看的颜色

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
# plt.title('Question Type Distribution in Our CoT Dataset')
plt.axis('equal') # 保证画出来是正圆
plt.savefig('categories_pie.png', dpi=300, bbox_inches='tight')
print("饼图已生成并保存为 categories_pie.png！")