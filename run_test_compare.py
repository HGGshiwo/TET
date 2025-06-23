file1 = r"D:\work\实时对话\VideoTree-e2e2\outputs\0621\answer3_dino_image_gpt-4.1-2025-04-14_48_crop.txt"
file2 = r"D:\work\实时对话\VideoTree-e2e2\outputs\0621\answer3_dino_image_gpt-4.1-2025-04-14_48.txt"

# 读取文件内容为集合
def read_file_to_set(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

# 读取A、B文件
setA = read_file_to_set(file1)
setB = read_file_to_set(file2)

# 计算特有和共有元素
only_in_A = setA - setB
only_in_B = setB - setA
in_both = setA & setB

# 输出结果
total = 1000 # 假设测试的是1000题
print(f"{file1}文件特有元素：({len(only_in_A)/total})[{len(only_in_A)}/{total}]")
for item in sorted(only_in_A):
    print(item)
print("=" * 20)
print(f"{file2}特有元素：({len(only_in_B)/total})[{len(only_in_B)}/{total}]")
for item in sorted(only_in_B):
    print(item)
print("=" * 20)
print(f"共有元素：({len(in_both)/total})[{len(in_both)}/{total}]")
for item in sorted(in_both):
    print(item)# 读取文件内容为集合