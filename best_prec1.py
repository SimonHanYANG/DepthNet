import re
import argparse

parser = argparse.ArgumentParser(description='Best Param in Training')

parser.add_argument('--logroot', metavar='logroot', default='hannet50_hs73_hannet50', 
                    help="log root name")
parser.add_argument('--top', default=1, type=int, 
                    help="top 1 or top 5 acc.")

global args
args = parser.parse_args()

# 用于从每行中提取数字的正则表达式
pattern = re.compile(r"tensor\((\d+\.\d+), device='cuda:\d+'\)")

def extract_max_value(file_name):
    max_value = float('-inf')  # 初始化最大值为无穷小
    with open(file_name, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # 将匹配的字符串转换为浮点数
                value = float(match.group(1))
                if value > max_value:
                    max_value = value  # 更新最大值
    return max_value

def save_best_value(best_value, output_file):
    with open(output_file, 'w') as file:
        file.write(str(best_value))

# 假设输入文件名为 'input.txt'
# input_file_name = 'runs/hannet18_hs73/val_prec1.txt'
logrootname = args.logroot
input_root = "runs/"
if args.top == 1:
    input_file_suffix = "/val_prec1.txt"
elif args.top == 5:
    input_file_suffix = "/val_prec5.txt"
else:
    print("Wrong top param!!!")
    
# input_file_name = 'runs/hannet50_hs73_hannet50/val_prec1.txt'
input_file_name = input_root + logrootname + input_file_suffix
print("Input file: {}.".format(input_file_name))

# 输出文件名为 'best.txt'
# output_file_name = 'best.txt'

# 提取最大值
max_value = extract_max_value(input_file_name)
# 保存最大值
# save_best_value(max_value, output_file_name)

print(f"The maximum value extracted is {max_value}.")
# print(f"The maximum value extracted is {max_value} and has been saved to {output_file_name}.")