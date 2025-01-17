import os
from tqdm import tqdm
import csv

def collect_csv_files(root_dir, output_file):
    # 打开输出CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入CSV头部
        writer.writerow(['file_id', 'file_path'])
        
        # 递归遍历目录
        for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
            # 遍历当前目录下的文件
            for filename in filenames:
                # 检查文件是否为.wav文件
                if filename.lower().endswith('.npy'):
                    # 获取文件的绝对路径
                    abs_path = os.path.abspath(os.path.join(dirpath, filename))
                    # 写入文件名（不带扩展名）和绝对路径
                    file_id = os.path.splitext(filename)[0].replace(".npy", "")
                    writer.writerow([file_id, abs_path])

def collect_wavcsv_files(root_dir, output_file):
    # 打开输出CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入CSV头部
        writer.writerow(['file_id', 'data','file_path'])
        # 递归遍历目录
        for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
            # 遍历当前目录下的文件
            for filename in filenames:
                # 检查文件是否为.wav文件
                if filename.lower().endswith('.wav'):
                    # 获取文件的绝对路径
                    abs_path = os.path.abspath(os.path.join(dirpath, filename))
                    # 写入文件名（不带扩展名）和绝对路径
                    file_id = os.path.splitext(filename)[0].replace(".wav", "")
                    writer.writerow([file_id, "diffsinger" , abs_path])


def collect_wav_files(root_dir, output_file):
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 递归遍历目录
        for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
            # 遍历当前目录下的文件
            for filename in filenames:
                # 检查文件是否为.wav文件
                if filename.lower().endswith('.npy'):
                    # 获取文件的绝对路径
                    abs_path = os.path.abspath(os.path.join(dirpath, filename))
                    # 写入文件名和绝对路径，用|分隔
                    filename = filename.replace(".npy","")
                    f.write(f"{filename}|{abs_path}\n")

# 使用示例
if __name__ == "__main__":
    # 设置要扫描的根目录（当前目录）
    root_directory = "/ssd12/exec/wanghx04/MakeDiffSinger/SOFA/segments"
    # root_directory = "/ssd6/other/liangzq02/data/RIR_44k/train"
    # 设置输出文件名
    # output_filename = "source_rir_ori.txt"

    output_filename = "sourc_wav.csv"
    
    collect_wavcsv_files(root_directory, output_filename)
    print(f"文件列表已保存到 {output_filename}")