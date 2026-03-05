import os
import shutil
import datetime

def copy_result_files(source_root, dest_root, log_file_path):
    """
    从 source_root 复制文件名包含 'result' 的 .xlsx 文件到 dest_root。
    保持文件夹结构，如果目标文件存在则覆盖。
    将复制的文件路径记录到 log_file_path。
    """
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_root):
        print(f"错误: 源文件夹 '{source_root}' 不存在。")
        return

    print(f"开始处理...")
    print(f"源目录: {source_root}")
    print(f"目标目录: {dest_root}")

    copied_count = 0
    
    try:
        # 打开日志文件准备写入
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"=== 复制日志 - 开始时间: {datetime.datetime.now()} ===\n")
            log_file.write(f"源目录: {source_root}\n")
            log_file.write(f"目标目录: {dest_root}\n")
            log_file.write("-" * 60 + "\n")

            # os.walk 递归遍历源目录
            for root, dirs, files in os.walk(source_root):
                for filename in files:
                    # 1. 筛选条件：后缀是 .xlsx 且 文件名包含 results (不区分大小写)
                    if filename.lower().endswith(".xlsx") and "results" in filename.lower():
                        
                        # 构建源文件的完整路径
                        src_file_path = os.path.join(root, filename)
                        
                        # 2. 计算相对路径，用于在目标文件夹中重建结构
                        # 例如：源文件在 D:\Source\A\B\data.xlsx，相对路径就是 A\B
                        rel_path = os.path.relpath(root, source_root)
                        
                        # 构建目标文件夹路径
                        dest_folder_path = os.path.join(dest_root, rel_path)
                        
                        # 3. 如果目标文件夹不存在，则创建它
                        os.makedirs(dest_folder_path, exist_ok=True)
                        
                        # 构建目标文件的完整路径
                        dest_file_path = os.path.join(dest_folder_path, filename)
                        
                        try:
                            # 4. 复制文件
                            # shutil.copy2 会保留文件的元数据（如创建时间），且默认会覆盖同名文件
                            shutil.copy2(src_file_path, dest_file_path)
                            
                            # 5. 记录日志和打印进度
                            print(f"[复制] {filename}")
                            log_file.write(f"{src_file_path}\n")
                            copied_count += 1
                            
                        except Exception as e:
                            error_msg = f"[错误] 无法复制 {src_file_path}: {e}"
                            print(error_msg)
                            log_file.write(f"{error_msg}\n")

            # 写入结束信息
            log_file.write("-" * 60 + "\n")
            log_file.write(f"完成时间: {datetime.datetime.now()}\n")
            log_file.write(f"共复制文件数: {copied_count}\n")

    except Exception as e:
        print(f"无法创建日志文件: {e}")

    print(f"\n处理完成！共复制了 {copied_count} 个文件。")
    print(f"详细日志已保存至: {os.path.abspath(log_file_path)}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 请在这里修改您的文件夹路径
    # 建议使用 r"" 原始字符串格式以避免反斜杠转义问题
    
    SOURCE_FOLDER = r"F:\jinrui_data"      # 源文件夹 (您要搜索的地方)
    DESTINATION_FOLDER = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Jinrui\hard drive data analysis" # 目标文件夹 (您要备份到的地方)
    LOG_FILE = "copy_log.txt"                          # 日志文件名

    # 执行函数
    copy_result_files(SOURCE_FOLDER, DESTINATION_FOLDER, LOG_FILE)