import os
import glob

# --- 1. 配置 ---
# 请将此路径更改为您要清理的目标文件夹
TARGET_DIR = r"F:\Jinrui\qCAP_QuantaRed_750um\Biotin_4Conc\t45min60min" 

def main():
    """
    此脚本会删除指定文件夹 (TARGET_DIR) 中所有文件名包含 "visualization" 的文件。
    """
    print(f"=== 开始清理调试文件 ===")
    print(f"目标文件夹: {os.path.abspath(TARGET_DIR)}")

    if not os.path.isdir(TARGET_DIR):
        print(f"[错误] 文件夹不存在: {TARGET_DIR}")
        return

    # --- 2. 查找所有文件名中包含 "visualization" 的文件 ---
    # 模式 '*visualization*' 会匹配文件名中任何位置包含 'visualization' 的文件。
    search_pattern = os.path.join(TARGET_DIR, "*map*")
    files_to_delete = glob.glob(search_pattern)
    
    if not files_to_delete:
        print("没有找到包含 'visualization' 的文件，无需任何操作。")
        print("=== 清理完成 ===")
        return

    # --- 3. 列出文件并请求确认 ---
    print("\n以下文件将被永久删除:")
    for f in files_to_delete:
        print(f"  - {os.path.basename(f)}")
    
    # 安全检查：需要用户确认
    confirm = input("\n您确定要删除这些文件吗？ (yes/no): ").lower().strip()

    if confirm != 'yes':
        print("操作已被用户取消。")
        print("=== 清理完成 ===")
        return

    # --- 4. 删除文件 ---
    print("\n正在删除文件...")
    deleted_count = 0
    for f_path in files_to_delete:
        try:
            os.remove(f_path)
            print(f"  - 已删除: {os.path.basename(f_path)}")
            deleted_count += 1
        except Exception as e:
            print(f"  - [错误] 删除失败 {os.path.basename(f_path)}: {e}")

    print(f"\n=== 清理完成 ===")
    print(f"成功删除 {len(files_to_delete)} 个文件中的 {deleted_count} 个。")

if __name__ == "__main__":
    main()