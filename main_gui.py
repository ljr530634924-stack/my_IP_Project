import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import sys
import os

# 导入您的分析脚本
# 注意：这些脚本必须在同一目录下
import batch_run_DSA
import batch_run_WN_MC
import batch_run_WN_DB_MC

class AnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Image Analysis Tool")
        self.root.geometry("700x600")

        # --- 1. 文件夹选择区域 ---
        frame_folder = tk.LabelFrame(root, text="1. Select Input Folder", padx=10, pady=10)
        frame_folder.pack(fill="x", padx=10, pady=5)

        self.entry_path = tk.Entry(frame_folder)
        self.entry_path.pack(side="left", fill="x", expand=True, padx=(0, 5))

        btn_browse = tk.Button(frame_folder, text="Browse...", command=self.browse_folder)
        btn_browse.pack(side="right")

        # --- 2. 算法选择区域 ---
        frame_method = tk.LabelFrame(root, text="2. Select Analysis Method", padx=10, pady=10)
        frame_method.pack(fill="x", padx=10, pady=5)

        self.method_var = tk.StringVar(value="DSA")

        modes = [
            ("DSA (Direct Signal Analysis - for particles without notch)", "DSA"),
            ("WN_MC (for particles with notch + clean background)", "WN_MC"),
            ("WN_DB_MC (for particles with notch + foggy/noisy backgrounds)", "WN_DB_MC")
        ]

        for text, mode in modes:
            rb = tk.Radiobutton(frame_method, text=text, variable=self.method_var, value=mode, font=("Arial", 10))
            rb.pack(anchor="w", pady=2)

        # --- 3. 控制区域 ---
        self.btn_run = tk.Button(root, text="RUN ANALYSIS", command=self.start_analysis_thread, 
                                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_run.pack(fill="x", padx=20, pady=15)

        # --- 4. 日志输出区域 ---
        frame_log = tk.LabelFrame(root, text="Logs / Output", padx=10, pady=10)
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)

        self.txt_log = scrolledtext.ScrolledText(frame_log, state='disabled', height=10, font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True)

        # 重定向 stdout 和 stderr 到 GUI
        sys.stdout = TextRedirector(self.txt_log)
        sys.stderr = TextRedirector(self.txt_log)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.entry_path.delete(0, tk.END)
            self.entry_path.insert(0, folder_selected)

    def start_analysis_thread(self):
        path = self.entry_path.get().strip()
        if not path:
            messagebox.showerror("Error", "Please select an input folder first.")
            return
        
        if not os.path.isdir(path):
            messagebox.showerror("Error", f"The folder does not exist:\n{path}")
            return

        method = self.method_var.get()
        
        # 锁定按钮防止重复点击
        self.btn_run.config(state="disabled", text="Running... Please Wait", bg="#cccccc")
        
        # 在新线程中运行，防止界面卡死
        thread = threading.Thread(target=self.run_analysis, args=(path, method))
        thread.daemon = True
        thread.start()

    def run_analysis(self, path, method):
        print(f"\n{'='*40}")
        print(f"Starting {method} Analysis...")
        print(f"Folder: {path}")
        print(f"{'='*40}\n")
        
        try:
            if method == "DSA":
                batch_run_DSA.run_batch_dsa(path)
            elif method == "WN_MC":
                batch_run_WN_MC.run_batch_wn_mc(path)
            elif method == "WN_DB_MC":
                batch_run_WN_DB_MC.run_batch_wn_db_mc(path)
            else:
                print(f"[ERROR] Unknown method selected: {method}")
            
            print(f"\n{'='*40}")
            print("ANALYSIS COMPLETE!")
            print(f"{'='*40}\n")
            
            # 任务完成后弹窗提示 (使用 after 在主线程执行)
            self.root.after(0, lambda: messagebox.showinfo("Success", f"{method} analysis finished successfully!"))

        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during execution:\n{str(e)}"))
        finally:
            # 恢复按钮状态
            self.root.after(0, self.reset_button)

    def reset_button(self):
        self.btn_run.config(state="normal", text="RUN ANALYSIS", bg="#4CAF50")

class TextRedirector(object):
    """用于将 print 输出重定向到 tkinter Text 控件"""
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        # 使用 after 方法确保在主线程更新 UI，防止多线程冲突
        self.widget.after(0, self._append_text, str)

    def _append_text(self, str):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END) # 自动滚动到底部
        self.widget.configure(state='disabled')

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()