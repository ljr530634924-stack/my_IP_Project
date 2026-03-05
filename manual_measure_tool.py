import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk  # 需要 pip install pillow
import gc

# 尝试导入 openpyxl，如果失败则提示
try:
    from openpyxl import Workbook
except ImportError:
    print("Error: openpyxl library is missing. Please install it using 'pip install openpyxl'")
    exit()

# --- Configuration ---
MEASURE_RADIUS = 70       # 原图上的测量半径 (像素)

class ManualMeasurer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Manual Measurement Tool (Scrollable & Zoomable)")
        self.root.geometry("1200x800")

        # Data
        self.points = []      # 存储选中的点 (x, y) -> 原图坐标
        self.raw_ch00 = None  # BGR numpy array
        self.vis_ch00 = None  # RGB numpy array (normalized)
        
        self.tk_img = None    # Keep reference to prevent GC
        self.zoom_level = 1.0
        self.ch00_path = ""
        self.ch01_path = ""   # Store path, load only when needed
        
        self.raw_w = 0
        self.raw_h = 0
        
        # UI Setup
        self._setup_ui()
        
        # Auto-load on start
        self.root.after(100, self.load_images)

    def _setup_ui(self):
        # 1. Toolbar
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_load = tk.Button(toolbar, text="Load New Pair", command=self.load_images)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Label(toolbar, text="Zoom:").pack(side=tk.LEFT, padx=5)
        self.scale_zoom = tk.Scale(toolbar, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.on_zoom_change)
        self.scale_zoom.set(1.0)
        self.scale_zoom.pack(side=tk.LEFT, padx=5)
        
        btn_undo = tk.Button(toolbar, text="Undo Point (Right Click)", command=self.undo_point)
        btn_undo.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(toolbar, text="Save & Export (Enter)", command=self.save_data, bg="#ddffdd")
        btn_save.pack(side=tk.LEFT, padx=20)
        
        self.lbl_status = tk.Label(toolbar, text="Please load images.")
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # 2. Canvas Area with Scrollbars
        frame_main = tk.Frame(self.root)
        frame_main.pack(fill=tk.BOTH, expand=True)
        
        self.v_scroll = Scrollbar(frame_main, orient=tk.VERTICAL)
        self.h_scroll = Scrollbar(frame_main, orient=tk.HORIZONTAL)
        
        self.canvas = Canvas(frame_main, bg="#202020",
                                xscrollcommand=self.h_scroll.set,
                                yscrollcommand=self.v_scroll.set)
        
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)
        
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bindings
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", lambda e: self.undo_point())
        self.root.bind("<Return>", lambda e: self.save_data())
        self.root.bind("<space>", lambda e: self.save_data())

    def run(self):
        self.root.mainloop()

    def load_images(self):
        """选择并加载图片"""
        file_path = filedialog.askopenfilename(
            title="Select ch00 image (ch01 will be loaded automatically)",
            filetypes=[("TIFF images", "*.tif"), ("All files", "*.*")]
        )
        if not file_path:
            return

        # --- Memory Cleanup: Release old images before loading new ones ---
        self.raw_ch00 = None
        self.vis_ch00 = None
        self.tk_img = None
        gc.collect()

        self.ch00_path = file_path
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # 自动寻找 ch01
        if "ch00" in filename:
            ch01_path = os.path.join(directory, filename.replace("ch00", "ch01"))
        else:
            print("Warning: Filename does not contain 'ch00'. Trying to guess ch01.")
            ch01_path = os.path.join(directory, filename.replace("ch00", "ch01"))

        if not os.path.exists(ch01_path):
            messagebox.showerror("Error", f"Could not find corresponding ch01 file:\n{ch01_path}")
            return
        
        self.ch01_path = ch01_path # Save path for later use

        # 读取原图 (只读 ch00)
        self.raw_ch00 = cv2.imread(self.ch00_path, cv2.IMREAD_UNCHANGED)
        
        if self.raw_ch00 is None:
            messagebox.showerror("Error", "Failed to load ch00 image.")
            return

        self.raw_h, self.raw_w = self.raw_ch00.shape[:2]
        self.points = [] # Reset points
        
        # 预处理显示用的图像 (归一化 + 转RGB) - 仅处理 ch00
        self.vis_ch00 = self.normalize_image(self.raw_ch00)
        
        # 自动设置合适的初始缩放比例 (适应屏幕高度)
        screen_h = self.root.winfo_screenheight()
        target_h = screen_h * 0.6
        initial_zoom = target_h / self.raw_h
        self.scale_zoom.set(round(initial_zoom, 2))
        
        self.refresh_canvas()
        self.lbl_status.config(text=f"Loaded: {filename}")

    def normalize_image(self, img):
        """将 16-bit 或任意图像归一化到 0-255 (8-bit) 并转为 RGB 用于显示"""
        # 优化：先降采样计算百分位数，避免全图转 float32 爆内存
        # 步长取 10，相当于只用 1/100 的像素计算统计值，速度快且内存低
        subsample = img[::10, ::10]
        vmin, vmax = np.percentile(subsample, (1, 99))
        
        # 使用 OpenCV 的 convertScaleAbs 进行快速线性变换和转 8-bit
        # 公式: dst = src * alpha + beta
        # 我们想要: (x - vmin) / (vmax - vmin) * 255
        if vmax > vmin:
            alpha = 255.0 / (vmax - vmin)
            beta = -vmin * alpha
        else:
            alpha = 0
            beta = 0
        
        img_8u = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 转为 RGB (PIL 需要 RGB)
        if img_8u.ndim == 2:
            img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_BGR2RGB)
        return img_rgb

    def on_zoom_change(self, val):
        self.zoom_level = float(val)
        self.refresh_canvas()

    def refresh_canvas(self):
        if self.vis_ch00 is None:
            return
        
        # 1. 根据缩放比例调整图像大小
        new_w = int(self.raw_w * self.zoom_level)
        new_h = int(self.raw_h * self.zoom_level)
        
        # 使用 OpenCV resize 比较快
        resized_ch00 = cv2.resize(self.vis_ch00, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 2. 单图显示，不再拼接
        # combined = np.hstack((resized_ch00, resized_ch01))
        
        # 3. 转为 Tkinter 图像
        pil_img = Image.fromarray(resized_ch00)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        # 4. 更新 Canvas
        self.canvas.delete("all") # 清除所有
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # 5. 重绘已选点
        self.redraw_annotations()

    def redraw_annotations(self):
        """重绘所有已选点"""
        self.canvas.delete("overlay") # 清除旧的标记
        
        scale = self.zoom_level
        r = MEASURE_RADIUS * scale
        
        for i, (px, py) in enumerate(self.points):
            cx = px * scale
            cy = py * scale
            
            # 绘制左图圆圈 (ch00)
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#00FF00", width=2, tags="overlay")
            self.canvas.create_text(cx, cy, text=str(i+1), fill="#00FF00", font=("Arial", 12, "bold"), tags="overlay")

    def on_mouse_move(self, event):
        if self.vis_ch00 is None: return
        
        # 获取 Canvas 坐标 (考虑滚动条偏移)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        scale = self.zoom_level
        rel_x = canvas_x
            
        rel_y = canvas_y
        
        # 绘制光标预览 (临时)
        self.canvas.delete("cursor")
        r = MEASURE_RADIUS * scale
        
        # 左侧光标
        self.canvas.create_oval(rel_x-r, rel_y-r, rel_x+r, rel_y+r, outline="yellow", width=1, tags="cursor")

    def on_left_click(self, event):
        if self.vis_ch00 is None: return
        
        # 获取 Canvas 坐标
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        scale = self.zoom_level
        rel_x = canvas_x
            
        # 计算原图坐标
        raw_x = int(rel_x / scale)
        raw_y = int(canvas_y / scale)
        
        # 限制坐标在图像范围内
        raw_x = max(0, min(raw_x, self.raw_w - 1))
        raw_y = max(0, min(raw_y, self.raw_h - 1))
        
        self.points.append((raw_x, raw_y))
        self.redraw_annotations()
        self.lbl_status.config(text=f"Added point {len(self.points)} at ({raw_x}, {raw_y})")

    def undo_point(self):
        if self.points:
            p = self.points.pop()
            self.redraw_annotations()
            self.lbl_status.config(text=f"Removed point at {p}")

    def save_data(self):
        if not self.points:
            print("No points selected. Exiting without saving.")
            messagebox.showwarning("Warning", "No points selected.")
            return

        print("Calculating and saving...")
        
        # --- Lazy Load ch01 just for measurement ---
        if not os.path.exists(self.ch01_path):
            messagebox.showerror("Error", "ch01 file missing.")
            return
        raw_ch01 = cv2.imread(self.ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            messagebox.showerror("Error", "Failed to load ch01.")
            return

        # 1. 排序：按 X 轴坐标从小到大
        # 使用 enumerate 保留原始顺序信息(如果需要)，这里主要按 x 排序
        # data 结构: {'x': x, 'y': y}
        data_list = [{"x": p[0], "y": p[1]} for p in self.points]
        data_list.sort(key=lambda p: p["x"])
        
        # 2. 准备输出
        results = []
        
        # 重新加载 ch00 用于画图 (ch00_MC.png)，这次用原分辨率
        # 注意：normalize_image 返回的是 RGB，OpenCV 保存需要 BGR
        vis_mc = cv2.cvtColor(self.vis_ch00, cv2.COLOR_RGB2BGR)
        
        for idx, item in enumerate(data_list):
            new_id = idx + 1
            cx, cy = item["x"], item["y"]
            
            # 在 ch01 原图上测量
            mask = np.zeros(raw_ch01.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), MEASURE_RADIUS, 255, -1)
            mean_val = cv2.mean(raw_ch01, mask=mask)[0]
            
            results.append([new_id, cx, cy, mean_val])
            
            # 在 ch00_MC 上画图
            cv2.circle(vis_mc, (cx, cy), MEASURE_RADIUS, (0, 255, 0), 4) # 线条粗一点
            cv2.putText(vis_mc, str(new_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        # Clean up ch01 immediately
        del raw_ch01
        gc.collect()

        # Calculate statistics
        all_means = [r[3] for r in results]
        global_mean = np.mean(all_means) if all_means else 0
        global_std = np.std(all_means, ddof=1) if len(all_means) > 1 else 0

        # 3. 保存文件
        directory = os.path.dirname(self.ch00_path)
        base_name = os.path.splitext(os.path.basename(self.ch00_path))[0]
        
        # 保存图片
        mc_path = os.path.join(directory, f"{base_name}_00visualization.png")
        cv2.imwrite(mc_path, vis_mc)
        
        # 保存 Excel
        excel_path = os.path.join(directory, f"{base_name}_results.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "Manual Measurements"
        ws.append(["ID", "Center_X", "Center_Y", "Mean_Intensity"])
        for row in results:
            ws.append(row)
            
        # Add footer statistics
        ws.append([])
        ws.append(["Statistics"])
        ws.append(["Global Mean", global_mean])
        ws.append(["Global Std Dev", global_std])
        
        wb.save(excel_path)
        
        print(f"Saved:\n  Image: {mc_path}\n  Excel: {excel_path}")
        messagebox.showinfo("Success", f"Saved {len(results)} points.\nCheck folder for _00visualization.png and .xlsx files.")

if __name__ == "__main__":
    app = ManualMeasurer()
    app.run()