"""
manual_measure_tool2.py

功能:
1. 手动在 ch00 上标记点，自动测量 ch01 的强度。
2. 逻辑：每4个点一组 (Q1->Q2->Q3->Q4)，循环颜色 (蓝->绿->紫->黄)。
3. 输出：生成包含各象限均值、面积及统计信息的 Excel 表格，以及带有 ID (位于四点中心) 的可视化图。
"""

import cv2
import numpy as np
import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
import gc

# 尝试导入 openpyxl
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font
except ImportError:
    print("Error: openpyxl library is missing. Please install it using 'pip install openpyxl'")
    # 为了防止闪退，这里不强制退出，但在保存时可能会报错或需要降级处理

# --- Configuration ---
MEASURE_RADIUS = 70       # 测量半径 (像素)

class ManualMeasurer2:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Manual Measurement Tool 2 (4-Quadrant Mode)")
        self.root.geometry("1200x800")

        # Data
        self.points = []      # 存储选中的点 [(x, y), ...]
        self.raw_ch00 = None  # BGR numpy array (用于处理)
        self.vis_ch00 = None  # RGB numpy array (用于显示)
        
        self.tk_img = None    # 保持引用防止GC回收
        self.zoom_level = 1.0
        self.is_first_load = True # [New] 标记是否为第一次加载，用于保持缩放
        self.ch00_path = ""
        self.ch01_path = ""
        self.image_files = [] # [New] 当前文件夹下的文件列表
        
        self.raw_w = 0
        self.raw_h = 0
        
        self.measure_radius = MEASURE_RADIUS
        
        # 象限配置
        # Tkinter 颜色 (Hex)
        self.colors_tk = ["#0000FF", "#00FF00", "#FF00FF", "#FFFF00"] # 蓝, 绿, 紫(洋红), 黄
        # OpenCV 颜色 (BGR) - 注意 OpenCV 是 BGR 顺序
        self.colors_cv = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)] # 蓝, 绿, 紫, 黄
        self.quadrant_names = ["Q1 (Blue)", "Q2 (Green)", "Q3 (Purple)", "Q4 (Yellow)"]
        
        self._setup_ui()
        
    def _setup_ui(self):
        # 1. 工具栏
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_load = tk.Button(toolbar, text="Load New Pair", command=self.load_images)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Label(toolbar, text="Zoom:").pack(side=tk.LEFT, padx=5)
        self.scale_zoom = tk.Scale(toolbar, from_=0.1, to=5.0, resolution=0.01, orient=tk.HORIZONTAL, command=self.on_zoom_change)
        self.scale_zoom.set(1.0)
        self.scale_zoom.pack(side=tk.LEFT, padx=5)
        
        tk.Label(toolbar, text="Radius:").pack(side=tk.LEFT, padx=5)
        self.scale_radius = tk.Scale(toolbar, from_=10, to=200, orient=tk.HORIZONTAL, command=self.on_radius_change)
        self.scale_radius.set(self.measure_radius)
        self.scale_radius.pack(side=tk.LEFT, padx=5)
        
        btn_undo = tk.Button(toolbar, text="Undo (Right Click)", command=self.undo_point)
        btn_undo.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(toolbar, text="Save & Export (Enter)", command=self.save_data, bg="#ddffdd")
        btn_save.pack(side=tk.LEFT, padx=20)
        
        # 状态栏 (显示下一个要点的颜色)
        self.lbl_status = tk.Label(toolbar, text="Please load images.", font=("Arial", 12, "bold"), width=40)
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # 2. 主容器 (左右分栏)
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # 2.1 左侧文件列表
        frame_left = tk.Frame(container, width=200, bg="#f0f0f0")
        frame_left.pack(side=tk.LEFT, fill=tk.Y)

        # Scrollbars for Listbox
        sb_y = tk.Scrollbar(frame_left, orient=tk.VERTICAL)
        sb_x = tk.Scrollbar(frame_left, orient=tk.HORIZONTAL)

        self.lst_files = tk.Listbox(frame_left, width=30, font=("Arial", 10), selectmode=tk.SINGLE,
                                    yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        
        sb_y.config(command=self.lst_files.yview)
        sb_x.config(command=self.lst_files.xview)

        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.lst_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lst_files.bind("<<ListboxSelect>>", self.on_file_select)

        # 2.2 右侧画布区域
        frame_right = tk.Frame(container)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.v_scroll = Scrollbar(frame_right, orient=tk.VERTICAL)
        self.h_scroll = Scrollbar(frame_right, orient=tk.HORIZONTAL)
        
        self.canvas = Canvas(frame_right, bg="#202020",
                                xscrollcommand=self.h_scroll.set,
                                yscrollcommand=self.v_scroll.set)
        
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)
        
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 事件绑定
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", lambda e: self.undo_point()) # 右键撤销
        self.root.bind("<Return>", lambda e: self.save_data())      # 回车保存
        self.root.bind("<space>", lambda e: self.save_data())       # 空格保存

    def run(self):
        self.root.mainloop()

    def on_file_select(self, event):
        """列表选择事件：自动保存并加载新图"""
        selection = self.lst_files.curselection()
        if not selection:
            return
        
        index = selection[0]
        selected_file = self.image_files[index]
        
        # 如果点击的是当前正在编辑的文件，不做任何操作
        if selected_file == self.ch00_path:
            return

        # [Auto Save] 切换前尝试自动保存当前工作
        # silent=True 表示如果是正常的保存，不要弹窗打扰
        if self.points:
            print(f"Auto-saving before switching to {os.path.basename(selected_file)}...")
            saved = self.save_data(silent=True)
            # 如果保存过程被用户取消（虽然目前逻辑不会取消，但预留接口），可以在这里 return

        # 加载新图
        self.load_image_file(selected_file)

    def load_images(self):
        """加载图片按钮：打开文件并初始化列表"""
        file_path = filedialog.askopenfilename(
            title="Select ch00 image",
            filetypes=[("TIFF images", "*.tif"), ("All files", "*.*")]
        )
        if not file_path:
            return

        directory = os.path.dirname(file_path)
        
        # 搜索目录下所有 ch00 图片
        search_pattern = os.path.join(directory, "*ch00*.tif")
        files = glob.glob(search_pattern)
        # 简单的自然排序
        self.image_files = sorted(files)
        
        # 填充列表
        self.lst_files.delete(0, tk.END)
        for i, f in enumerate(self.image_files):
            self.lst_files.insert(tk.END, os.path.basename(f))
            
            # Check if results exist and mark green
            base_name = os.path.splitext(os.path.basename(f))[0]
            result_path = os.path.join(directory, f"{base_name}_results.xlsx")
            if os.path.exists(result_path):
                self.lst_files.itemconfig(i, {'bg': '#ddffdd'})

        # 加载选中的这张图
        self.load_image_file(file_path)
        
        # 在列表中高亮选中项
        try:
            idx = self.image_files.index(file_path)
            self.lst_files.selection_clear(0, tk.END)
            self.lst_files.selection_set(idx)
            self.lst_files.see(idx)
        except ValueError:
            pass

    def load_from_directory(self, directory):
        """从指定目录自动加载文件列表并打开第一张图"""
        if not os.path.isdir(directory):
            return

        search_pattern = os.path.join(directory, "*ch00*.tif")
        files = glob.glob(search_pattern)
        self.image_files = sorted(files)

        if not self.image_files:
            messagebox.showinfo("Info", "No suitable images found in folder.")
            return

        self.lst_files.delete(0, tk.END)
        for i, f in enumerate(self.image_files):
            self.lst_files.insert(tk.END, os.path.basename(f))
            base_name = os.path.splitext(os.path.basename(f))[0]
            result_path = os.path.join(directory, f"{base_name}_results.xlsx")
            if os.path.exists(result_path):
                self.lst_files.itemconfig(i, {'bg': '#ddffdd'})

        # 自动加载第一张
        self.load_image_file(self.image_files[0])
        
        # 在列表中选中第一项
        self.lst_files.selection_clear(0, tk.END)
        self.lst_files.selection_set(0)
        self.lst_files.see(0)

    def load_image_file(self, file_path):
        """加载单张图片的实际逻辑"""
        # 清理旧数据
        self.raw_ch00 = None
        self.vis_ch00 = None
        self.tk_img = None
        gc.collect()

        self.ch00_path = file_path
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # 猜测 ch01 路径
        if "ch00" in filename:
            ch01_path = os.path.join(directory, filename.replace("ch00", "ch01"))
        else:
            ch01_path = os.path.join(directory, filename.replace("ch00", "ch01")) # 简单替换尝试

        if not os.path.exists(ch01_path):
            messagebox.showerror("Error", f"Could not find corresponding ch01 file:\n{ch01_path}")
            return
        
        self.ch01_path = ch01_path

        # 读取 ch00 (这里是耗时操作，UI可能会稍微卡顿一下)
        self.raw_ch00 = cv2.imread(self.ch00_path, cv2.IMREAD_UNCHANGED)
        if self.raw_ch00 is None:
            messagebox.showerror("Error", "Failed to load ch00 image.")
            return

        self.raw_h, self.raw_w = self.raw_ch00.shape[:2]
        self.points = [] # 重置点位
        
        # 归一化用于显示
        self.vis_ch00 = self.normalize_image(self.raw_ch00)
        
        # [Modified] 仅在第一次加载时计算自适应缩放，后续切换保持当前缩放
        if self.is_first_load:
            self.canvas.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w > 10 and canvas_h > 10:
                scale_w = canvas_w / self.raw_w
                scale_h = canvas_h / self.raw_h
                initial_zoom = min(scale_w, scale_h) * 0.95
                self.scale_zoom.set(round(initial_zoom, 3))
            else:
                # Fallback if canvas not ready
                screen_h = self.root.winfo_screenheight()
                target_h = screen_h * 0.6
                initial_zoom = target_h / self.raw_h
                self.scale_zoom.set(round(initial_zoom, 2))
            
            self.is_first_load = False
        
        self.refresh_canvas()
        self.update_status_label()

    def normalize_image(self, img):
        """归一化任意位深图像到 8-bit RGB"""
        # 降采样计算范围，加速处理
        subsample = img[::10, ::10]
        vmin, vmax = np.percentile(subsample, (1, 99))
        
        if vmax > vmin:
            alpha = 255.0 / (vmax - vmin)
            beta = -vmin * alpha
        else:
            alpha = 0
            beta = 0
        
        img_8u = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        if img_8u.ndim == 2:
            img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_BGR2RGB)
        return img_rgb

    def on_zoom_change(self, val):
        self.zoom_level = float(val)
        self.refresh_canvas()
        
    def on_radius_change(self, val):
        self.measure_radius = int(val)
        self.redraw_annotations()

    def refresh_canvas(self):
        if self.vis_ch00 is None:
            return
        
        new_w = int(self.raw_w * self.zoom_level)
        new_h = int(self.raw_h * self.zoom_level)
        
        resized_ch00 = cv2.resize(self.vis_ch00, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        pil_img = Image.fromarray(resized_ch00)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.redraw_annotations()

    def update_status_label(self):
        """更新状态栏，显示下一个点的颜色"""
        count = len(self.points)
        step = count % 4
        particle_idx = count // 4 + 1
        
        text = f"Next: {self.quadrant_names[step]} | Particle #{particle_idx} | Total Points: {count}"
        
        # 设置背景色为对应象限颜色，文字颜色根据背景深浅调整
        bg_color = self.colors_tk[step]
        # 简单判断文字颜色：蓝/紫背景用白字，绿/黄背景用黑字
        fg_color = "white" if step in [0, 2] else "black"
        
        self.lbl_status.config(text=text, bg=bg_color, fg=fg_color)

    def redraw_annotations(self):
        self.canvas.delete("overlay")
        
        scale = self.zoom_level
        r = self.measure_radius * scale
        
        for i, (px, py) in enumerate(self.points):
            cx = px * scale
            cy = py * scale
            
            step = i % 4
            color = self.colors_tk[step]
            
            # 画圆
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=color, width=2, tags="overlay")
            # 只有当一组点满4个时，在几何中心显示 ID（这是保存时的逻辑，这里为了简洁，只显示点的序号或小点）
            # 或者我们在画布上也实时显示点序：
            self.canvas.create_text(cx, cy, text=str(step+1), fill=color, font=("Arial", 10, "bold"), tags="overlay")

    def on_mouse_move(self, event):
        if self.vis_ch00 is None: return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        self.canvas.delete("cursor")
        r = self.measure_radius * self.zoom_level
        
        # 获取当前应该用的颜色
        step = len(self.points) % 4
        color = self.colors_tk[step]
        
        self.canvas.create_oval(canvas_x-r, canvas_y-r, canvas_x+r, canvas_y+r, outline=color, width=1, dash=(4, 4), tags="cursor")

    def on_left_click(self, event):
        if self.vis_ch00 is None: return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        raw_x = int(canvas_x / self.zoom_level)
        raw_y = int(canvas_y / self.zoom_level)
        
        raw_x = max(0, min(raw_x, self.raw_w - 1))
        raw_y = max(0, min(raw_y, self.raw_h - 1))
        
        self.points.append((raw_x, raw_y))
        self.redraw_annotations()
        self.update_status_label()

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.redraw_annotations()
            self.update_status_label()

    def save_data(self, silent=False):
        """
        保存数据。
        :param silent: 如果为 True，则保存成功后不弹窗提示（用于自动保存）。但如果有警告（数据不完整），仍然会弹窗。
        """
        if not self.points:
            if not silent:
                messagebox.showwarning("Warning", "No points selected.")
            return False
        
        # 1. 检查分组完整性
        num_points = len(self.points)
        remainder = num_points % 4
        valid_count = num_points - remainder
        
        if valid_count == 0:
            if not silent: # 即使是自动保存，如果是全废数据，最好也提示一下或者直接忽略？这里选择提示
                messagebox.showwarning("Warning", "Not enough points to form a complete particle (need 4).")
            return False
            
        if remainder > 0:
            # [Forced Prompt] 即使是自动保存模式，因为丢弃了数据，必须告知用户
            messagebox.showinfo("Incomplete Particle", f"Ignored the last {remainder} incomplete marking points.")
            
        # 仅处理完整的组
        final_points = self.points[:valid_count]
        
        print("Calculating and saving...")
        
        # 加载 ch01 进行测量
        if not os.path.exists(self.ch01_path):
            if not silent: messagebox.showerror("Error", "ch01 file missing.")
            return False
        raw_ch01 = cv2.imread(self.ch01_path, cv2.IMREAD_UNCHANGED)
        if raw_ch01 is None:
            if not silent: messagebox.showerror("Error", "Failed to load ch01.")
            return False

        # 准备结果容器
        particle_results = [] # Stores [id, Q1_m, Q2_m, Q3_m, Q4_m, Q1_a, Q2_a, Q3_a, Q4_a]
        
        # 准备可视化图 (BGR)
        vis_mc = cv2.cvtColor(self.vis_ch00, cv2.COLOR_RGB2BGR)
        
        # 按4个一组遍历
        num_particles = valid_count // 4
        
        for i in range(num_particles):
            p_id = i + 1
            group = final_points[i*4 : (i+1)*4] # [(x,y), (x,y), (x,y), (x,y)]
            
            means = []
            areas = []
            
            # 计算几何中心用于画ID
            sum_x, sum_y = 0, 0
            
            for step, (cx, cy) in enumerate(group):
                sum_x += cx
                sum_y += cy
                
                # 测量
                mask = np.zeros(raw_ch01.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (cx, cy), self.measure_radius, 255, -1)
                
                # 计算 Mean 和 Area (pixels count)
                # cv2.mean 计算的是 mask 非零区域的平均值
                mean_val = cv2.mean(raw_ch01, mask=mask)[0]
                area_val = cv2.countNonZero(mask)
                
                means.append(mean_val)
                areas.append(area_val)
                
                # 可视化画圆
                color = self.colors_cv[step] # BGR
                cv2.circle(vis_mc, (cx, cy), self.measure_radius, color, 4)
            
            # 画 ID (几何中心)
            center_x = int(sum_x / 4)
            center_y = int(sum_y / 4)
            cv2.putText(vis_mc, str(p_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            
            # 收集数据
            # Row: [id, Q1_m, Q2_m, Q3_m, Q4_m, Q1_a, Q2_a, Q3_a, Q4_a]
            row_data = [p_id] + means + areas
            particle_results.append(row_data)

        # 释放内存
        del raw_ch01
        gc.collect()

        # 计算统计数据 (Footer)
        # Columns indices in particle_results: 1,2,3,4 (Means) and 5,6,7,8 (Areas)
        stats_means = ["Mean"]
        stats_sd = ["SD"]
        stats_se = ["Std. Error"]
        
        num_cols = 8 # 4 Means + 4 Areas
        for col_idx in range(1, num_cols + 1):
            vals = [r[col_idx] for r in particle_results]
            if vals:
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if len(vals) > 1 else 0
                se = s / np.sqrt(len(vals)) if len(vals) > 0 else 0
                stats_means.append(m)
                stats_sd.append(s)
                stats_se.append(se)
            else:
                stats_means.append(0)
                stats_sd.append(0)
                stats_se.append(0)

        # 3. 保存文件
        directory = os.path.dirname(self.ch00_path)
        base_name = os.path.splitext(os.path.basename(self.ch00_path))[0]
        
        # Excel
        excel_path = os.path.join(directory, f"{base_name}_results.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "Manual Data"
        
        headers = ["particle_id", "Q1_Mean", "Q2_Mean", "Q3_Mean", "Q4_Mean", "Q1_Area", "Q2_Area", "Q3_Area", "Q4_Area"]
        ws.append(headers)
        
        # Define colors (ARGB Hex) for text readability
        # Q1: Blue, Q2: Green, Q3: Purple, Q4: Dark Yellow
        col_map = {
            2: "0000FF", 6: "0000FF", # Q1 (Mean, Area)
            3: "008000", 7: "008000", # Q2
            4: "800080", 8: "800080", # Q3
            5: "CCCC00", 9: "CCCC00"  # Q4
        }
        
        # Format Header
        for cell in ws[1]:
            c_code = col_map.get(cell.column)
            if c_code:
                cell.font = Font(bold=True, color=c_code)
            else:
                cell.font = Font(bold=True)
            
        for row in particle_results:
            ws.append(row)
            
        # Footer
        ws.append(stats_means)
        ws.append(stats_sd)
        ws.append(stats_se)
        
        # Format Data & Footer
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                c_code = col_map.get(cell.column)
                if c_code:
                    cell.font = Font(color=c_code)
        
        wb.save(excel_path)
        
        # Image
        vis_path = os.path.join(directory, f"{base_name}_00visualization.png")
        cv2.imwrite(vis_path, vis_mc)
        
        print(f"Saved {num_particles} particles.")
        print(f"Excel: {excel_path}")
        print(f"Vis: {vis_path}")
        
        # Update listbox color to green immediately
        try:
            if self.ch00_path in self.image_files:
                idx = self.image_files.index(self.ch00_path)
                self.lst_files.itemconfig(idx, {'bg': '#ddffdd'})
        except ValueError:
            pass

        if not silent:
            messagebox.showinfo("Success", f"Saved {num_particles} particles successfully!\nIgnored {remainder} points.")
        else:
            # 自动保存模式下，更新状态栏提示即可
            self.lbl_status.config(text=f"Auto-saved {num_particles} particles.", bg="#ddffdd", fg="black")
            
        return True

if __name__ == "__main__":
    app = ManualMeasurer2()
    # 检查是否有命令行参数传入文件夹路径
    if len(sys.argv) > 1:
        app.load_from_directory(sys.argv[1])
    app.run()