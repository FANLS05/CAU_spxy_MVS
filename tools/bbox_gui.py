# -*- coding: utf-8 -*-
"""
图形化窗口:选图 -> 框选 -> 3D 浏览

左边选一张无畸变图,中间在图上拉一个矩形框,点「投射到 3D」,
弹出 open3d 窗口把框投到 OpenMVS 网格上。

核心计算全部复用 bbox_to_mesh.py 里的 load_scene / load_mesh / project_bbox /
visualize,本文件只负责界面。

用法(在 0916 目录下):
    python tools\\bbox_gui.py --scene work\\food\\dense\\sparse

--mesh 和 --images 不写会自动从 --scene 推出来。
"""
import argparse
import os
import sys
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbox_to_mesh as b2m                                   # noqa: E402

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit('缺 pillow,请先装:pip install pillow')


# =============================================================================
# 路径猜测 —— 让命令行只写一个 --scene 就能跑
# =============================================================================
def default_images_dir(scene_dir):
    """scene 是 <...>/dense/sparse,无畸变图通常在 <...>/dense/images"""
    par = os.path.dirname(os.path.abspath(scene_dir))
    for nm in ('images', os.path.join('dense', 'images')):
        p = os.path.join(par, nm)
        if os.path.isdir(p):
            return p
    return os.path.join(par, 'images')


def default_mesh(scene_dir):
    """scene 是 <...>/<场景>/dense/sparse,网格在 <...>/<场景>/mvs/

    不一定跑过 RefineMesh(稠密网格已经够细时就不必跑),而且「哪份网格
    有贴图」取决于最后一次 TextureMesh 贴的是谁。所以这里挑**有配套贴图
    的那份**,而不是硬性偏好 refine —— 否则会配上几何不一致的贴图,
    3D 里退回灰模。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(scene_dir)))
    mvs = os.path.join(root, 'mvs')
    cands = [os.path.join(mvs, nm) for nm in
             ('scene_dense_mesh_refine.ply', 'scene_dense_mesh.ply')]
    cands = [p for p in cands if os.path.isfile(p)]
    if not cands:
        return None

    # 带贴图的 ply 覆盖了哪些几何(顶点数, 面数)
    textured = set()
    for tp in b2m.find_textured_ply(cands[0]):
        try:
            nv, nf, tf = b2m.ply_header_info(tp)
        except (ValueError, IndexError):
            continue
        if tf:
            textured.add((nv, nf))

    for p in cands:
        try:
            nv, nf, _ = b2m.ply_header_info(p)
        except (ValueError, IndexError):
            continue
        if (nv, nf) in textured:
            return p
    return cands[0]


# =============================================================================
# 主窗口
# =============================================================================
class BBoxGUI:
    MIN_ZOOM = 0.02
    MAX_ZOOM = 8.0
    MIN_DRAG = 6.0              # 小于这么多像素的拖动当成误点

    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg

        self.scene = {}         # {图像名: 位姿/内参}
        self.mesh = None        # (V, F, vcols),第一次投射时才读
        self.texture = (None, None)   # (uv, 贴图图像)
        self.texture_tried = False    # 找过没找到就别反复找
        self.pil_img = None     # 当前图的 PIL 原图
        self.img_name = None
        self.zoom = 1.0
        self.auto_fit = True
        self.rect_bbox = None   # 已确认的框,图像像素坐标 (x0,y0,x1,y1)
        self.drag_start = None
        self._live = None       # 拖动过程中的实时框
        self.rect_id = None
        self._photo = None      # 必须留引用,否则 Tk 会把图回收掉
        self._redraw_job = None
        self._in_config = False
        self.last_result = None

        self._build_widgets()
        self._load_scene()
        self._bind_keys()

    # ------------------------------------------------------------------ 界面
    def _build_widgets(self):
        self.root.title('2D 框 → OpenMVS 3D 网格')
        self.root.geometry('1400x920')
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # ---- 顶栏 ----
        top = ttk.Frame(self.root, padding=(8, 6, 8, 0))
        top.grid(row=0, column=0, sticky='ew')
        btn = ttk.Frame(top)
        btn.pack(fill='x')

        ttk.Button(btn, text='投射到 3D  (Ctrl+Enter)',
                   command=self.project).pack(side='left')
        ttk.Button(btn, text='清空框选  (Esc)',
                   command=self.clear_rect).pack(side='left', padx=(6, 0))
        ttk.Button(btn, text='适应窗口  (F)',
                   command=self.fit).pack(side='left', padx=(6, 0))
        ttk.Button(btn, text='导出框内网格…',
                   command=self.export_crop).pack(side='left', padx=(6, 0))

        self.var_notexture = tk.BooleanVar(value=self.cfg['no_texture'])
        ttk.Checkbutton(btn, text='网格显示成纯灰', variable=self.var_notexture,
                        command=self._on_notexture).pack(side='left', padx=(14, 0))

        ttk.Button(btn, text='打开 3D 窗口',
                   command=self.show_3d).pack(side='right')

        ttk.Label(top, justify='left', foreground='#555',
                  text=f"场景: {self.cfg['scene']}\n"
                       f"图像: {self.cfg['images']}\n"
                       f"网格: {self.cfg['mesh']}").pack(fill='x', pady=(6, 0))

        # ---- 中部:左列表 + 右画布 ----
        mid = ttk.PanedWindow(self.root, orient='horizontal')
        mid.grid(row=1, column=0, sticky='nsew', padx=8, pady=6)

        left = ttk.Frame(mid)
        mid.add(left, weight=0)
        ttk.Label(left, text='图像(点击选择)').pack(anchor='w')
        lf = ttk.Frame(left)
        lf.pack(fill='both', expand=True)
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(lf, width=26, exportselection=False,
                                  font=('Consolas', 9))
        self.listbox.grid(row=0, column=0, sticky='nsew')
        lsb = ttk.Scrollbar(lf, orient='vertical', command=self.listbox.yview)
        lsb.grid(row=0, column=1, sticky='ns')
        self.listbox.config(yscrollcommand=lsb.set)
        self.listbox.bind('<<ListboxSelect>>', self._on_pick_image)

        right = ttk.Frame(mid)
        mid.add(right, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(right, bg='#2b2b2b', highlightthickness=0,
                                cursor='crosshair')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vsb = ttk.Scrollbar(right, orient='vertical', command=self._yview)
        vsb.grid(row=0, column=1, sticky='ns')
        hsb = ttk.Scrollbar(right, orient='horizontal', command=self._xview)
        hsb.grid(row=1, column=0, sticky='ew')
        self.canvas.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Configure>', self._on_configure)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Shift-MouseWheel>', self._on_wheel_h)

        # ---- 日志 ----
        bot = ttk.LabelFrame(self.root, text='日志', padding=4)
        bot.grid(row=2, column=0, sticky='ew', padx=8)
        bot.columnconfigure(0, weight=1)
        self.logtext = tk.Text(bot, height=9, wrap='word', state='disabled',
                               bg='#1e1e1e', fg='#d4d4d4', insertbackground='#d4d4d4',
                               font=('Consolas', 9))
        self.logtext.grid(row=0, column=0, sticky='ew')
        bsb = ttk.Scrollbar(bot, orient='vertical', command=self.logtext.yview)
        bsb.grid(row=0, column=1, sticky='ns')
        self.logtext.config(yscrollcommand=bsb.set)

        # ---- 状态栏 ----
        self.status = tk.StringVar(value='就绪')
        ttk.Label(self.root, textvariable=self.status, relief='sunken',
                  anchor='w', padding=(6, 2)).grid(row=3, column=0, sticky='ew')

    def _bind_keys(self):
        self.root.bind('<Escape>', lambda e: self.clear_rect())
        self.root.bind('<Control-Return>', lambda e: self.project())
        self.root.bind('<F>', lambda e: self.fit())
        self.root.bind('<f>', lambda e: self.fit())

    # ------------------------------------------------------------- 日志/状态
    def log(self, msg=''):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}' if msg else ''
        self.logtext.config(state='normal')
        self.logtext.insert('end', line + '\n')
        self.logtext.see('end')
        self.logtext.config(state='disabled')
        print(line)

    def log_err(self, msg):
        self.log('✗ ' + msg)

    # ------------------------------------------------------------ 读场景位姿
    def _load_scene(self):
        try:
            self.scene = b2m.load_scene(self.cfg['scene'])
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror('读位姿失败',
                                 f'{self.cfg["scene"]}\n\n{e}')
            self.root.destroy()
            return

        names = []
        for nm in sorted(self.scene):
            if os.path.isfile(os.path.join(self.cfg['images'], nm)):
                names.append(nm)
            elif os.path.isfile(os.path.join(self.cfg['images'],
                                             os.path.basename(nm))):
                names.append(nm)
        for nm in names:
            self.listbox.insert('end', nm)

        self.log(f'场景读入:{len(self.scene)} 张定位成功的图')
        if not names:
            self.log_err(f'图像目录里一张都找不到:{self.cfg["images"]}')
            self.log('    (位姿用的名字和磁盘上的文件名对不上?先确认路径)')
            return
        self.log(f'可框选的图:{len(names)} 张 —— 已列在左边')
        self.log('操作:左边点图 → 在图上拖一个框 → Ctrl+Enter 投射到 3D')
        self.listbox.selection_set(0)
        self._on_pick_image()

    # -------------------------------------------------------------- 载入图像
    def _on_pick_image(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        nm = self.listbox.get(sel[0])
        path = os.path.join(self.cfg['images'], nm)
        if not os.path.isfile(path):
            path = os.path.join(self.cfg['images'], os.path.basename(nm))
        if not os.path.isfile(path):
            self.log_err(f'磁盘上找不到 {nm}')
            return

        try:
            im = Image.open(path)
            im = im.convert('RGB')
        except Exception as e:
            self.log_err(f'打开 {path} 失败:{e}')
            return

        self.pil_img = im
        self.img_name = nm
        self.clear_rect(quiet=True)
        self.last_result = None

        W, H = im.size
        sc = self.scene[nm]
        if (W, H) != (sc['W'], sc['H']):
            self.log_err(f'⚠ 磁盘图 {W}x{H} 与位姿里的 {sc["W"]}x{sc["H"]} 不一致!'
                         f'框的坐标会错位 —— 确认这张图就是 COLMAP 用的那张。')

        self.log(f'已载入 {nm}  ({W} x {H}  px)')
        self.auto_fit = True
        self.fit()

    # ------------------------------------------------------------------ 缩放
    def _fit(self, redraw=True):
        if self.pil_img is None:
            return
        W, H = self.pil_img.size
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        z = min(cw / W, ch / H, 1.0)
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z))
        if redraw:
            self.redraw(reset_scroll=True)

    def fit(self):
        self.auto_fit = True
        self._fit()

    def _zoom_at(self, factor, sx, sy):
        """以画布坐标 (sx, sy) 为锚点缩放"""
        if self.pil_img is None:
            return
        z0 = self.zoom
        z1 = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z0 * factor))
        if abs(z1 - z0) < 1e-9:
            return
        # 锚点处的图像坐标
        ix = self.canvas.canvasx(sx) / z0
        iy = self.canvas.canvasy(sy) / z0
        self.zoom = z1
        self.auto_fit = False
        self.redraw()
        W, H = self.pil_img.size
        tw, th = W * z1, H * z1
        if tw > 1:
            self.canvas.xview_moveto(max(0.0, (ix * z1 - sx) / tw))
        if th > 1:
            self.canvas.yview_moveto(max(0.0, (iy * z1 - sy) / th))
        self.redraw()

    def _on_wheel(self, e):
        self._zoom_at(1.25 if e.delta > 0 else 0.8, e.x, e.y)

    def _on_wheel_h(self, e):
        """Shift+滚轮 = 横向滚动"""
        self.canvas.xview_scroll(-1 if e.delta > 0 else 1, 'units')
        self.redraw()

    def _xview(self, *a):
        self.canvas.xview(*a)
        self.redraw()

    def _yview(self, *a):
        self.canvas.yview(*a)
        self.redraw()

    def _on_configure(self, e):
        if self._in_config:
            return
        self._in_config = True
        try:
            if self.auto_fit:
                self._fit()
            else:
                self.redraw()
        finally:
            self._in_config = False

    # ------------------------------------------------------------------ 绘制
    def redraw(self, reset_scroll=False):
        """只渲染当前视口那一块,缩放多少倍内存都不涨。"""
        if self.pil_img is None:
            return
        c = self.canvas
        W, H = self.pil_img.size
        z = self.zoom
        cw = max(c.winfo_width(), 1)
        ch = max(c.winfo_height(), 1)

        c.delete('all')
        self._photo = None
        c.config(scrollregion=(0, 0, W * z, H * z))
        if reset_scroll:
            c.xview_moveto(0.0)
            c.yview_moveto(0.0)

        ox, oy = c.canvasx(0), c.canvasy(0)
        ix0, iy0 = ox / z, oy / z
        ix1, iy1 = ix0 + cw / z, iy0 + ch / z
        cx0, cy0 = max(0.0, ix0), max(0.0, iy0)
        cx1, cy1 = min(float(W), ix1), min(float(H), iy1)

        if cx1 - cx0 > 0.5 and cy1 - cy0 > 0.5:
            tw = max(1, int(round((cx1 - cx0) * z)))
            th = max(1, int(round((cy1 - cy0) * z)))
            resample = Image.LANCZOS if z < 1 else Image.BICUBIC
            reg = self.pil_img.resize(
                (tw, th), resample,
                box=(int(cx0), int(cy0), int(round(cx1)), int(round(cy1))))
            self._photo = ImageTk.PhotoImage(reg)
            c.create_image(cx0 * z, cy0 * z, anchor='nw',
                           image=self._photo, tags='img')

        # 已确认的框
        if self.rect_bbox:
            x0, y0, x1, y1 = self.rect_bbox
            c.create_rectangle(x0 * z, y0 * z, x1 * z, y1 * z,
                               outline='#ff2b2b', width=2, tags='rect')
        # 正在拖的框(重绘会清空画布,得把它画回来,否则滚轮一转框就没了)
        self.rect_id = None
        if self.drag_start is not None and self._live:
            x0, y0, x1, y1 = self._live
            self.rect_id = c.create_rectangle(
                x0 * z, y0 * z, x1 * z, y1 * z,
                outline='#ff2b2b', width=2, dash=(5, 3), tags='live')

    def _schedule_redraw(self):
        if self._redraw_job is not None:
            self.root.after_cancel(self._redraw_job)
        self._redraw_job = self.root.after(20, self._do_redraw)

    def _do_redraw(self):
        self._redraw_job = None
        self.redraw()

    # ------------------------------------------------------------ 鼠标画框
    def _canvas_to_image(self, sx, sy):
        return self.canvas.canvasx(sx) / self.zoom, self.canvas.canvasy(sy) / self.zoom

    def _on_press(self, e):
        if self.pil_img is None:
            return
        x, y = self._canvas_to_image(e.x, e.y)
        self.drag_start = (x, y)
        self._live = None
        self.canvas.delete('live')
        self.rect_id = self.canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline='#ff2b2b', width=2, dash=(5, 3),
            tags='live')

    def _on_drag(self, e):
        if self.drag_start is None or self.rect_id is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self._canvas_to_image(e.x, e.y)
        W, H = self.pil_img.size
        x0c, x1c = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
        y0c, y1c = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
        z = self.zoom
        self.canvas.coords(self.rect_id, x0c * z, y0c * z, x1c * z, y1c * z)
        self._live = (x0c, y0c, x1c, y1c)
        self.status.set(f'框:({x0c:.0f}, {y0c:.0f}) - ({x1c:.0f}, {y1c:.0f})  '
                        f'= {x1c - x0c:.0f} x {y1c - y0c:.0f} px')

    def _on_release(self, e):
        if self.drag_start is None:
            return
        box = getattr(self, '_live', None)
        self.drag_start = None
        if box is None:
            return
        x0, y0, x1, y1 = box
        if (x1 - x0) < self.MIN_DRAG or (y1 - y0) < self.MIN_DRAG:
            self.log('框太小了,忽略。拖大一点再松手。')
            self.clear_rect(quiet=True)
            return
        self.rect_bbox = box
        self._live = None
        self.rect_id = None
        self.canvas.delete('live')
        self.redraw()
        self.log(f'框选完成:({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f})   '
                 f'{x1 - x0:.0f} x {y1 - y0:.0f} px   ——  Ctrl+Enter 投射')
        self.status.set('框选完成,按 Ctrl+Enter 投射到 3D')

    def clear_rect(self, quiet=False):
        self.rect_bbox = None
        self._live = None
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.canvas.delete('rect')
        self.canvas.delete('live')
        self.status.set('已清空框选')
        if not quiet:
            self.log('已清空框选')

    def _on_notexture(self):
        self.log(f'贴图:{"关" if self.var_notexture.get() else "开"}'
                 f' —— 下次开 3D 窗口生效')

    # ------------------------------------------------------------ 载入网格
    def _ensure_mesh(self):
        if self.mesh is not None:
            return True
        self.status.set('正在读网格…')
        self.root.update()
        t0 = time.time()
        try:
            V, F, vcols = b2m.load_mesh(self.cfg['mesh'])
        except Exception as e:
            self.log_err(f'读网格失败:{e}')
            messagebox.showerror('读网格失败', str(e))
            self.status.set('读网格失败')
            return False
        self.mesh = (V, F, vcols)
        self.log(f'网格读入:{len(V)} 顶点 / {len(F)} 三角形 / '
                 f'顶点色 {"有" if vcols is not None else "无"}   '
                 f'({time.time() - t0:.1f}s)')

        if not self.var_notexture.get():
            self._ensure_texture()

        self.status.set('就绪')
        return True

    def _ensure_texture(self):
        """按需加载贴图。scene_textured0.png 有 22MB、8192x8192,
        读一次要两秒,勾了「纯灰」就别白读。"""
        if self.texture[0] is not None or self.texture_tried:
            return
        self.texture_tried = True
        t0 = time.time()
        try:
            uv, tex, src = b2m.load_texture(self.cfg['mesh'],
                                            self.mesh[0], self.mesh[1])
        except Exception as e:
            uv, tex, src = None, None, None
            self.log_err(f'找贴图出错(不影响用):{e}')
        if uv is not None:
            self.texture = (uv, tex)
            self.log(f'贴图绑定好:{os.path.basename(src)}  ({time.time()-t0:.1f}s)')
        else:
            self.log('没找到同源的带贴图 ply —— 3D 里会用顶点色/灰色。')
            self.log('  (要贴图的话,先跑 TextureMesh 生成 scene_textured.ply)')

    # ---------------------------------------------------------------- 投射
    def project(self):
        if self.pil_img is None:
            messagebox.showinfo('还没选图', '先在左边选一张图。')
            return
        if self.rect_bbox is None:
            messagebox.showinfo('还没框选', '在图上按住左键拖一个矩形框。')
            return
        if not self._ensure_mesh():
            return

        sc = self.scene[self.img_name]
        bbox = self.rect_bbox
        self.status.set('计算中…')
        self.root.update()

        t0 = time.time()
        try:
            res = b2m.project_bbox(self.mesh[0], self.mesh[1], sc['K'], sc['R'],
                                   sc['t'], sc['C'], bbox)
        except Exception as e:
            traceback.print_exc()
            self.log_err(f'投射失败:{e}')
            messagebox.showerror('投射失败', str(e))
            self.status.set('投射失败')
            return
        dt = time.time() - t0

        self.last_result = res
        x0, y0, x1, y1 = res['bbox']
        self.log('─' * 60)
        self.log(f'参考图    : {self.img_name}   {sc["W"]}x{sc["H"]}   '
                 f'{sc["model"]}')
        self.log(f'2D 框     : ({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f})   '
                 f'{x1 - x0:.0f} x {y1 - y0:.0f} px')
        self.log(f'光心(世界): [{sc["C"][0]:.4f}, {sc["C"][1]:.4f}, {sc["C"][2]:.4f}]')
        self.log(f'四角光线  : 命中 {res["n_hit"]}/4')
        for i, h in enumerate(res['hits']):
            self.log(f'    角{i} ' + (f'世界坐标 [{h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}]'
                                      if h is not None else '未命中(框角打到背景/空洞)'))
        self.log(f'投影轮廓  : 一圈 {res["n_outline"]} 条射线,'
                 f'命中 {res["n_outline_hit"]} 条,'
                 f'连成 {len(res["outline_seg"])} 段')
        self.log(f'视锥裁剪  : {res["n_tri"]}/{res["n_faces_total"]} 个三角形 '
                 f'({100.0 * res["n_tri"] / max(1, res["n_faces_total"]):.1f}%)')
        if res['n_tri'] == 0:
            self.log('    ⚠ 这个框里一个三角形都没有 —— 框错地方了,或者选错图了。')
        self.log(f'耗时      : {dt * 1000:.1f} ms')

        self.status.set(f'算完了(命中 {res["n_hit"]}/4,{res["n_tri"]} 个三角形)'
                        f' —— 正在打开 3D 窗口')
        self.root.update()
        self.show_3d()

    def show_3d(self):
        if self.last_result is None:
            messagebox.showinfo('还没有结果', '先框选并投射一次。')
            return
        res = self.last_result
        sc = self.scene[self.img_name]
        if self.var_notexture.get():
            uv, tex = None, None
        else:
            self._ensure_texture()
            uv, tex = self.texture
        self.log(f'打开 3D 窗口…(贴图 {"有" if uv is not None else "无"},'
                 f'关掉它才能回到这个界面)')
        self.root.update()
        try:
            b2m.visualize(self.mesh[0], self.mesh[1], self.mesh[2],
                          res, sc, scale=None, uv=uv, tex_img=tex)
        except Exception as e:
            traceback.print_exc()
            self.log_err(f'3D 窗口出错:{e}')
            messagebox.showerror('3D 窗口出错', str(e))
        self.status.set('就绪')

    # ------------------------------------------------------------ 导出碎片
    def export_crop(self):
        if self.last_result is None or self.last_result['frag_v'] is None:
            messagebox.showinfo('没有可导出的东西', '先框选并投射一次。')
            return
        res = self.last_result
        base = os.path.splitext(os.path.basename(self.img_name))[0]
        path = filedialog.asksaveasfilename(
            title='把框内的网格存成 ply',
            defaultextension='.ply', filetypes=[('PLY 网格', '*.ply')],
            initialfile=f'{base}_crop.ply',
            initialdir=os.path.dirname(os.path.abspath(self.cfg['mesh'])))
        if not path:
            return
        try:
            import trimesh
            trimesh.Trimesh(vertices=res['frag_v'], faces=res['frag_f'],
                            process=False).export(path)
        except Exception as e:
            self.log_err(f'导出失败:{e}')
            return
        self.log(f'框内网格已存:{path}   '
                 f'({len(res["frag_v"])} 顶点 / {len(res["frag_f"])} 三角形)')

    # ------------------------------------------------------------------ 关闭
    def quit(self):
        self.root.destroy()


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='选图 + 框选 + 3D 浏览 的图形界面',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True,
                    help='COLMAP sparse 目录,如 work\\food\\dense\\sparse')
    ap.add_argument('--images', default=None,
                    help='无畸变图像目录(默认从 --scene 推)')
    ap.add_argument('--mesh', default=None,
                    help='网格 .ply(默认从 --scene 推 mvs\\scene_dense_mesh_refine.ply)')
    ap.add_argument('--no-texture', action='store_true',
                    help='3D 里网格不加载顶点色')
    args = ap.parse_args()

    if not os.path.isdir(args.scene):
        raise SystemExit(f'--scene 目录不存在:{args.scene}')

    images = args.images or default_images_dir(args.scene)
    if not os.path.isdir(images):
        raise SystemExit(f'图像目录不存在:{images}\n用 --images 指定。')

    mesh = args.mesh or default_mesh(args.scene)
    if mesh is None:
        raise SystemExit('推不出网格路径,用 --mesh 指定 .ply。')
    if not os.path.isfile(mesh):
        raise SystemExit(f'网格文件不存在:{mesh}')

    cfg = {'scene': args.scene, 'images': images, 'mesh': mesh,
           'no_texture': args.no_texture}

    root = tk.Tk()
    app = BBoxGUI(root, cfg)
    root.mainloop()


if __name__ == '__main__':
    main()
