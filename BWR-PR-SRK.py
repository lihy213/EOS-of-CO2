import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLineEdit, QLabel,
                             QPushButton, QCheckBox, QComboBox, QGroupBox,
                             QMessageBox, QFrame, QRadioButton, QButtonGroup,
                             QColorDialog, QScrollArea, QMenu, QAction, QFileDialog,
                             QDialog, QDialogButtonBox)
from PyQt5.QtGui import QClipboard, QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
import CoolProp.CoolProp as CP
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略计算中可能出现的 Numpy 警告


# ==========================================
# 0. 辅助类：导出配置对话框 (保持不变)
# ==========================================
class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Settings")
        self.setFixedSize(320, 260)
        layout = QVBoxLayout(self)
        grp_bg = QGroupBox("Background Options");
        vbox = QVBoxLayout()
        self.rb_white = QRadioButton("White Background (Publication)");
        self.rb_current = QRadioButton("Keep Current Dark Theme")
        self.rb_transparent = QRadioButton("Transparent Background");
        self.rb_custom = QRadioButton("Custom Color...")
        self.rb_white.setChecked(True)
        vbox.addWidget(self.rb_white);
        vbox.addWidget(self.rb_current);
        vbox.addWidget(self.rb_transparent);
        vbox.addWidget(self.rb_custom)
        grp_bg.setLayout(vbox);
        layout.addWidget(grp_bg)
        self.custom_color = "#FFFFFF";
        self.btn_color = QPushButton("Pick Custom Color");
        self.btn_color.setEnabled(False)
        self.btn_color.clicked.connect(self.pick_color);
        layout.addWidget(self.btn_color)
        self.rb_custom.toggled.connect(lambda: self.btn_color.setEnabled(self.rb_custom.isChecked()))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel);
        buttons.accepted.connect(self.accept);
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def pick_color(self):
        c = QColorDialog.getColor(initial=Qt.white)
        if c.isValid():
            self.custom_color = c.name()
            self.btn_color.setText(f"Color: {self.custom_color}");
            self.btn_color.setStyleSheet(f"background-color: {self.custom_color}; color: black;")

    def get_config(self):
        if self.rb_white.isChecked(): return 'white', '#FFFFFF'
        if self.rb_transparent.isChecked(): return 'transparent', None
        if self.rb_custom.isChecked(): return 'custom', self.custom_color
        return 'current', None


# ==========================================
# 1. 物理计算核心 (新增 BWR 逻辑)
# ==========================================
class ThermoEngine:
    def __init__(self):
        self.MW = 0.04401;
        self.R = 8.31446;
        self.Tc = 304.13;
        self.Pc = 7.377e6;
        self.omega = 0.2239

        # --- BWR-like 方程系数 ---
        self.R_MPA_CM3 = 8.314  # MPa * cm3 / (mol * K) (系数来自图片)
        self.MW_G = 44.01
        self.MW_KG_M3_CONV = self.MW_G * 1000  # 44010
        self.BWR_C4 = 1.33125e12
        self.BWR_C5 = 4.48266e3
        self.BWR_C3_FACTOR = 1.30765e7 * 6.478e4
        self.BWR_TERM2_CONST = 4.32945e1
        self.BWR_TERM2_TCONST = 2.52730e5
        self.BWR_TERM2_T2CONST = 1.43215e10
        self.BWR_TERM3_CONST = 4.18926e3
        self.BWR_TERM3_TCONST = 1.30765e7
        # 物理限制
        self.X_MIN = self.MW_KG_M3_CONV / 1500  # 最小体积 (cm3/mol)它远高于 $\text{CO}_2$ 的已知物理最大密度，因此不会排除任何有效的液相根。

    def P_of_X_and_T(self, X, T):
        """
        [BWR-like Equation] 计算给定摩尔体积 X (cm3/mol) 和温度 T (K) 下的压力 P (MPa)。
        """
        # Term 1: Ideal Gas term
        P1 = self.R_MPA_CM3 * T / X

        # Term 2 Coefficient C1
        C1 = (self.BWR_TERM2_CONST * self.R_MPA_CM3 * T - self.BWR_TERM2_TCONST) - (self.BWR_TERM2_T2CONST / (T ** 2))
        P2 = C1 / (X ** 2)

        # Term 3 Coefficient C2
        C2 = (self.BWR_TERM3_CONST * self.R_MPA_CM3 * T - self.BWR_TERM3_TCONST)
        P3 = C2 / (X ** 3)

        # Term 4: 1/X^6 term
        P4 = self.BWR_C3_FACTOR / (X ** 6)

        # Term 5: Exponential term
        E_term = np.exp(-self.BWR_C5 / (X ** 2))
        P5_factor = self.BWR_C4 / (T ** 2)
        P5_term = P5_factor * (1 / (X ** 3) + self.BWR_C5 / (X ** 5)) * E_term

        return P1 + P2 + P3 + P4 + P5_term

    def dP_dX(self, X, T, dX=1e-6):
        """数值导数 dP/dX"""
        return (self.P_of_X_and_T(X + dX, T) - self.P_of_X_and_T(X - dX, T)) / (2 * dX)

    def solve_bwr_density(self, P_MPa, T_K, initial_guess_X):
        """牛顿法求解 BWR 摩尔体积 X"""
        P = P_MPa;
        T = T_K
        X_k = initial_guess_X
        tolerance = 1e-8  # MPa
        max_iter = 100
        X_min = self.X_MIN

        for k in range(max_iter):
            if X_k <= X_min or T <= 0: return np.nan
            P_calc = self.P_of_X_and_T(X_k, T)
            f_X = P - P_calc
            if abs(f_X) < tolerance: return self.MW_KG_M3_CONV / X_k

            f_prime_X = -self.dP_dX(X_k, T)
            if abs(f_prime_X) < 1e-15: return np.nan

            X_next = X_k - f_X / f_prime_X

            if X_next <= X_min:
                X_next = (X_k + X_min * 1.01) / 2  # 步长减半
                if X_next <= X_min: return np.nan

            X_k = X_next
        return np.nan

    def calculate_bwr_density(self, P_Pa, T_K):
        P_MPa = P_Pa / 1e6

        # 1. 理想气体猜测 (cm3/mol)
        X_ideal_guess = self.R_MPA_CM3 * T_K / P_MPa

        # 2. 液相猜测 (从 1000 kg/m3 密度附近开始)
        X_liquid_guess = (self.MW_KG_M3_CONV / 1000) * 0.9

        # 求解
        rho_gas = self.solve_bwr_density(P_MPa, T_K, X_ideal_guess)
        rho_liquid = self.solve_bwr_density(P_MPa, T_K, X_liquid_guess)

        valid_roots = [r for r in [rho_gas, rho_liquid] if r is not np.nan and r > 0]

        if valid_roots:
            return max(valid_roots)  # 选最高密度（稳定相）
        return np.nan

    def calculate_density(self, P, T, method):
        if method == 'BWR':
            return self.calculate_bwr_density(P, T)

        # --- SRK / PR 逻辑 (保持原样) ---
        if method == 'SRK':
            m = 0.48 + 1.574 * self.omega - 0.176 * self.omega ** 2
            b = 0.08664 * self.R * self.Tc / self.Pc
            ac = 0.42748 * (self.R * self.Tc) ** 2 / self.Pc
            u, w = 1, 0
        else:  # PR
            m = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
            b = 0.07780 * self.R * self.Tc / self.Pc
            ac = 0.45724 * (self.R * self.Tc) ** 2 / self.Pc
            u, w = 2, -1

        sqrt_Tr = np.sqrt(T / self.Tc)
        alpha = (1 + m * (1 - sqrt_Tr)) ** 2
        a_val = ac * alpha
        c3 = P
        c2 = P * b * (u - 1) - self.R * T
        c1 = a_val + P * b ** 2 * (w - u) - self.R * T * b * u
        c0 = - (a_val * b + P * b ** 3 * w + self.R * T * b ** 2 * w)

        roots = np.roots([c3, c2, c1, c0])
        real_roots = roots[np.isreal(roots)].real
        valid_roots = real_roots[real_roots > b]

        if len(valid_roots) == 0: return np.nan
        final_V = min(valid_roots) if (P > 1e5 and T < 500) else max(valid_roots)
        return self.MW / final_V

    def get_coolprop_density(self, P, T):
        try:
            return CP.PropsSI('D', 'T', T, 'P', P, 'CO2')
        except:
            return np.nan


# ==========================================
# 2. 图形界面 (UI 更新)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CO2 Property Tool (Advanced Integration)")
        self.resize(1300, 900)
        self.engine = ThermoEngine()

        # 默认参数
        self.current_bg_color = "#121212";
        self.current_ui_text_color = "#e0e0e0"
        self.current_axis_color = "#ffffff";
        self.current_legend_color = "#00bcd4"
        self.current_panel_bg = "#1e1e1e";
        self.accent_color = "#00bcd4"
        self.current_cmap = "plasma"
        self.current_eos_color = "#00bcd4";
        self.current_cp_color = "#ff4081"

        self.theme_mode = "dark"

        self.themes = {
            "dark": {
                "bg": "#121417",
                "panel": "#1E222A",
                "group": "#242A33",
                "input": "#2E3440",
                "text": "#E6E6E6",
                "accent": "#00BCD4",
                "border": "#3A4048"
            },
            "light": {
                "bg": "#F5F7FA",
                "panel": "#FFFFFF",
                "group": "#F0F2F5",
                "input": "#FFFFFF",
                "text": "#1A1A1A",
                "accent": "#007ACC",
                "border": "#C8CCD0"
            }
        }

        self.apply_dynamic_style()

        # UI 构建
        main_widget = QWidget();
        self.setCentralWidget(main_widget);
        layout = QHBoxLayout(main_widget);
        layout.setContentsMargins(10, 10, 10, 10)

        # 左侧面板
        scroll_area = QScrollArea();
        scroll_area.setFixedWidth(360);
        scroll_area.setWidgetResizable(True);
        scroll_area.setFrameShape(QFrame.NoFrame)
        controls_frame = QFrame(objectName="Panel");
        controls_layout = QVBoxLayout(controls_frame);
        controls_layout.setSpacing(15)
        scroll_area.setWidget(controls_frame)
        controls_layout.addWidget(QLabel("PARAMETER CONTROL", objectName="TitleLabel"))
        self.btn_theme = QPushButton("🌙 Dark")
        self.btn_theme.clicked.connect(self.toggle_theme)
        controls_layout.addWidget(self.btn_theme)
        # 视觉组
        grp_vis = QGroupBox(":: VISUAL SETTINGS ::");
        f_vis = QFormLayout()
        self.combo_cmap = QComboBox();
        self.combo_cmap.addItems(["plasma", "viridis", "inferno", "magma", "coolwarm", "jet"])
        self.combo_cmap.currentTextChanged.connect(self.update_cmap);
        f_vis.addRow("Colormap:", self.combo_cmap)
        self.btn_bg = QPushButton("Set Background");
        self.btn_bg.clicked.connect(self.pick_bg_color);
        f_vis.addRow(self.btn_bg)
        self.btn_txt = QPushButton("Set UI Text");
        self.btn_txt.clicked.connect(self.pick_ui_text_color);
        f_vis.addRow(self.btn_txt)
        self.btn_ax = QPushButton("Set Axis Color");
        self.btn_ax.clicked.connect(self.pick_axis_color);
        f_vis.addRow(self.btn_ax)
        self.btn_lg = QPushButton("Set Legend Color");
        self.btn_lg.clicked.connect(self.pick_legend_color);
        f_vis.addRow(self.btn_lg)
        self.btn_eos_color = QPushButton("Set EOS Line Color");
        self.btn_eos_color.clicked.connect(self.pick_eos_color);
        f_vis.addRow(self.btn_eos_color)
        self.btn_cp_color = QPushButton("Set CoolProp Line Color");
        self.btn_cp_color.clicked.connect(self.pick_cp_color);
        f_vis.addRow(self.btn_cp_color)
        grp_vis.setLayout(f_vis);
        controls_layout.addWidget(grp_vis)

        # 计算组
        grp_calc = QGroupBox(":: CONFIG (3D Only) ::");
        f_calc = QFormLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(["PR Equation", "SRK Equation", "BWR Equation"])  # ADDED BWR
        f_calc.addRow("Model:", self.combo_method);
        self.chk_cp = QCheckBox("Verify CoolProp");
        f_calc.addRow(self.chk_cp)
        # self.chk_csv = QCheckBox("Save CSV");
        # f_calc.addRow(self.chk_csv);
        grp_calc.setLayout(f_calc);
        controls_layout.addWidget(grp_calc)

        # ====================== MULTI EOS EXPORT ======================
        grp_multi = QGroupBox(":: MULTI EOS EXPORT ::")
        v_multi = QVBoxLayout()

        self.chk_multi = QCheckBox("Enable Multi-EOS CSV Export")
        self.chk_eos_pr = QCheckBox("PR")
        self.chk_eos_srk = QCheckBox("SRK")
        self.chk_eos_bwr = QCheckBox("BWR")
        self.chk_multi_cp = QCheckBox("Include CoolProp")
        # ====================== ERROR DISPLAY ======================
        self.chk_err = QCheckBox("Show Relative Error (%)")
        self.chk_err.setEnabled(False)

        def toggle_err():
            self.chk_err.setEnabled(self.chk_cp.isChecked() or self.chk_multi_cp.isChecked())

        self.chk_cp.stateChanged.connect(toggle_err)
        self.chk_multi_cp.stateChanged.connect(toggle_err)

        # controls_layout.addWidget(self.chk_err)

        self.chk_eos_pr.setChecked(True)
        self.chk_eos_srk.setChecked(True)
        self.chk_eos_bwr.setChecked(True)

        v_multi.addWidget(self.chk_multi)
        v_multi.addWidget(self.chk_eos_pr)
        v_multi.addWidget(self.chk_eos_srk)
        v_multi.addWidget(self.chk_eos_bwr)
        v_multi.addWidget(self.chk_multi_cp)
        v_multi.addWidget(self.chk_err)

        grp_multi.setLayout(v_multi)
        controls_layout.addWidget(grp_multi)

        # 模式组
        grp_mode = QGroupBox(":: MODE ::");
        v_mode = QVBoxLayout();
        self.rb_3d = QRadioButton("3D Surface");
        self.rb_3d.setChecked(True)
        self.rb_p = QRadioButton("2D Isobaric");
        self.rb_t = QRadioButton("2D Isothermal");
        bg = QButtonGroup(self);
        bg.addButton(self.rb_3d);
        bg.addButton(self.rb_p);
        bg.addButton(self.rb_t)
        v_mode.addWidget(self.rb_3d);
        v_mode.addWidget(self.rb_p);
        v_mode.addWidget(self.rb_t);
        grp_mode.setLayout(v_mode);
        controls_layout.addWidget(grp_mode)

        self.chk_csv = QCheckBox("Save CSV");
        controls_layout.addWidget(self.chk_csv)

        # 范围组
        grp_rng = QGroupBox(":: BOUNDS ::");
        f_rng = QFormLayout();
        self.inp_p1 = QLineEdit("0.1");
        self.inp_p2 = QLineEdit("300.0")  # Increased range for BWR
        self.inp_t1 = QLineEdit("300.0");
        self.inp_t2 = QLineEdit("800.0");
        self.inp_res = QLineEdit("50")
        f_rng.addRow("P Min/Max:", self.h_layout([self.inp_p1, self.inp_p2]));
        f_rng.addRow("T Min/Max:", self.h_layout([self.inp_t1, self.inp_t2]))
        f_rng.addRow("Points:", self.inp_res);
        grp_rng.setLayout(f_rng);
        controls_layout.addWidget(grp_rng)

        # 3D 轴缩放
        grp_scale = QGroupBox(":: 3D AXIS SCALING (Normalized) ::");
        f_scale = QFormLayout()
        self.inp_sx = QLineEdit("1.0");
        self.inp_sy = QLineEdit("1.0");
        self.inp_sz = QLineEdit("1.0")
        f_scale.addRow("X Scale (Temp):", self.inp_sx);
        f_scale.addRow("Y Scale (Rho):", self.inp_sy);
        f_scale.addRow("Z Scale (Pres):", self.inp_sz)
        self.btn_update_scale = QPushButton("Manual Update Scale");
        self.btn_update_scale.clicked.connect(self.manual_update_scale)
        self.btn_update_scale.setStyleSheet("background-color: #555; border: 1px solid #777;");
        f_scale.addRow(self.btn_update_scale)
        grp_scale.setLayout(f_scale);
        controls_layout.addWidget(grp_scale)

        self.btn_run = QPushButton("INITIATE CALCULATION");
        self.btn_run.setFixedHeight(45);
        self.btn_run.clicked.connect(self.run_calculation)
        controls_layout.addWidget(self.btn_run);
        controls_layout.addStretch();
        layout.addWidget(scroll_area)

        # 右侧绘图区
        self.plot_container = QFrame(objectName="PlotArea");
        p_layout = QVBoxLayout(self.plot_container);
        p_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_container, stretch=1);
        self.fig_2d = Figure(facecolor=self.current_bg_color)
        self.cv_2d = FigureCanvas(self.fig_2d);
        p_layout.addWidget(self.cv_2d);
        self.cv_2d.hide()
        self.pl_3d = QtInteractor(self.plot_container);
        p_layout.addWidget(self.pl_3d.interactor);
        self.pl_3d.hide()
        self.lbl_info = QLabel("SYSTEM READY\nAwaiting Input Parameters...");
        self.lbl_info.setObjectName("InfoLabel");
        self.lbl_info.setAlignment(Qt.AlignCenter)
        p_layout.addWidget(self.lbl_info)

        # 绑定右键
        self.cv_2d.setContextMenuPolicy(Qt.CustomContextMenu);
        self.cv_2d.customContextMenuRequested.connect(lambda pos: self.show_menu(pos, "2D"))
        self.pl_3d.setContextMenuPolicy(Qt.CustomContextMenu);
        self.pl_3d.customContextMenuRequested.connect(lambda pos: self.show_menu(pos, "3D"))

    def h_layout(self, widgets):
        w = QWidget();
        l = QHBoxLayout(w);
        l.setContentsMargins(0, 0, 0, 0)
        for wid in widgets: l.addWidget(wid)
        return w

    # --- 3D 轴缩放功能 ---
    def manual_update_scale(self):
        try:
            sx = float(self.inp_sx.text());
            sy = float(self.inp_sy.text());
            sz = float(self.inp_sz.text())
            self.pl_3d.set_scale(sx, sy, sz)
            self.statusBar().showMessage(f"Scale updated: {sx:.2f}, {sy:.2f}, {sz:.2f}", 2000)
        except Exception as e:
            QMessageBox.warning(self, "Scale Error", "Invalid Scale Values")

    # --- 导出与右键菜单 ---
    def show_menu(self, pos, mode):
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {self.current_panel_bg}; color: {self.current_ui_text_color}; border: 1px solid {self.accent_color}; }} QMenu::item:selected {{ background: {self.accent_color}; color: white; }}")
        act_copy = QAction("Copy to Clipboard", self);
        act_save = QAction("Save as Image", self)
        if mode == "2D":
            act_copy.triggered.connect(self.copy_2d); act_save.triggered.connect(self.save_2d); widget = self.cv_2d
        else:
            act_copy.triggered.connect(self.copy_3d); act_save.triggered.connect(self.save_3d); widget = self.pl_3d
        menu.addAction(act_copy);
        menu.addAction(act_save)
        menu.exec_(widget.mapToGlobal(pos))

    def get_export_config(self):
        dlg = ExportDialog(self);
        if dlg.exec_() == QDialog.Accepted: return dlg.get_config()
        return None

    def copy_3d(self):
        if not self.pl_3d.renderer.actors:
            QMessageBox.warning(self, "Export Error", "No 3D scene to export.")
            return

        cfg = self.get_export_config()
        if not cfg: return
        mode, color = cfg
        trans = (mode == 'transparent');
        orig_bg = self._prepare_3d_export(mode, color)
        try:
            img = self.pl_3d.screenshot(
                transparent_background=trans,
                return_img=True
            )

            if img is None:
                QMessageBox.warning(self, "Copy Error", "Failed to capture 3D image.")
                return

            # --- 强制转换为连续内存 ---
            img = np.ascontiguousarray(img)

            h, w, ch = img.shape
            # --- 确定格式 ---
            if ch == 3:
                fmt = QImage.Format_RGB888
            elif ch == 4:
                fmt = QImage.Format_RGBA8888
            else:
                QMessageBox.warning(self, "Copy Error", "Unsupported image format.")
                return

            # --- 使用 Qt 深拷贝，彻底断开 numpy 生命周期 ---
            qimg = QImage(
                img.tobytes(),  # 关键：不要用 img.data
                w,
                h,
                w * ch,
                fmt
            )

            qimg = qimg.copy()  # 强制 Qt 内部独立内存

            QApplication.clipboard().setImage(qimg)

            self.statusBar().showMessage("Copied", 2000)
        finally:
            self._restore_3d_export(orig_bg)

    def save_3d(self):
        if not self.pl_3d.renderer.actors:
            QMessageBox.warning(self, "Export Error", "No 3D scene to export.")
            return

        cfg = self.get_export_config()
        if not cfg: return
        mode, color = cfg
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D", "plot_3d.png", "PNG (*.png)")
        if path:
            trans = (mode == 'transparent');
            orig_bg = self._prepare_3d_export(mode, color)
            try:
                self.pl_3d.screenshot(path, transparent_background=trans)
                self.statusBar().showMessage(f"Saved: {path}", 2000)
            finally:
                self._restore_3d_export(orig_bg)

    def _prepare_3d_export(self, mode, color):
        orig_bg = self.pl_3d.background_color
        if mode == 'white' or mode == 'custom':
            bg_color = color if mode == 'custom' else 'white'
            self.pl_3d.set_background(bg_color)
        return orig_bg

    def _restore_3d_export(self, orig_bg):
        self.pl_3d.set_background(orig_bg)

    def _get_2d_args(self, mode, color):
        args = {'dpi': 300}
        if mode == 'transparent':
            args['transparent'] = True
        elif mode == 'white':
            args['facecolor'] = 'white'
        elif mode == 'custom':
            args['facecolor'] = color
        else:
            args['facecolor'] = self.fig_2d.get_facecolor()
        return args

    def copy_2d(self):
        cfg = self.get_export_config();
        if not cfg:
            return
        mode, color = cfg
        orig_ax_color = self.current_axis_color;
        temp_changed = False
        if mode == 'white' and self.current_axis_color.lower() in ['#ffffff', 'white', '#fff']:
            self._apply_2d_colors('black', 'black');
            temp_changed = True
        buf = io.BytesIO();
        self.fig_2d.savefig(buf, format='png', **self._get_2d_args(mode, color))
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        if temp_changed: self._apply_2d_colors(self.current_axis_color, self.current_legend_color)

    def save_2d(self):
        cfg = self.get_export_config();
        if not cfg:
            return
        mode, color = cfg
        path, _ = QFileDialog.getSaveFileName(self, "Save 2D", "plot_2d.png", "PNG (*.png)")
        if path:
            temp_changed = False
            if mode == 'white' and self.current_axis_color.lower() in ['#ffffff', 'white', '#fff']:
                self._apply_2d_colors('black', 'black');
                temp_changed = True
            self.fig_2d.savefig(path, **self._get_2d_args(mode, color))
            if temp_changed: self._apply_2d_colors(self.current_axis_color, self.current_legend_color)
            self.statusBar().showMessage(f"Saved: {path}", 2000)

    def _apply_2d_colors(self, axis_c, leg_c):
        if not self.fig_2d.axes: return
        ax = self.fig_2d.axes[0]
        ax.xaxis.label.set_color(axis_c);
        ax.yaxis.label.set_color(axis_c)
        ax.tick_params(colors=axis_c);
        ax.spines['bottom'].set_color(axis_c)
        ax.spines['top'].set_color(axis_c);
        ax.spines['left'].set_color(axis_c);
        ax.spines['right'].set_color(axis_c)
        ax.title.set_color(self.accent_color)
        if ax.get_legend():
            for t in ax.get_legend().get_texts(): t.set_color(leg_c)
            ax.get_legend().get_frame().set_edgecolor(leg_c)
        self.cv_2d.draw()

    # --- 样式与绘图 ---
    def pick_bg_color(self):
        c = QColorDialog.getColor(initial=Qt.black)
        if c.isValid():
            self.current_bg_color = c.name();
            self.current_panel_bg = c.lighter(120).name()
            self.apply_dynamic_style();
            self.pl_3d.set_background(self.current_bg_color)
            self.fig_2d.patch.set_facecolor(self.current_bg_color);
            self.cv_2d.draw()

    def pick_ui_text_color(self):
        c = QColorDialog.getColor(initial=Qt.white)
        if c.isValid(): self.current_ui_text_color = c.name(); self.apply_dynamic_style()

    def pick_axis_color(self):
        c = QColorDialog.getColor(initial=Qt.white)
        if c.isValid(): self.current_axis_color = c.name()

    def pick_legend_color(self):
        c = QColorDialog.getColor(initial=Qt.cyan)
        if c.isValid(): self.current_legend_color = c.name()

    def pick_eos_color(self):
        c = QColorDialog.getColor(initial=QColor(self.current_eos_color), title="EOS Line Color")
        if c.isValid(): self.current_eos_color = c.name(); self.update_2d_lines_instantly()

    def pick_cp_color(self):
        c = QColorDialog.getColor(initial=QColor(self.current_cp_color), title="CoolProp Line Color")
        if c.isValid(): self.current_cp_color = c.name(); self.update_2d_lines_instantly()

    def update_2d_lines_instantly(self):
        if not self.fig_2d.axes: return
        ax = self.fig_2d.axes[0];
        lines = ax.get_lines()
        for line in lines:
            label = line.get_label()
            if 'EOS' in label or 'Equation' in label:
                line.set_color(self.current_eos_color)
            elif 'CoolProp' in label:
                line.set_color(self.current_cp_color)
        self.cv_2d.draw()

    def update_cmap(self, t):
        self.current_cmap = t

    def apply_dynamic_style(self):
        theme = self.themes[self.theme_mode]

        style = f"""
        QMainWindow {{
            background-color: {theme['bg']};
        }}

        QWidget {{
            color: {theme['text']};
            font-family: "Segoe UI", "Inter", "Arial";
            font-size: 9pt;
        }}

        /* 左侧主面板 */
        QFrame#Panel {{
            background-color: {theme['panel']};
            border-right: 2px solid {theme['accent']};
        }}

        /* 分组框 */
        QGroupBox {{
            background-color: {theme['group']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            margin-top: 18px;
            padding-top: 10px;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: {theme['accent']};
            font-weight: bold;
            background-color: {theme['panel']};
        }}

        /* 输入框 */
        QLineEdit, QComboBox {{
            background-color: {theme['input']};
            border: 1px solid {theme['border']};
            border-radius: 6px;
            padding: 4px;
            color: {theme['text']};
        }}

        /* 按钮 */
        QPushButton {{
            background-color: {theme['accent']};
            color: white;
            border-radius: 8px;
            padding: 6px;
            font-weight: bold;
        }}

        QPushButton:hover {{
            background-color: {theme['accent']}CC;
        }}

        QLabel#TitleLabel {{
            color: {theme['accent']};
            font-size: 14pt;
            font-weight: bold;
            padding-bottom: 6px;
            border-bottom: 1px solid {theme['accent']};
        }}
        """

        self.setStyleSheet(style)

    def toggle_theme(self):
        if self.theme_mode == "dark":
            self.theme_mode = "light"
            self.btn_theme.setText("☀ Light")
        else:
            self.theme_mode = "dark"
            self.btn_theme.setText("🌙 Dark")

        self.apply_dynamic_style()

    def run_calculation(self):
        try:
            m = self.combo_method.currentText().split()[0];
            pts = int(self.inp_res.text())
            p1, p2 = float(self.inp_p1.text()), float(self.inp_p2.text())
            t1, t2 = float(self.inp_t1.text()), float(self.inp_t2.text())
            mode = "3D"
            if self.rb_p.isChecked():
                mode = "FixP"; p = np.linspace(p1, p2, pts); t = np.linspace(t1, t2, pts);
                Tg, Pg = np.meshgrid(t, p)
            elif self.rb_t.isChecked():
                mode = "FixT"; p = np.linspace(p1, p2, pts); t = np.linspace(t1, t2, pts);
                Tg, Pg = np.meshgrid(t, p)
            if mode == "3D":
                p = np.linspace(p1, p2, pts); t = np.linspace(t1, t2, pts); Tg, Pg = np.meshgrid(t, p);
                if self.chk_multi.isChecked():
                    self.process_3d_multi_eos(Tg, Pg)
                    return
                else:
                    self.process_3d(Tg, Pg, m)
            elif mode == "FixP":
                if self.chk_multi.isChecked():
                    self.process_2d_multi_eos(np.linspace(t1, t2, pts), p1, "T")
                else:
                    self.process_2d(np.linspace(t1, t2, pts), p1, "T", m)
            elif mode == "FixT":
                if self.chk_multi.isChecked():
                    self.process_2d_multi_eos(np.linspace(p1, p2, pts), t1, "P")
                else:
                    self.process_2d(np.linspace(p1, p2, pts), t1, "P", m)
        except Exception as e:
            QMessageBox.critical(self, "Err", str(e))

    def process_3d(self, T, P, m):
        self.lbl_info.hide()
        self.cv_2d.hide()
        self.pl_3d.interactor.show()

        # ====================== 计算密度 ======================
        P_pa = P * 1e6
        Rho = np.zeros_like(T)

        r, c = T.shape
        data_rows = []
        for i in range(r):
            for j in range(c):
                Rho[i, j] = self.engine.calculate_density(
                    P_pa[i, j], T[i, j], m
                )
                if self.chk_csv.isChecked():
                    data_rows.append({"Pressure (MPa)": P[i, j], "Temperature (K)": T[i, j], "Density (kg/m3)": Rho[i,j]})

        # ====================== 视觉归一化（正方体） ======================
        T_min, T_max = T.min(), T.max()
        R_min, R_max = Rho.min(), Rho.max()
        P_min, P_max = P.min(), P.max()

        Tn = (T - T_min) / (T_max - T_min)
        Rn = (Rho - R_min) / (R_max - R_min)
        Pn = (P - P_min) / (P_max - P_min)

        # ====================== PyVista 绘图 ======================
        self.pl_3d.clear()
        self.pl_3d.set_background(self.current_bg_color)

        grid = pv.StructuredGrid(Tn, Rn, Pn)
        grid["Density"] = Rho.flatten(order="F")

        self.pl_3d.add_mesh(
            grid,
            scalars="Density",
            cmap=self.current_cmap,
            opacity=0.95,
            scalar_bar_args={
                "title": "Density (kg/m3)",
                "color": self.current_legend_color
            }
        )

        # ====================== 正方体坐标轴 ======================
        self.pl_3d.set_scale(1.0, 1.0, 1.0)

        self.pl_3d.show_bounds(
            bounds=(0, 1, 0, 1, 0, 1),
            xlabel="Temperature",
            ylabel="Density",
            zlabel="Pressure",
            color=self.current_axis_color,
            grid="back",
            location="outer",
            all_edges=True
        )

        # ====================== 物理量说明（避免刻度歧义） ======================
        self.pl_3d.add_axes(
            xlabel=f"T: {T_min:.0f} – {T_max:.0f} K",
            ylabel=f"ρ: {R_min:.0f} – {R_max:.0f} kg/m³",
            zlabel=f"P: {P_min:.2f} – {P_max:.2f} MPa",
            color=self.current_axis_color
        )

        # ====================== 相机 ======================
        self.pl_3d.camera_position = "iso"
        self.pl_3d.reset_camera()
        self.pl_3d.camera.zoom(0.85)

        self.pl_3d.add_text(
            f"CO₂ Density Surface ({m})",
            position="upper_left",
            font_size=12,
            color=self.accent_color
        )

        if self.chk_csv.isChecked():
            pd.DataFrame(data_rows).to_csv(f"CO2_Data_3D_{m}.csv", index=False)
            self.statusBar().showMessage(f"3D {m} CSV Saved", 2000)

    def process_3d_multi_eos(self, T, P):
        P_pa = P * 1e6
        r, c = T.shape

        eos_list = []
        if self.chk_eos_pr.isChecked():
            eos_list.append("PR")
        if self.chk_eos_srk.isChecked():
            eos_list.append("SRK")
        if self.chk_eos_bwr.isChecked():
            eos_list.append("BWR")

        if not eos_list:
            QMessageBox.warning(self, "EOS Error", "No EOS selected for export.")
            return

        include_cp = self.chk_multi_cp.isChecked()
        show_err = include_cp and self.chk_err.isChecked()

        # ====================== 数据容器 ======================
        data = []

        for i in range(r):
            for j in range(c):
                row = {
                    "Pressure (MPa)": P[i, j],
                    "Temperature (K)": T[i, j]
                }
                if include_cp:
                    rho_cp = self.engine.get_coolprop_density(P_pa[i, j], T[i, j])
                    row["Density_CoolProp (kg/m3)"] = rho_cp

                for eos in eos_list:
                    rho = self.engine.calculate_density(P_pa[i, j], T[i, j], eos)
                    row[f"Density_{eos} (kg/m3)"] = rho
                    if show_err:
                        err = abs(rho - rho_cp) / rho_cp * 100
                        row[f"RelError_{eos} (%)"] = err

                data.append(row)

        if self.chk_csv.isChecked():
            eos_tag = "_".join(eos_list)
            cp_tag = "_vs_CoolProp" if include_cp else ""
            filename = f"CO2_3D_MultiEOS_{eos_tag}_{cp_tag}.csv"
            pd.DataFrame(data).to_csv(filename, index=False)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Multi-EOS CSV exported:\n{filename}"
            )
        else:
            # 仅计算/绘图，不导出
            self.statusBar().showMessage(
                "Multi-EOS calculation completed (CSV export disabled)",
                3000
            )

    def process_2d(self, x, fix, mode, m):
        self.lbl_info.hide()
        self.pl_3d.interactor.hide()
        self.cv_2d.show()

        y, y_cp, y_err = [], [], []
        data = []

        for val in x:
            if mode == "T":
                T = val
                P_pa = fix * 1e6
                x_label = "Temperature (K)"
            else:
                T = fix
                P_pa = val * 1e6
                x_label = "Pressure (MPa)"

            rho = self.engine.calculate_density(P_pa, T, m)
            rho_cp = self.engine.get_coolprop_density(P_pa, T) if self.chk_cp.isChecked() else np.nan

            y.append(rho)
            if self.chk_cp.isChecked():
                y_cp.append(rho_cp)

            if self.chk_cp.isChecked() and self.chk_err.isChecked():
                err = abs(rho - rho_cp) / rho_cp * 100
                y_err.append(err)

            if self.chk_csv.isChecked():
                row = {
                    x_label: val,
                    f"Density_{m} (kg/m3)": rho
                }
                if self.chk_cp.isChecked():
                    row["Density_CoolProp (kg/m3)"] = rho_cp
                if self.chk_cp.isChecked() and self.chk_err.isChecked():
                    row[f"RelError_{m} (%)"] = err
                data.append(row)

        # ====================== 绘图 ======================
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        ax2 = ax.twinx() if self.chk_err.isChecked() and self.chk_cp.isChecked() else None

        ax.plot(x, y, linewidth=2.5, label=f"{m} Density")

        if self.chk_cp.isChecked():
            ax.plot(x, y_cp, "--", linewidth=2, label="CoolProp")

        if ax2:
            ax2.plot(x, y_err, ":", linewidth=2, label=f"{m} Error (%)")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Density (kg/m3)")
        if ax2:
            ax2.set_ylabel("Relative Error (%)")

        lines, labels = ax.get_legend_handles_labels()
        if ax2:
            l2, lb2 = ax2.get_legend_handles_labels()
            lines += l2
            labels += lb2

        ax.legend(lines, labels)
        ax.grid(True)
        self.cv_2d.draw()

        if self.chk_csv.isChecked():
            pd.DataFrame(data).to_csv(f"CO2_2D_{m}.csv", index=False)
            self.statusBar().showMessage(f"2D {m} CSV Saved", 2000)


    def process_2d_multi_eos(self, x, fix, mode):
        self.lbl_info.hide()
        self.pl_3d.interactor.hide()
        self.cv_2d.show()

        # ====================== EOS 选择 ======================
        eos_list = []
        if self.chk_eos_pr.isChecked():
            eos_list.append("PR")
        if self.chk_eos_srk.isChecked():
            eos_list.append("SRK")
        if self.chk_eos_bwr.isChecked():
            eos_list.append("BWR")

        if not eos_list:
            QMessageBox.warning(self, "EOS Error", "No EOS selected for Multi-EOS export.")
            return

        include_cp = self.chk_multi_cp.isChecked()
        show_err = include_cp and self.chk_err.isChecked()

        # ====================== 横轴与固定参数 ======================
        if mode == "T":  # Fix P
            x_label = "Temperature (K)"
            fix_tag = f"FixP_{fix:.1f}MPa"
        else:  # Fix T
            x_label = "Pressure (MPa)"
            fix_tag = f"FixT_{fix:.0f}K"

        # ====================== 数据容器 ======================
        data = {x_label: []}

        for eos in eos_list:
            data[f"Density_{eos} (kg/m3)"] = []
            if show_err:
                data[f"RelError_{eos} (%)"] = []

        if include_cp:
            data["Density_CoolProp (kg/m3)"] = []

        # ====================== 计算循环 ======================
        for val in x:
            if mode == "T":
                T = val
                P_pa = fix * 1e6
            else:
                T = fix
                P_pa = val * 1e6

            data[x_label].append(val)

            rho_cp = None
            if include_cp:
                rho_cp = self.engine.get_coolprop_density(P_pa, T)
                data["Density_CoolProp (kg/m3)"].append(rho_cp)

            for eos in eos_list:
                rho = self.engine.calculate_density(P_pa, T, eos)
                data[f"Density_{eos} (kg/m3)"].append(rho)

                if show_err:
                    err = abs(rho - rho_cp) / rho_cp * 100
                    data[f"RelError_{eos} (%)"].append(err)

        # ====================== 绘图 ======================
        self.fig_2d.clear()
        self.fig_2d.patch.set_facecolor(self.current_bg_color)

        ax = self.fig_2d.add_subplot(111)
        ax.set_facecolor("none")

        ax2 = ax.twinx() if show_err else None

        # ---------- 左轴：密度 ----------
        density_styles = ["-", "--", "-."]
        for eos, ls in zip(eos_list, density_styles):
            ax.plot(
                x,
                data[f"Density_{eos} (kg/m3)"],
                linestyle=ls,
                linewidth=2.3,
                label=f"{eos} Density"
            )

        if include_cp:
            ax.plot(
                x,
                data["Density_CoolProp (kg/m3)"],
                linestyle=":",
                linewidth=2.5,
                label="CoolProp"
            )

        # ---------- 右轴：误差 ----------
        if show_err:
            error_styles = [":", "--", "-."]
            for eos, ls in zip(eos_list, error_styles):
                ax2.plot(
                    x,
                    data[f"RelError_{eos} (%)"],
                    linestyle=ls,
                    linewidth=2,
                    label=f"{eos} Error (%)"
                )

        # ====================== 轴与刻度（修复重点） ======================
        ac = self.current_axis_color

        ax.set_xlabel(x_label, color=ac)
        ax.set_ylabel("Density (kg/m3)", color=ac)
        ax.tick_params(axis="both", colors=ac, direction="in", length=6)

        for spine in ax.spines.values():
            spine.set_color(ac)

        if ax2:
            ax2.set_ylabel("Relative Error (%)", color=ac)
            ax2.tick_params(axis="y", colors=ac, direction="in", length=6)
            ax2.spines["right"].set_color(ac)

        # ====================== 图例 ======================
        l1, lb1 = ax.get_legend_handles_labels()
        if ax2:
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2)
        else:
            ax.legend()

        ax.grid(True, linestyle="--", alpha=0.3)
        self.cv_2d.draw()

        # ====================== CSV 保存 ======================
        if self.chk_csv.isChecked():
            eos_tag = "_".join(eos_list)
            cp_tag = "_vs_CoolProp" if include_cp else ""
            filename = f"CO2_2D_MultiEOS_{eos_tag}_{fix_tag}{cp_tag}.csv"
            pd.DataFrame(data).to_csv(filename, index=False)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Multi-EOS CSV exported:\n{filename}"
            )
        else:
            self.statusBar().showMessage(
                "Multi-EOS calculation completed (CSV export disabled)",
                3000
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ===== 新增：全局字体 =====
    from PyQt5.QtGui import QFont

    app.setFont(QFont("Segoe UI", 9))

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
