import sys
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QSlider, QStyle,
                               QFileDialog, QFrame, QSizePolicy, QToolButton)
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QFont,
                           QPainterPath, QLinearGradient, QPixmap)

STYLE_SHEET = """
QWidget {
    background-color: #0f0f0f;
    color: #f1f1f1;
    font-family: 'Segoe UI', sans-serif;
}
QFrame#panel {
    background-color: #181818;
    border-radius: 12px;
}
QPushButton#loadBtn {
    background-color: #ff3b3b;
    color: white;
    border-radius: 6px;
    font-weight: bold;
    padding: 8px 16px;
    font-size: 13px;
}
QPushButton#loadBtn:hover {
    background-color: #ff5c5c;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #3f3f3f;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #ff3b3b;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f1f1f1;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QToolButton {
    background-color: #181818;
    border-radius: 18px;
    padding: 8px;
    icon-size: 20px;
}
QToolButton:hover {
    background-color: #2a2a2a;
}
QToolButton:disabled {
    background-color: #181818;
    opacity: 0.5;
}
QToolButton#playBtn {
    background-color: #ff3b3b;
    border-radius: 26px;
    padding: 12px;
    icon-size: 28px;
}
QToolButton#playBtn:hover {
    background-color: #ff5c5c;
}
QLabel {
    background-color: transparent;
}
"""

class Sparkline(QWidget):
    def __init__(self):
        super().__init__()
        self.costs = []
        self.current_frame = 0
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, costs, current):
        self.costs = costs
        self.current_frame = current
        self.update()

    def paintEvent(self, event):
        if not self.costs:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        margin_x, margin_y = 10, 10
        
        cmin, cmax = min(self.costs), max(self.costs)
        span = (cmax - cmin) if (cmax - cmin) > 0 else 1.0
        n = len(self.costs)

        def get_pt(i, c):
            px = margin_x if n <= 1 else margin_x + (i / (n - 1)) * (w - 2 * margin_x)
            py = h - margin_y - ((c - cmin) / span) * (h - 2 * margin_y)
            return px, py

        points = [get_pt(i, c) for i, c in enumerate(self.costs)]

        if len(points) > 1:
            path_fill = QPainterPath()
            path_fill.moveTo(points[0][0], h - margin_y)
            for px, py in points:
                path_fill.lineTo(px, py)
            path_fill.lineTo(points[-1][0], h - margin_y)
            
            pixmap = QPixmap(w, h)
            pixmap.fill(Qt.transparent)
            pix_painter = QPainter(pixmap)
            pix_painter.setRenderHint(QPainter.Antialiasing)
            
            gradient_h = QLinearGradient(margin_x, 0, w - margin_x, 0)
            gradient_h.setColorAt(0, QColor(255, 59, 59, 220))
            gradient_h.setColorAt(1, QColor(76, 175, 80, 220))
            
            pix_painter.setPen(Qt.NoPen)
            pix_painter.setBrush(gradient_h)
            pix_painter.drawPath(path_fill)
            
            pix_painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            gradient_v = QLinearGradient(0, margin_y, 0, h - margin_y)
            gradient_v.setColorAt(0, QColor(0, 0, 0, 255))
            gradient_v.setColorAt(1, QColor(0, 0, 0, 0))
            
            pix_painter.setBrush(gradient_v)
            pix_painter.drawRect(0, 0, w, h)
            pix_painter.end()
            
            painter.drawPixmap(0, 0, pixmap)

            path_line = QPainterPath()
            path_line.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                path_line.lineTo(px, py)
            
            line_gradient = QLinearGradient(margin_x, 0, w - margin_x, 0)
            line_gradient.setColorAt(0, QColor("#ff3b3b"))
            line_gradient.setColorAt(1, QColor("#4caf50"))
            
            pen = QPen(line_gradient, 2.5)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path_line)

        if 0 <= self.current_frame < n:
            cx, cy = points[self.current_frame]
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawLine(int(cx), margin_y, int(cx), h - margin_y)
            painter.setPen(Qt.NoPen)
            
            ratio = max(0.0, min(1.0, (cx - margin_x) / (w - 2 * margin_x)))
            r = int(255 - ratio * (255 - 76))
            g = int(59 + ratio * (175 - 59))
            b = int(59 + ratio * (80 - 59))
            
            painter.setBrush(QColor(r, g, b))
            painter.drawEllipse(QRectF(cx - 5, cy - 5, 10, 10))

class NodeGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.state = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_state(self, state):
        self.state = state
        self.update()

    def paintEvent(self, event):
        if not self.state:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cols = 5
        rows = 4
        pad = 16

        cell_w = (w - pad * (cols + 1)) / cols
        cell_h = (h - pad * (rows + 1)) / rows
        size = min(cell_w, cell_h)

        start_x = (w - (cols * size + (cols - 1) * pad)) / 2
        start_y = (h - (rows * size + (rows - 1) * pad)) / 2

        font_label = QFont("Segoe UI", max(8, int(size * 0.12)))
        font_value = QFont("Segoe UI", max(10, int(size * 0.22)), QFont.Bold)

        for i in range(min(len(self.state), 20)):
            val = self.state[i]
            c = i % cols
            r = i // cols

            x = start_x + c * (size + pad)
            y = start_y + r * (size + pad)

            t = max(0, min(90, val)) / 90.0
            red = (255, 59, 59)
            yellow = (255, 215, 80)
            green = (74, 222, 128)

            if t <= 0.5:
                u = t / 0.5
                r_c = int(red[0] + (yellow[0] - red[0]) * u)
                g_c = int(red[1] + (yellow[1] - red[1]) * u)
                b_c = int(red[2] + (yellow[2] - red[2]) * u)
            else:
                u = (t - 0.5) / 0.5
                r_c = int(yellow[0] + (green[0] - yellow[0]) * u)
                g_c = int(yellow[1] + (green[1] - yellow[1]) * u)
                b_c = int(yellow[2] + (green[2] - yellow[2]) * u)

            rect = QRectF(x, y, size, size)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(r_c, g_c, b_c))
            painter.drawRoundedRect(rect, 12, 12)

            painter.setPen(QColor("#111111"))
            painter.setFont(font_label)
            painter.drawText(rect.adjusted(0, -size*0.4, 0, 0), Qt.AlignCenter, f"Node {i}")
            
            painter.setFont(font_value)
            painter.drawText(rect.adjusted(0, size*0.2, 0, 0), Qt.AlignCenter, f"{val}s")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.history = []
        self.costs = []
        self.current_frame = 0
        self.playing = False
        self.speed_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        self.base_delay_ms = 100

        self.setWindowTitle("Local Search Replay")
        self.resize(1100, 800)
        self.setStyleSheet(STYLE_SHEET)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_step)

        self.init_ui()
        self.update_controls_state()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        top_bar = QHBoxLayout()
        lbl_title = QLabel("🔎 Local Search Replay")
        lbl_title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        btn_load = QPushButton("📂 Load Log File")
        btn_load.setObjectName("loadBtn")
        btn_load.setFixedSize(160, 40)
        btn_load.clicked.connect(self.load_log)
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        top_bar.addWidget(btn_load)
        main_layout.addLayout(top_bar)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        self.canvas_panel = QFrame()
        self.canvas_panel.setObjectName("panel")
        canvas_layout = QVBoxLayout(self.canvas_panel)
        self.node_grid = NodeGrid()
        canvas_layout.addWidget(self.node_grid)
        body_layout.addWidget(self.canvas_panel, stretch=3)

        self.side_panel = QFrame()
        self.side_panel.setObjectName("panel")
        self.side_panel.setFixedWidth(320)
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(20, 24, 20, 24)
        side_layout.setSpacing(8)

        lbl_status = QLabel("Status")
        lbl_status.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 13px;")
        self.lbl_iter = QLabel("—")
        self.lbl_iter.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.lbl_cost = QLabel("Cost: —")
        self.lbl_cost.setStyleSheet("color: #ff3b3b; font-size: 16px;")
        self.lbl_delta = QLabel("")
        self.lbl_delta.setStyleSheet("color: #aaaaaa; font-size: 13px;")

        side_layout.addWidget(lbl_status)
        side_layout.addWidget(self.lbl_iter)
        side_layout.addWidget(self.lbl_cost)
        side_layout.addWidget(self.lbl_delta)
        side_layout.addSpacing(16)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background-color: #3f3f3f;")
        side_layout.addWidget(div)
        side_layout.addSpacing(16)

        lbl_trend = QLabel("Tren Cost")
        lbl_trend.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 13px;")
        side_layout.addWidget(lbl_trend)

        self.sparkline = Sparkline()
        side_layout.addWidget(self.sparkline)
        side_layout.addStretch()

        self.lbl_info = QLabel("Silakan load berkas log untuk memulai.")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.lbl_info.setWordWrap(True)
        side_layout.addWidget(self.lbl_info)

        body_layout.addWidget(self.side_panel)
        main_layout.addLayout(body_layout, stretch=1)

        timeline_panel = QFrame()
        timeline_panel.setObjectName("panel")
        timeline_panel.setFixedHeight(70)
        timeline_layout = QVBoxLayout(timeline_panel)
        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.valueChanged.connect(self.seek_to)
        self.lbl_frame_count = QLabel("0 / 0")
        self.lbl_frame_count.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.lbl_frame_count.setAlignment(Qt.AlignRight)
        timeline_layout.addWidget(self.slider_timeline)
        timeline_layout.addWidget(self.lbl_frame_count)
        main_layout.addWidget(timeline_panel)

        controls_layout = QHBoxLayout()
        
        speed_layout = QHBoxLayout()
        lbl_speed_icon = QLabel("⚡ Kecepatan")
        lbl_speed_icon.setStyleSheet("color: #aaaaaa; font-size: 13px;")
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, len(self.speed_multipliers) - 1)
        self.slider_speed.setValue(3)
        self.slider_speed.setFixedWidth(140)
        self.slider_speed.valueChanged.connect(self.change_speed)
        self.lbl_speed_val = QLabel("1.00x")
        self.lbl_speed_val.setStyleSheet("font-size: 13px; width: 40px;")
        speed_layout.addWidget(lbl_speed_icon)
        speed_layout.addWidget(self.slider_speed)
        speed_layout.addWidget(self.lbl_speed_val)
        
        transport_layout = QHBoxLayout()
        transport_layout.setSpacing(12)
        
        style = self.style()
        self.btn_restart = QToolButton()
        self.btn_restart.setIcon(style.standardIcon(QStyle.SP_MediaSkipBackward))
        self.btn_restart.setFixedSize(36, 36)
        self.btn_restart.clicked.connect(self.go_to_start)

        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(style.standardIcon(QStyle.SP_MediaSeekBackward))
        self.btn_prev.setFixedSize(36, 36)
        self.btn_prev.clicked.connect(self.prev_frame)

        self.btn_play = QToolButton()
        self.btn_play.setObjectName("playBtn")
        self.btn_play.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setFixedSize(52, 52)
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_next = QToolButton()
        self.btn_next.setIcon(style.standardIcon(QStyle.SP_MediaSeekForward))
        self.btn_next.setFixedSize(36, 36)
        self.btn_next.clicked.connect(self.next_frame)

        self.btn_end = QToolButton()
        self.btn_end.setIcon(style.standardIcon(QStyle.SP_MediaSkipForward))
        self.btn_end.setFixedSize(36, 36)
        self.btn_end.clicked.connect(self.go_to_end)

        transport_layout.addStretch()
        transport_layout.addWidget(self.btn_restart)
        transport_layout.addWidget(self.btn_prev)
        transport_layout.addWidget(self.btn_play)
        transport_layout.addWidget(self.btn_next)
        transport_layout.addWidget(self.btn_end)
        transport_layout.addStretch()

        controls_layout.addLayout(speed_layout, stretch=1)
        controls_layout.addLayout(transport_layout, stretch=2)
        
        dummy_spacer = QLabel("")
        controls_layout.addWidget(dummy_spacer, stretch=1)
        
        main_layout.addLayout(controls_layout)

    def load_log(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Pilih Berkas Log Eksperimen", "", "JSON files (*.json);;All files (*.*)")
        if not filepath:
            return
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                return
            self.history = data
            self.costs = [entry[1] for entry in self.history]
            self.current_frame = 0
            self.playing = False
            self.timer.stop()
            self.slider_timeline.blockSignals(True)
            self.slider_timeline.setRange(0, len(self.history) - 1)
            self.slider_timeline.setValue(0)
            self.slider_timeline.blockSignals(False)
            self.lbl_info.setText(f"Log dimuat: {len(self.history)} tahapan iterasi.")
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.update_controls_state()
            self.draw_frame()
        except Exception:
            pass

    def update_controls_state(self):
        has_data = bool(self.history)
        for btn in (self.btn_restart, self.btn_prev, self.btn_play, self.btn_next, self.btn_end, self.slider_timeline):
            btn.setEnabled(has_data)

    def seek_to(self, frame):
        if not self.history:
            return
        self.current_frame = max(0, min(frame, len(self.history) - 1))
        self.draw_frame()

    def prev_frame(self):
        if self.history and self.current_frame > 0:
            self.current_frame -= 1
            self.slider_timeline.setValue(self.current_frame)

    def next_frame(self):
        if self.history and self.current_frame < len(self.history) - 1:
            self.current_frame += 1
            self.slider_timeline.setValue(self.current_frame)

    def go_to_start(self):
        self.seek_to(0)
        self.slider_timeline.setValue(0)

    def go_to_end(self):
        if self.history:
            end_idx = len(self.history) - 1
            self.seek_to(end_idx)
            self.slider_timeline.setValue(end_idx)

    def toggle_play(self):
        if not self.history:
            return
        self.playing = not self.playing
        if self.playing:
            if self.current_frame >= len(self.history) - 1:
                self.current_frame = 0
                self.slider_timeline.setValue(0)
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.apply_speed()
            self.timer.start()
        else:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()

    def play_step(self):
        if self.current_frame < len(self.history) - 1:
            self.next_frame()
        else:
            self.playing = False
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()

    def change_speed(self, value):
        multiplier = self.speed_multipliers[value]
        self.lbl_speed_val.setText(f"{multiplier:.2f}x")
        if self.playing:
            self.apply_speed()

    def apply_speed(self):
        idx = self.slider_speed.value()
        multiplier = self.speed_multipliers[idx]
        delay = int(self.base_delay_ms / multiplier)
        self.timer.setInterval(delay)

    def draw_frame(self):
        if not self.history:
            return
            
        state, cost = self.history[self.current_frame][0], self.history[self.current_frame][1]
        
        self.lbl_iter.setText(f"Iterasi {self.current_frame}")
        self.lbl_cost.setText(f"Cost: {cost:.2f}")
        
        if self.current_frame > 0:
            prev_cost = self.history[self.current_frame - 1][1]
            delta = cost - prev_cost
            arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
            self.lbl_delta.setText(f"{arrow} {abs(delta):.2f} dari iterasi sebelumnya")
        else:
            self.lbl_delta.setText("Kondisi awal")
            
        self.lbl_frame_count.setText(f"{self.current_frame} / {len(self.history) - 1}")
        
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(self.current_frame)
        self.slider_timeline.blockSignals(False)

        self.node_grid.set_state(state)
        self.sparkline.set_data(self.costs, self.current_frame)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Left:
            self.prev_frame()
        elif event.key() == Qt.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key_Home:
            self.go_to_start()
        elif event.key() == Qt.Key_End:
            self.go_to_end()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())