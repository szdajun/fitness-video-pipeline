# Matting Studio — UI 详细设计 (QML 前端 + Python 后端)

> **配套**: `matting-studio-design.md` (设计文档), `architecture.md` (架构图), `algorithms.md` (算法细节)
> **状态**: Phase 0 设计产物 (2026-07-04)
> **技术栈**: Python 3.11 + PyQt6 + Qt Quick / QML + qt-material + OpenCV (视频帧) + QSettings

> **⚠ 跨项目文档镜像 (2026-07-05)**:
> 本文档**仅作为主管线侧的 Matting Studio 设计根**。
> Matting Studio 是独立项目 (`F:\wkspace\matting-studio\`, 23 commit + v0.1.0 / v1.0.0 两个 tag),
> 本设计配套三件套在独立仓库 (`docs/architecture.md` / `docs/algorithms.md` / `docs/ui-design.md`)。
> 主管线不再维护这些文件 (本文件是 ghost 副本, 不要再 commit 内容改动)。

---

## 0. 关键决策 (2026-07-04 用户拍板)

**架构: QML 前端 + Python 后端** (混合架构)

**理由**:
- ✅ QML 视频原生渲染 (QML `VideoOutput` + `QAbstractVideoSurface`) = GPU 加速
- ✅ QML 现代 Material 风格 + 流畅动画
- ✅ QML 适合视频/时间轴/画布交互
- ✅ Python 后端 = RVM/YOLO/SAM2 推理 + Pipeline 调度 (AI 生态)
- ✅ 主流专业工具架构 (OBS/剪映/Kdenlive 同款)

**主窗口不用 PyQt 原生标题栏** = **QML FramelessWindow + 自绘 TitleBar**

---

## 1. 整体窗口结构 (QML)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [🎬 Matting Studio]  File  Edit  View  Tools  Help   [Preset ▼] [🌙] [─ □ ×] │  ← TitleBar.qml
├─────────────────────────────────────────────────────────────────────────┤
│ [📂] [💾] [▶] [⏸] [⏹] [✂ SAM2] [🖼] [🔍+] [🔍-]                          │  ← ToolBar.qml
├──────────┬──────────────────────────────────────────────────────────┤
│          │  Preview Player (QML VideoOutput)                          │
│  Project │  ┌────────────────────────────────────────────────────┐  │
│  Files   │  │                                                    │  │
│  ┌─────┐ │  │            视频预览 (GPU 加速渲染)                │  │
│  │clip1│ │  │                                                    │  │
│  │clip2│ │  │   SAM2 Canvas 叠加层 (透明 Canvas 2D)            │  │
│  │clip3│ │  │   - 左键=点选前景 (绿点)                          │  │
│  └─────┘ │  │   - 右键=点选背景 (红点)                          │  │
│          │  │   - 双击=提交 SAM2 推理                            │  │
│  Presets │  └────────────────────────────────────────────────────┘  │
│  ┌─────┐ │  [◄◄] [▶] [⏸] [⏹] [►►]  ──●────────  1:23 / 3:45       │
│  │fitn │ │                                                          │
│  │danc │ │  Timeline (QML Repeater + Canvas 2D)                    │
│  │multi│ │  ┌────────────────────────────────────────────────────┐  │
│  └─────┘ │  │ ▼ f0  f30  f60  f90  f120 ... (拖动播放头)        │  │
│          │  │ ▮▮▮▮▮▮▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯│  │
│  Models  │  └────────────────────────────────────────────────────┘  │
│  ┌─────┐ │                                                          │
│  │RVM  │ │  Properties Panel (QML Form)                            │
│  │YOLO │ │  ┌────────────────────────────────────────────────────┐  │
│  │SAM2 │ │  │ Frame: #90 (t=3.0s)                                │  │
│  └─────┘ │  │ Alpha: 0.957                                       │  │
│          │  │ Ghost Filter: ✓ ON                                 │  │
│  Log     │  │ Feather: 11 px                                    │  │
│  ┌─────┐ │  │ Encoder: h264_nvenc                               │  │
│  │INFO │ │  └────────────────────────────────────────────────────┘  │
│  │WARN │ │                                                          │
│  │ERR  │ │                                                          │
│  └─────┘ │                                                          │
└──────────┴──────────────────────────────────────────────────────────┤
│ [Status] Ready | GPU 18% | 18.2 FPS | Frame 1842/1842 | 2:34 elapsed │  ← StatusBar.qml
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 混合架构 (QML ↔ Python)

### 2.1 分层

```
┌─────────────────────────────────────────┐
│ QML 前端 (qml/)                          │
│  ├─ Main.qml              (主窗口)      │
│  ├─ TitleBar.qml          (自绘标题栏)  │
│  ├─ ToolBar.qml           (工具条)      │
│  ├─ SAM2Canvas.qml        (修帧画布)    │
│  ├─ Timeline.qml          (时间轴)      │
│  ├─ PropertiesPanel.qml   (属性)        │
│  ├─ LogDock.qml           (日志)        │
│  └─ StatusBar.qml         (状态栏)      │
├─────────────────────────────────────────┤
│ Python 后端 (modules/)                  │
│  ├─ backend.py            (QObject, 暴露 Pipeline) │
│  ├─ pipeline.py           (8 Stage 流水线)       │
│  ├─ matting.py            (RVM + YOLO)            │
│  ├─ postprocess.py        (中值 + SAM2)          │
│  └─ compose.py            (合成)                  │
└─────────────────────────────────────────┘
         ↕ Q_INVOKABLE / Signal / Property / Slot
```

### 2.2 主程序入口 (Python)

```python
# matting_studio.py
import sys
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQml import QQmlApplicationEngine, qmlRegisterType
from modules.backend import PipelineBackend
from modules.ui.video_surface import VideoFrameSurface

def main():
    app = QGuiApplication(sys.argv)
    app.setOrganizationName("MattingStudio")
    app.setApplicationName("Matting Studio")
    
    # 注册 QML 类型
    qmlRegisterType(PipelineBackend, "MattingStudio", 1, 0, "PipelineBackend")
    qmlRegisterType(VideoFrameSurface, "MattingStudio", 1, 0, "VideoFrameSurface")
    
    engine = QQmlApplicationEngine()
    engine.loadData(Main_QML)  # 加载 Main.qml
    
    if not engine.rootObjects():
        sys.exit(-1)
    
    sys.exit(app.exec())

Main_QML = b'''
import QtQuick
import QtQuick.Controls
import MattingStudio 1.0
ApplicationWindow {
    id: root
    visible: true
    width: 1600
    height: 900
    flags: Qt.FramelessWindowHint | Qt.Window
    
    PipelineBackend {
        id: backend
    }
    
    // ... 主窗口布局 (见 §3)
}
'''

if __name__ == "__main__":
    main()
```

---

## 3. Main.qml (主窗口 + TitleBar + 工具条)

```qml
// qml/Main.qml
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import MattingStudio 1.0

ApplicationWindow {
    id: root
    visible: true
    width: 1600
    height: 900
    visible: true
    flags: Qt.FramelessWindowHint | Qt.Window  // 去掉原生标题栏
    
    // 主题 (qt-material)
    Material.theme: Material.Dark
    Material.accent: Material.Teal
    Material.primary: Material.BlueGrey
    
    // Python 后端
    PipelineBackend {
        id: backend
    }
    
    // ============== 自绘 TitleBar ==============
    menuBar: TitleBar {
        id: titlebar
        height: 30
        backend: backend
    }
    
    // ============== 工具条 ==============
    header: ToolBar {
        id: toolbar
        height: 50
        background: Rectangle { color: "#2b2b2b" }
        
        RowLayout {
            anchors.fill: parent
            spacing: 4
            
            ToolButton {
                text: "📂"
                ToolTip.text: "Open Video (Ctrl+O)"
                onClicked: backend.openVideo()
            }
            ToolButton {
                text: "💾"
                ToolTip.text: "Save Project (Ctrl+S)"
                onClicked: backend.saveProject()
            }
            
            ToolSeparator {}
            
            ToolButton {
                text: backend.running ? "⏸" : "▶"
                ToolTip.text: backend.running ? "Pause" : "Run"
                onClicked: backend.running ? backend.pause() : backend.run()
            }
            ToolButton {
                text: "⏹"
                ToolTip.text: "Stop"
                onClicked: backend.stop()
            }
            
            ToolSeparator {}
            
            ToolButton {
                text: "✂"
                ToolTip.text: "SAM2 Repair Tool (S)"
                checkable: true
                checked: sam2Mode
                onClicked: sam2Mode = !sam2Mode
            }
            
            ToolSeparator {}
            
            ToolButton {
                text: "🖼"
                ToolTip.text: "Export (Ctrl+E)"
                onClicked: backend.exportVideo()
            }
            
            Item { Layout.fillWidth: true }  // 弹性空间
            
            ToolButton {
                text: "⚙"
                ToolTip.text: "Settings"
                onClicked: settingsDialog.open()
            }
        }
    }
    
    // ============== 主内容 ==============
    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal
        
        // 左侧: 项目文件 + 预设 + 模型
        LeftPanel {
            id: leftPanel
            SplitView.preferredWidth: 200
            SplitView.minimumWidth: 150
            backend: backend
        }
        
        // 中央: 视频预览 + 时间轴
        SplitView {
            orientation: Qt.Vertical
            
            // 上: 视频预览 + SAM2 画布
            VideoPreview {
                id: preview
                SplitView.fillHeight: true
                backend: backend
                sam2Mode: sam2Mode
            }
            
            // 下: 时间轴
            Timeline {
                id: timeline
                SplitView.preferredHeight: 120
                SplitView.minimumHeight: 80
                backend: backend
            }
        }
        
        // 右侧: 属性面板
        PropertiesPanel {
            id: properties
            SplitView.preferredWidth: 280
            SplitView.minimumWidth: 200
            backend: backend
        }
    }
    
    // ============== 底部 Dock: 日志 ==============
    footer: LogDock {
        id: logDock
        height: 150
        backend: backend
    }
    
    // ============== 状态栏 ==============
    footer: StatusBar {
        id: statusbar
        backend: backend
    }
    
    // ============== 状态属性 ==============
    property bool sam2Mode: false
    
    // 快捷键
    Shortcut { sequence: "Ctrl+O"; onActivated: backend.openVideo() }
    Shortcut { sequence: "Ctrl+S"; onActivated: backend.saveProject() }
    Shortcut { sequence: "Ctrl+E"; onActivated: backend.exportVideo() }
    Shortcut { sequence: "F5"; onActivated: backend.run() }
    Shortcut { sequence: "F6"; onActivated: backend.pause() }
    Shortcut { sequence: "F7"; onActivated: backend.stop() }
    Shortcut { sequence: "S"; onActivated: sam2Mode = !sam2Mode }
    
    // SAM2 修复回调
    Connections {
        target: backend
        function onSam2RepairRequested(frameIdx, points, labels) {
            // 触发 SAM2 推理 (异步)
            backend.sam2Repair(frameIdx, points, labels)
        }
    }
}
```

---

## 4. TitleBar.qml (自绘标题栏)

```qml
// qml/TitleBar.qml
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    height: 30
    color: "#2b2b2b"
    
    property var backend
    
    // 拖动窗口支持
    property point dragStart: Qt.point(0, 0)
    property bool dragging: false
    
    // 顶级菜单
    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 8
        spacing: 0
        
        // Logo + 应用名
        Label {
            text: "🎬 Matting Studio"
            color: "#ffffff"
            font.bold: true
            Layout.leftMargin: 4
            Layout.rightMargin: 12
            
            MouseArea {
                anchors.fill: parent
                onDoubleClicked: window.toggleMaximize()
            }
        }
        
        // 顶级菜单按钮
        Repeater {
            model: [
                { label: "File", menu: "fileMenu" },
                { label: "Edit", menu: "editMenu" },
                { label: "View", menu: "viewMenu" },
                { label: "Tools", menu: "toolsMenu" },
                { label: "Help", menu: "helpMenu" },
            ]
            
            delegate: MenuBarItem {
                text: modelData.label
                Menu {
                    id: menu_${modelData.menu}
                    // 内容由各 Menu 组件填充
                }
            }
        }
        
        Item { Layout.fillWidth: true }  // 弹性空间
        
        // 预设选择
        RowLayout {
            spacing: 4
            Label { text: "Preset:"; color: "#888888"; font.pixelSize: 11 }
            ComboBox {
                model: ["Single Person", "Multi Person", "Livestream", "Pro Filmmaking", "Custom..."]
                currentIndex: 0
                onActivated: backend.setPreset(currentIndex)
                Layout.preferredWidth: 150
            }
        }
        
        // 主题切换
        ToolButton {
            text: "🌙"
            implicitWidth: 30
            implicitHeight: 24
            onClicked: themeSwitch()
        }
        
        // 窗口按钮
        WindowButton { text: "─"; onClicked: window.showMinimized() }
        WindowButton { text: "□"; onClicked: window.toggleMaximize() }
        WindowButton { text: "×"; onClicked: Qt.quit(); hoverColor: "#e81123" }
    }
    
    // 拖动窗口 (排除菜单按钮区域)
    MouseArea {
        anchors.fill: parent
        z: -1
        propagateComposedEvents: true
        property point startPos: Qt.point(0, 0)
        onPressed: (mouse) => {
            startPos = Qt.point(mouse.x, mouse.y)
        }
        onPositionChanged: (mouse) => {
            var dx = mouse.x - startPos.x
            var dy = mouse.y - startPos.y
            window.x += dx
            window.y += dy
        }
        onDoubleClicked: window.toggleMaximize()
    }
}

// 窗口按钮组件
component WindowButton: Button {
    implicitWidth: 46
    implicitHeight: 30
    background: Rectangle {
        color: parent.hovered ? (parent.text === "×" ? "#e81123" : "#3d3d3d") : "transparent"
    }
    contentItem: Label {
        text: parent.text
        color: "#ffffff"
        font.pixelSize: 12
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
```

**顶级菜单子组件** (qml/Menus.qml):

```qml
// qml/Menus.qml
import QtQuick
import QtQuick.Controls

Menu {
    id: fileMenu
    title: qsTr("&File")
    
    MenuItem { text: qsTr("Open Video..."); shortcut: "Ctrl+O"; onTriggered: backend.openVideo() }
    MenuItem { text: qsTr("Open Project..."); shortcut: "Ctrl+Shift+O"; onTriggered: backend.openProject() }
    MenuItem { text: qsTr("Save Project"); shortcut: "Ctrl+S"; onTriggered: backend.saveProject() }
    MenuSeparator {}
    MenuItem { text: qsTr("Import Background..."); shortcut: "Ctrl+B"; onTriggered: backend.importBackground() }
    MenuItem { text: qsTr("Export Video..."); shortcut: "Ctrl+E"; onTriggered: backend.exportVideo() }
    MenuSeparator {}
    MenuItem { text: qsTr("Quit"); shortcut: "Ctrl+Q"; onTriggered: Qt.quit() }
}
```

---

## 5. VideoPreview.qml (视频预览 + SAM2 画布)

```qml
// qml/VideoPreview.qml
import QtQuick
import QtQuick.Controls
import MattingStudio 1.0

Item {
    id: root
    property var backend
    property bool sam2Mode: false
    
    // 视频帧渲染表面
    VideoFrameSurface {
        id: videoSurface
        anchors.fill: parent
        backend: backend
    }
    
    // SAM2 画布叠加层 (透明, 接收鼠标事件)
    Canvas {
        id: sam2Canvas
        anchors.fill: parent
        visible: root.sam2Mode
        opacity: 1.0
        
        // 收集点选
        property var points: []  // [{x, y, label}, ...]
        
        // 鼠标事件
        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.LeftButton | Qt.RightButton
            onPressed: (mouse) => {
                if (mouse.button === Qt.LeftButton) {
                    // 前景点 (绿)
                    sam2Canvas.points.push({x: mouse.x, y: mouse.y, label: 1})
                } else if (mouse.button === Qt.RightButton) {
                    // 背景点 (红)
                    sam2Canvas.points.push({x: mouse.x, y: mouse.y, label: 0})
                }
                sam2Canvas.requestPaint()
            }
            onDoubleClicked: {
                // 提交 SAM2 推理
                var pts = sam2Canvas.points.map(p => Qt.point(p.x, p.y))
                var labels = sam2Canvas.points.map(p => p.label)
                backend.sam2Repair(currentFrameIdx, pts, labels)
                sam2Canvas.points = []
                sam2Canvas.requestPaint()
            }
        }
        
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            // 画点
            for (var i = 0; i < points.length; i++) {
                var p = points[i]
                ctx.fillStyle = p.label === 1 ? "#00ff00" : "#ff0000"
                ctx.beginPath()
                ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI)
                ctx.fill()
                ctx.strokeStyle = "#ffffff"
                ctx.lineWidth = 2
                ctx.stroke()
            }
        }
    }
    
    // 当前帧号 (从 backend 同步)
    property int currentFrameIdx: 0
    Connections {
        target: backend
        function onCurrentFrameChanged(frameIdx) {
            currentFrameIdx = frameIdx
        }
    }
}
```

**VideoFrameSurface (Python QObject)**:

```python
# modules/ui/video_surface.py
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty
from PyQt6.QtMultimedia import QAbstractVideoSurface, QVideoFrame, QVideoSurfaceFormat
from PyQt6.QtGui import QImage
import numpy as np
import cv2

class VideoFrameSurface(QAbstractVideoSurface):
    """QML 视频帧渲染表面, 接收 QVideoFrame 并显示."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend = None
    
    def supportedPixelFormats(self, handleType=QAbstractVideoSurface.HandleType.NoHandle):
        return [QVideoFrame.PixelFormat.Format_BGR24, QVideoFrame.PixelFormat.Format_RGB24]
    
    def present(self, frame: QVideoFrame):
        """接收视频帧 (从 Python Pipeline 推过来)."""
        if not frame.isValid():
            return False
        # 转换 QVideoFrame → QImage → OpenCV ndarray
        frame.map(QAbstractVideoSurface.MapDirection.ReadOnly)
        img = QImage(
            frame.bits(), frame.width(), frame.height(),
            frame.bytesPerLine(), QImage.Format.Format_BGR24
        )
        img = img.copy()  # 必须 copy, 否则 frame.unmap 后失效
        frame.unmap()
        
        # 转换为 OpenCV ndarray (SAM2 修帧用)
        ndarray = self._qimage_to_ndarray(img)
        # 推给 SAM2 画布 (QML 用)
        self.frameReady.emit(ndarray)
        return True
    
    @pyqtSignal
    def frameReady(self, ndarray): pass
    
    def _qimage_to_ndarray(self, img: QImage) -> np.ndarray:
        """QImage → numpy ndarray (H, W, 3) BGR uint8."""
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3).copy()
        return arr  # QImage Format_BGR24 → BGR
```

---

## 6. Timeline.qml (时间轴)

```qml
// qml/Timeline.qml
import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    height: 120
    color: "#1a1a1a"
    border.color: "#333333"
    
    property var backend
    property int totalFrames: 0
    property int currentFrame: 0
    
    // 缩略图条
    Flickable {
        id: flickable
        anchors.fill: parent
        contentWidth: totalFrames * 30  // 每帧 30px
        contentHeight: parent.height
        
        // 缩略图行
        Row {
            spacing: 1
            Repeater {
                model: totalFrames
                delegate: Rectangle {
                    width: 28
                    height: 80
                    color: index === currentFrame ? "#0078d4" : "#3d3d3d"
                    border.color: "#555555"
                    Label {
                        anchors.centerIn: parent
                        text: index
                        color: "#ffffff"
                        font.pixelSize: 9
                    }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: backend.seekFrame(index)
                    }
                }
            }
        }
        
        // 播放头 (红线)
        Rectangle {
            x: currentFrame * 30
            width: 2
            height: parent.height
            color: "#ff0000"
        }
    }
    
    // 时间标签
    Label {
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.margins: 4
        text: formatTime(currentFrame / 30) + " / " + formatTime(totalFrames / 30)
        color: "#888888"
        font.family: "Consolas"
        font.pixelSize: 10
    }
    
    function formatTime(seconds) {
        var m = Math.floor(seconds / 60)
        var s = Math.floor(seconds % 60)
        return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s
    }
    
    Connections {
        target: backend
        function onTotalFramesChanged(frames) { totalFrames = frames }
        function onCurrentFrameChanged(frame) { currentFrame = frame }
    }
}
```

---

## 7. PropertiesPanel.qml (属性面板)

```qml
// qml/PropertiesPanel.qml
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    color: "#2b2b2b"
    
    property var backend
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8
        
        Label {
            text: "Properties"
            color: "#ffffff"
            font.bold: true
            font.pixelSize: 14
        }
        
        // 当前帧
        RowLayout {
            Label { text: "Frame:"; color: "#cccccc" }
            SpinBox {
                from: 0
                to: 99999
                value: backend.currentFrame
                onValueModified: backend.seekFrame(value)
            }
        }
        
        // Mask Thresh
        RowLayout {
            Label { text: "Mask Thresh:"; color: "#cccccc" }
            Slider {
                from: 0.0
                to: 1.0
                stepSize: 0.05
                value: backend.maskThresh
                onValueChanged: backend.setMaskThresh(value)
            }
            Label { text: backend.maskThresh.toFixed(2); color: "#ffffff" }
        }
        
        // Ghost Filter
        RowLayout {
            Switch {
                text: "YOLO Ghost Filter"
                checked: backend.ghostFilter
                onCheckedChanged: backend.setGhostFilter(checked)
            }
        }
        
        // Edge Feather
        RowLayout {
            Label { text: "Edge Feather:"; color: "#cccccc" }
            SpinBox {
                from: 0
                to: 31
                stepSize: 2
                value: backend.edgeFeather
                onValueChanged: backend.setEdgeFeather(value)
            }
        }
        
        // 编码器
        RowLayout {
            Label { text: "Encoder:"; color: "#cccccc" }
            ComboBox {
                model: ["h264_nvenc", "h264_qsv", "libx264", "ProRes 4444"]
                currentIndex: 0
                onActivated: backend.setEncoder(currentIndex)
            }
        }
        
        // GPU 占用
        GroupBox {
            title: "GPU Memory"
            Layout.fillWidth: true
            ColumnLayout {
                Label { text: "RVM: " + backend.gpuRvmMB.toFixed(0) + " MB"; color: "#4caf50" }
                Label { text: "YOLO: " + backend.gpuYoloMB.toFixed(0) + " MB"; color: "#4caf50" }
                Label { text: "SAM2: " + backend.gpuSam2MB.toFixed(0) + " MB"; color: "#4caf50" }
                Label { text: "Total: " + backend.gpuTotalMB.toFixed(0) + " MB / 10240 MB"; 
                        color: backend.gpuTotalMB > 9000 ? "#f44336" : "#4caf50" }
            }
        }
    }
}
```

---

## 8. LogDock.qml (日志 + 进度)

```qml
// qml/LogDock.qml
import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    height: 150
    color: "#1a1a1a"
    
    property var backend
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 4
        
        // 进度条
        ProgressBar {
            Layout.fillWidth: true
            value: backend.progress
        }
        
        // 日志
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            TextArea {
                id: logText
                readOnly: true
                color: "#00ff00"
                background: Rectangle { color: "#0a0a0a" }
                font.family: "Consolas"
                font.pixelSize: 10
                text: backend.logs
            }
        }
    }
    
    // 自动滚动到底部
    Connections {
        target: backend
        function onLogsChanged() {
            logText.cursorPosition = logText.length
        }
    }
}
```

---

## 9. StatusBar.qml (状态栏)

```qml
// qml/StatusBar.qml
import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    height: 24
    color: "#1a1a1a"
    border.color: "#333333"
    
    property var backend
    
    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 8
        anchors.rightMargin: 8
        
        Label {
            text: backend.status
            color: "#cccccc"
        }
        
        Item { Layout.fillWidth: true }
        
        Label {
            text: "GPU " + backend.gpuPercent.toFixed(0) + "%"
            color: "#888888"
        }
        Label {
            text: "FPS " + backend.fps.toFixed(1)
            color: "#888888"
        }
        Label {
            text: "Frame " + backend.currentFrame + "/" + backend.totalFrames
            color: "#888888"
        }
    }
}
```

---

## 10. Python Backend (PipelineBackend QObject)

```python
# modules/backend.py
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, pyqtProperty, QThread
from PyQt6.QtCore import pyqtSignal as Signal
import numpy as np
import time

class PipelineBackend(QObject):
    """Python 后端, 暴露 Pipeline 状态给 QML."""
    
    # ============== 属性 (QML 可见) ==============
    
    # 当前状态
    statusChanged = Signal()
    runningChanged = Signal()
    progressChanged = Signal()
    fpsChanged = Signal()
    currentFrameChanged = Signal()
    totalFramesChanged = Signal()
    
    # 渲染参数
    maskThreshChanged = Signal()
    ghostFilterChanged = Signal()
    edgeFeatherChanged = Signal()
    gpuStatsChanged = Signal()
    
    # 日志
    logsChanged = Signal()
    
    # SAM2 修复请求 (QML → Python)
    sam2RepairRequested = Signal(int, list, list)  # frame_idx, points, labels
    
    def __init__(self):
        super().__init__()
        self._status = "Ready"
        self._running = False
        self._progress = 0
        self._fps = 0.0
        self._current_frame = 0
        self._total_frames = 0
        self._mask_thresh = 0.0
        self._ghost_filter = True
        self._edge_feather = 11
        self._gpu_rvm_mb = 0
        self._gpu_yolo_mb = 0
        self._gpu_sam2_mb = 0
        self._gpu_percent = 0
        self._logs = ""
        # Pipeline 实例 (后续)
        # self.pipeline = PipelineBackend_Impl(...)
    
    # ============== 属性 getter/setter (QML 绑定) ==============
    
    @pyqtProperty(str, notify=statusChanged)
    def status(self):
        return self._status
    
    @status.setter
    def status(self, value):
        if self._status != value:
            self._status = value
            self.statusChanged.emit()
    
    @pyqtProperty(bool, notify=runningChanged)
    def running(self):
        return self._running
    
    @running.setter
    def running(self, value):
        if self._running != value:
            self._running = value
            self.runningChanged.emit()
    
    @pyqtProperty(float, notify=progressChanged)
    def progress(self):
        return self._progress
    
    @progress.setter
    def progress(self, value):
        if abs(self._progress - value) > 0.01:
            self._progress = value
            self.progressChanged.emit()
    
    @pyqtProperty(float, notify=fpsChanged)
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, value):
        if abs(self._fps - value) > 0.1:
            self._fps = value
            self.fpsChanged.emit()
    
    @pyqtProperty(int, notify=currentFrameChanged)
    def currentFrame(self):
        return self._current_frame
    
    @currentFrame.setter
    def currentFrame(self, value):
        if self._current_frame != value:
            self._current_frame = value
            self.currentFrameChanged.emit()
    
    @pyqtProperty(int, notify=totalFramesChanged)
    def totalFrames(self):
        return self._total_frames
    
    @totalFrames.setter
    def totalFrames(self, value):
        if self._total_frames != value:
            self._total_frames = value
            self.totalFramesChanged.emit()
    
    @pyqtProperty(float, notify=maskThreshChanged)
    def maskThresh(self):
        return self._mask_thresh
    
    @maskThresh.setter
    def mask_thresh(self, value):
        if abs(self._mask_thresh - value) > 0.01:
            self._mask_thresh = value
            self.maskThreshChanged.emit()
    
    @pyqtProperty(bool, notify=ghostFilterChanged)
    def ghostFilter(self):
        return self._ghost_filter
    
    @ghostFilter.setter
    def ghost_filter(self, value):
        if self._ghost_filter != value:
            self._ghost_filter = value
            self.ghostFilterChanged.emit()
    
    @pyqtProperty(int, notify=edgeFeatherChanged)
    def edgeFeather(self):
        return self._edge_feather
    
    @pyqtProperty(float, notify=gpuStatsChanged)
    def gpuRvmMB(self):
        return self._gpu_rvm_mb
    
    @pyqtProperty(float, notify=gpuStatsChanged)
    def gpuYoloMB(self):
        return self._gpu_yolo_mb
    
    @pyqtProperty(float, notify=gpuStatsChanged)
    def gpuSam2MB(self):
        return self._gpu_sam2_mb
    
    @pyqtProperty(float, notify=gpuStatsChanged)
    def gpuTotalMB(self):
        return self._gpu_rvm_mb + self._gpu_yolo_mb + self._gpu_sam2_mb
    
    @pyqtProperty(float, notify=gpuStatsChanged)
    def gpuPercent(self):
        return self._gpu_percent
    
    @pyqtProperty(str, notify=logsChanged)
    def logs(self):
        return self._logs
    
    # ============== QML 调用方法 (@pyqtSlot) ==============
    
    @pyqtSlot()
    def openVideo(self):
        """QML 调用: 打开视频文件."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            None, "Open Video", "", "Video (*.mp4 *.mov *.webm *.avi)"
        )
        if path:
            self._load_video(path)
            self.log("INFO", f"Opened video: {path}")
    
    @pyqtSlot()
    def saveProject(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            None, "Save Project", "", "Matting Studio Project (*.msproj)"
        )
        if path:
            self._save_project(path)
            self.log("INFO", f"Saved project: {path}")
    
    @pyqtSlot()
    def run(self):
        """开始渲染 (异步 QThread)."""
        if self._running:
            return
        self.running = True
        self.status = "Running..."
        # 启动 QThread 跑 Pipeline
        self._worker_thread = QThread()
        self._worker = PipelineWorker(self)  # 见下
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker_thread.start()
    
    @pyqtSlot()
    def pause(self):
        self.status = "Paused"
        if self._worker:
            self._worker.pause()
    
    @pyqtSlot()
    def stop(self):
        self.status = "Stopped"
        if self._worker:
            self._worker.stop()
    
    @pyqtSlot(int)
    def seekFrame(self, frame_idx):
        if self._worker:
            self._worker.seek(frame_idx)
    
    @pyqtSlot(int, list, list)
    def sam2Repair(self, frame_idx, points, labels):
        """QML 调用: SAM2 修帧 (异步)."""
        self.log("INFO", f"SAM2 repair on frame {frame_idx}, {len(points)} points")
        if self._worker:
            self._worker.sam2_repair(frame_idx, points, labels)
    
    @pyqtSlot(int)
    def setPreset(self, index):
        presets = ["single_person", "multi_person", "livestream", "pro_filmmaking"]
        if 0 <= index < len(presets):
            self._apply_preset(presets[index])
            self.log("INFO", f"Preset changed: {presets[index]}")
    
    @pyqtSlot(float)
    def setMaskThresh(self, value):
        self.mask_thresh = value
        self.log("INFO", f"Mask thresh: {value}")
    
    @pyqtSlot(bool)
    def setGhostFilter(self, value):
        self.ghost_filter = value
        self.log("INFO", f"Ghost filter: {value}")
    
    @pyqtSlot(int)
    def setEdgeFeather(self, value):
        self._edge_feather = value
        self.edgeFeatherChanged.emit()
        self.log("INFO", f"Edge feather: {value}")
    
    @pyqtSlot(int)
    def setEncoder(self, index):
        encoders = ["h264_nvenc", "h264_qsv", "libx264", "ProRes 4444"]
        if 0 <= index < len(encoders):
            self._encoder = encoders[index]
            self.log("INFO", f"Encoder: {self._encoder}")
    
    # ============== 内部方法 ==============
    
    def log(self, level, message):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self._logs += f"[{ts}] [{level}] {message}\n"
        self.logsChanged.emit()
    
    def _on_progress(self, current, total, fps):
        self.currentFrame = current
        self.totalFrames = total
        self.fps = fps
        self.progress = (current / total) if total > 0 else 0
    
    def _on_frame_ready(self, frame_idx, ndarray):
        """推视频帧给 VideoFrameSurface (QML 显示)."""
        # 推给 QML (通过 signal)
        self.frameReady.emit(frame_idx, ndarray)
    
    def _on_finished(self):
        self.running = False
        self.status = "Done"
        self.log("INFO", "Rendering finished")


class PipelineWorker(QObject):
    """Pipeline 工作线程 (在 QThread 跑)."""
    
    progress = Signal(int, int, float)  # current, total, fps
    frame_ready = Signal(int, object)    # frame_idx, ndarray
    finished = Signal()
    
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self._paused = False
        self._stopped = False
        self._seek_target = None
    
    @pyqtSlot()
    def run(self):
        """主渲染循环."""
        # 1. 加载视频
        # 2. 8 Stage 流水线
        for i, frame in enumerate(self._pipeline):
            if self._stopped:
                break
            while self._paused:
                time.sleep(0.1)
            if self._seek_target is not None:
                i = self._seek_target
                self._seek_target = None
            # 3. 处理
            processed = self._process_frame(frame)
            # 4. 推给 QML
            self.frame_ready.emit(i, processed)
            # 5. 进度
            self.progress.emit(i + 1, self._total, self._fps_calc())
        self.finished.emit()
    
    def pause(self):
        self._paused = not self._paused
    
    def stop(self):
        self._stopped = True
    
    def seek(self, frame_idx):
        self._seek_target = frame_idx
```

---

## 11. 关键 UI 决策 (QML)

| 决策 | 理由 |
|------|------|
| **QML ApplicationWindow + FramelessWindowHint** | 自绘 TitleBar, 跨平台一致, 影视行业标准 |
| **QML 视频渲染用 VideoOutput + QAbstractVideoSurface** | GPU 加速, Python 端推送 QVideoFrame |
| **Python Backend 用 QObject + pyqtProperty/Signal/Slot** | QML 直接绑定, 无需手动 Q_INVOKABLE 包装 |
| **QThread 跑 Pipeline (主线程只更新 UI)** | UI 不卡顿, Pipeline 后台跑 |
| **SAM2 画布用 QML Canvas 2D** | 矢量点选/框选, GPU 加速, 比 QWidget QLabel 灵活 |
| **时间轴用 QML Repeater + Rectangle** | 缩略图懒加载, 拖动播放头 |
| **主题用 qt-material (dark_teal)** | 现代 Material 风格, 5 行代码换主题 |
| **Dock 用 QML SplitView** | 用户可拖动调整, 持久化布局 |
| **快捷键用 QML Shortcut** | 原生支持, 可在 preferences 自定义 |

---

## 12. 数据流 (QML ↔ Python)

```
QML (UI)                           Python (Backend)
├─ Main.qml                        ├─ PipelineBackend (QObject)
│  ├─ TitleBar.qml                 │   ├─ pyqtProperty: status, fps, ...
│  ├─ ToolBar.qml                  │   ├─ pyqtSignal: progress, frame_ready, ...
│  ├─ VideoPreview.qml             │   └─ pyqtSlot: openVideo(), run(), sam2Repair()
│  │  └─ VideoFrameSurface         │
│  ├─ Timeline.qml                 ├─ PipelineWorker (QObject in QThread)
│  ├─ PropertiesPanel.qml          │   └─ 8 Stage: input → preprocess → matting → ...
│  ├─ LogDock.qml                       │      → postprocess → compose → export
│  └─ StatusBar.qml                
│                                   
├─ 绑定 (Property binding)         ├─ 发送 (Signal emit)
│  backend.running  ←──────        │  self.running = True; runningChanged.emit()
│  backend.fps  ←───────────       │  self.fps = 30.5; fpsChanged.emit()
│  backend.logs  ←──────────       │  self._logs += "..."; logsChanged.emit()
│                                   
├─ 调用 (@pyqtSlot)                ├─ 处理
│  backend.openVideo()  ──────→    │  QFileDialog → _load_video()
│  backend.run()  ────────────→    │  QThread.start() → PipelineWorker.run()
│  backend.sam2Repair(...)  ─→    │  QThread → SAM2.predict() → emit frame
```

---

## 13. 视频帧推送 (Python → QML)

```python
# 在 PipelineWorker 中
def _process_frame(self, frame: np.ndarray) -> np.ndarray:
    # ... 8 stage 处理 ...
    return processed_frame

@pyqtSlot()
def run(self):
    for i, frame in enumerate(self._pipeline):
        processed = self._process_frame(frame)
        # 1. 推给 QML 显示
        self.frame_ready.emit(i, processed)
        # 2. QML VideoFrameSurface.present() 接收并显示
        # 3. 进度
        self.progress.emit(i, total, fps)
```

```qml
// qml/VideoPreview.qml
VideoFrameSurface {
    id: videoSurface
    backend: backend
    // Python 推 frame_ready 时, QML 自动更新 (绑定到 frameSource)
}
```

---

## 14. 实施路线 (Phase 2: 2-3 月)

**Week 9-10**: 基础设施
- [ ] QML 项目结构 (qml/ + modules/)
- [ ] PipelineBackend QObject 框架 (属性 + 信号 + 槽)
- [ ] TitleBar + MenuBar (自绘)
- [ ] qt-material 主题集成 + 暗/亮切换

**Week 11-12**: 核心交互
- [ ] VideoFrameSurface (QAbstractVideoSurface)
- [ ] VideoPreview.qml (VideoOutput + Canvas 2D SAM2)
- [ ] Timeline.qml (缩略图 + 播放头)
- [ ] PropertiesPanel.qml (参数实时绑定)

**Week 13-14**: Pipeline 集成
- [ ] QThread 跑 Pipeline
- [ ] 进度条 + FPS + GPU 监控
- [ ] SAM2 修帧工作流 (点选 → 推理 → 精炼 mask)
- [ ] 4 方案预设

**Week 15-16**: 高级 + 打包
- [ ] 主题切换 (暗/亮/自定义)
- [ ] QSettings 持久化
- [ ] PyInstaller 打包 (Windows .exe + macOS .app + Linux AppImage)
- [ ] 用户文档 + 教程视频

**Phase 2 验收**:
- QML 桌面 UI 完整
- SAM2 修帧工作流 (穿帮帧 1-2 次点击修好)
- 4 方案预设
- 跨平台打包
- v1.0.0 GitHub release

---

## 15. 性能优化 (QML 特定)

| 优化 | 实施 |
|------|------|
| **QML Repeater 懒加载** | 时间轴只在 viewport 范围渲染缩略图 |
| **QQuickPaintedItem 替代 Canvas 2D** (高性能) | SAM2 画布点选用 `QQuickPaintedItem` (OpenGL 加速) |
| **VideoOutput Surface 直接渲染** | 跳过 QImage 转换, 直接传 QVideoFrame |
| **QML 编译 (qmlc)** | 启动时间 <2s |
| **多线程 Pipeline** | 主线程只更新 UI, 渲染线程 8 Stage 流水 |
| **GPU 异步推理** | RVM/YOLO/SAM2 异步调用, 不阻塞 UI |

---

**下一步**: 创建新 GitHub repo 脚手架 (`F:/wkspace/matting-studio/`), 准备 Phase 1 编码 (Python 后端 + 8 Stage 流水线, 先不写 QML).
