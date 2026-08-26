# gui/styles/theme.py

DARK_THEME_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #121214;
    color: #f0f0f5;
    font-family: 'Segoe UI', 'Ubuntu', 'Helvetica Neue', sans-serif;
}

QToolBar {
    background-color: #1a1a1e;
    border-bottom: 1px solid #2a2a30;
    padding: 6px 10px;
    spacing: 10px;
}

QMenu {
    background-color: #1e1e24;
    color: #f0f0f5;
    border: 1px solid #33333d;
    padding: 5px;
    border-radius: 6px;
}

QMenu::item {
    padding: 6px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #00887a;
    color: #ffffff;
}

QGroupBox {
    background-color: #18181c;
    border: 1px solid #282830;
    border-radius: 8px;
    margin-top: 14px;
    font-weight: bold;
    color: #00ffcc;
    padding: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
}

QComboBox, QDoubleSpinBox, QLineEdit {
    background-color: #23232a;
    color: #f0f0f5;
    border: 1px solid #383844;
    border-radius: 5px;
    padding: 6px;
    min-height: 24px;
}

QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #00ffcc;
}

QPushButton {
    background-color: #23232a;
    color: #f0f0f5;
    border: 1px solid #383844;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #2e2e38;
    border-color: #00ffcc;
}

QPushButton#scan_btn {
    background-color: #00887a;
    color: #ffffff;
    font-size: 13px;
    padding: 8px 18px;
    border-radius: 6px;
}

QPushButton#scan_btn:checked {
    background-color: #d9383a;
}

QProgressBar {
    border: 1px solid #383844;
    border-radius: 5px;
    text-align: center;
    background-color: #23232a;
    color: #ffffff;
    height: 16px;
}

QProgressBar::chunk {
    background-color: #ff6b35;
    border-radius: 4px;
}

QStatusBar {
    background-color: #121214;
    color: #888894;
    border-top: 1px solid #222228;
    font-size: 11px;
}
"""

LIGHT_THEME_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #f4f5f8;
    color: #222226;
    font-family: 'Segoe UI', 'Ubuntu', 'Helvetica Neue', sans-serif;
}

QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #d4d6dd;
    padding: 6px 10px;
    spacing: 10px;
}

QMenu {
    background-color: #ffffff;
    color: #222226;
    border: 1px solid #cfd2dc;
    padding: 5px;
    border-radius: 6px;
}

QMenu::item {
    padding: 6px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #00887a;
    color: #ffffff;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #dcdfe6;
    border-radius: 8px;
    margin-top: 14px;
    font-weight: bold;
    color: #00887a;
    padding: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
}

QComboBox, QDoubleSpinBox, QLineEdit {
    background-color: #fafbfc;
    color: #222226;
    border: 1px solid #c2c6d0;
    border-radius: 5px;
    padding: 6px;
    min-height: 24px;
}

QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #00887a;
}

QPushButton {
    background-color: #eaecf1;
    color: #222226;
    border: 1px solid #c2c6d0;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #dfe3eb;
    border-color: #00887a;
}

QPushButton#scan_btn {
    background-color: #00887a;
    color: #ffffff;
    font-size: 13px;
    padding: 8px 18px;
    border-radius: 6px;
}

QPushButton#scan_btn:checked {
    background-color: #d9383a;
}

QProgressBar {
    border: 1px solid #c2c6d0;
    border-radius: 5px;
    text-align: center;
    background-color: #e2e5ec;
    color: #111114;
    height: 16px;
}

QProgressBar::chunk {
    background-color: #ff6b35;
    border-radius: 4px;
}

QStatusBar {
    background-color: #f4f5f8;
    color: #555562;
    border-top: 1px solid #d4d6dd;
    font-size: 11px;
}
"""