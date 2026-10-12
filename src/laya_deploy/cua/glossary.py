"""English <-> Chinese words for common UI labels, so "File menu" can find 檔案 on a Chinese desktop.

The planner usually writes English; Windows shows the labels in the display language. Lexical
shortlisting and the exact-name shortcut use these translations, and the target handed to laya
gets them as a hint, e.g. "File menu (檔案 / 文件)".
"""
from __future__ import annotations

# english word -> zh-TW / zh-CN labels Windows apps use for it
EN_ZH: dict[str, tuple[str, ...]] = {
    "file": ("檔案", "文件"), "edit": ("編輯",), "view": ("檢視", "查看"), "help": ("說明", "幫助"),
    "format": ("格式",), "tools": ("工具",), "window": ("視窗", "窗口"), "insert": ("插入",),
    "save": ("儲存", "保存"), "open": ("開啟", "打開"), "close": ("關閉",), "new": ("新增", "新建", "開新"),
    "cancel": ("取消",), "ok": ("確定",), "yes": ("是",), "no": ("否",), "apply": ("套用", "應用"),
    "delete": ("刪除",), "remove": ("移除",), "copy": ("複製", "复制"), "paste": ("貼上", "粘贴"),
    "cut": ("剪下", "剪切"), "undo": ("復原", "撤销"), "redo": ("取消復原", "重做"), "find": ("尋找", "查找"),
    "replace": ("取代", "替换"), "search": ("搜尋", "搜索"), "print": ("列印", "打印"), "settings": ("設定", "设置"),
    "options": ("選項", "选项"), "back": ("上一頁", "返回"), "forward": ("下一頁", "前进"), "send": ("傳送", "发送"),
    "name": ("名稱", "名称"), "folder": ("資料夾", "文件夹"), "desktop": ("桌面",), "downloads": ("下載", "下载"),
    "documents": ("文件",), "minimize": ("最小化",), "maximize": ("最大化",), "exit": ("結束", "退出"),
    "select": ("選取", "选择"), "all": ("全部",), "zoom": ("縮放", "缩放"), "font": ("字型", "字体"),
    "text": ("文字",), "editor": ("編輯器",), "area": ("區域",), "menu": ("選單", "功能表"),
    "next": ("下一步", "下一個"), "previous": ("上一步", "上一個"), "finish": ("完成",), "refresh": ("重新整理", "刷新"),
}

ZH_EN: dict[str, str] = {zh: en for en, zhs in EN_ZH.items() for zh in zhs}


def translations(target: str, english_tokens: list[str]) -> list[str]:
    """Labels the target may appear as in the other language (no duplicates, original order)."""
    out: list[str] = []
    for tok in english_tokens:
        out.extend(EN_ZH.get(tok, ()))
    for zh in sorted(ZH_EN, key=len, reverse=True):   # longest first: 取消復原 before 取消
        if zh in target:
            out.append(ZH_EN[zh])
    return list(dict.fromkeys(out))
