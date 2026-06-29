#!/usr/bin/env python3
"""自动递增 MK/Mod 版本号 — 供 OpenCode commit 前调用

用法:
    python3 scripts/bump_version.py [--mk | --mod | --no-bump]
        --mk       强制 MK++ (Mod 重置为 0)
        --mod      强制 Mod++ (默认)
        --no-bump  仅读取当前版本, 不递增

读取 docs/v6/VERSION, 分析 git diff 决定递增策略:
    - models/**/forward() 数据流重构 或 新增/删除模块文件 → MK++
    - 其他修改 → Mod++
    - 仅文档/配置 → no-bump
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO / "docs" / "v6" / "VERSION"


def get_git_diff_files():
    """返回 (staged_files, unstaged_files)"""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, cwd=REPO,
    )
    staged = set(result.stdout.strip().split("\n")) - {""}
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True, text=True, cwd=REPO,
    )
    unstaged = set(result.stdout.strip().split("\n")) - {""}
    return staged, unstaged


def read_version():
    with open(VERSION_FILE) as f:
        text = f.read()
    ver = {}
    for line in text.strip().split("\n"):
        m = re.match(r"^(\w[\w_]*)\s*:\s*(.+)$", line)
        if m:
            ver[m.group(1)] = m.group(2).strip()
    return ver


def write_version(ver):
    lines = [
        f"name: {ver['name']}",
        f"series: {ver['series']}",
        f"block: {ver['block']}",
        f"mk: {ver['mk']}",
        f"mod: {ver['mod']}",
        f"description: {ver.get('description', '')}",
    ]
    with open(VERSION_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → VERSION 已更新: MK{ver['mk']}-Mod{ver['mod']}")


def infer_bump(files):
    """根据改动的文件推断 MK++ / Mod++ / no-bump"""
    if not files:
        return "no-bump"

    # MK++ 条件: 新增/删除模块文件, 或 forward() 数据流重构
    mk_patterns = [
        r"^models/modules/.+",           # 模块文件新增/删除
        r"^models/tfs_net\.py$",          # 主模型重构
    ]
    mod_patterns = [
        r"^models/.+\.py$",               # 其他模型文件修改
        r"^configs/.+\.yaml$",            # 配置修改
        r"^train\.py$",                   # 训练脚本修改
    ]

    added_or_removed = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        capture_output=True, text=True, cwd=REPO,
    ).stdout.strip().split("\n")

    for line in added_or_removed:
        if line.startswith("A") or line.startswith("D"):
            fname = line.split("\t")[-1] if "\t" in line else ""
            for pat in mk_patterns:
                if re.match(pat, fname):
                    return "mk"

    # 检查 staged 中是否有 mk_patterns 匹配的文件
    for f in files:
        for pat in mk_patterns:
            if re.match(pat, f):
                return "mk"

    for f in files:
        for pat in mod_patterns:
            if re.match(pat, f):
                return "mod"

    return "no-bump"


def format_version(ver):
    return f"v6-{ver['name']}-{ver['series']}{ver['block']}-MK{ver['mk']}-Mod{ver['mod']}"


def main():
    staged, unstaged = get_git_diff_files()
    all_changed = staged | unstaged

    ver = read_version()

    force = None
    if "--mk" in sys.argv:
        force = "mk"
    elif "--mod" in sys.argv:
        force = "mod"
    elif "--no-bump" in sys.argv:
        force = "no-bump"

    bump = force or infer_bump(all_changed)
    ver["mk"] = int(ver.get("mk", 0))
    ver["mod"] = int(ver.get("mod", 0))

    if bump == "mk":
        ver["mk"] += 1
        ver["mod"] = 0
        print(f"  ◉ MK++ → MK{ver['mk']} (构型变更)")
    elif bump == "mod":
        ver["mod"] += 1
        print(f"  ◇ Mod++ → Mod{ver['mod']} (局部升级)")
    else:
        print(f"  ○ 版本不变 (文档/配置, 或未检测到模型改动)")

    write_version(ver)
    print(f"  → {format_version(ver)}")
    print()
    print(f"建议 commit message:")
    print(f"  v6-{ver['name']}-{ver['series']}{ver['block']}-MK{ver['mk']}-Mod{ver['mod']}")


if __name__ == "__main__":
    main()
