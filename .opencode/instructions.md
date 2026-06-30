# TFS-Net 版本命名规范

## 层级

```
v6-[Name]-[Series][Block][-MK][-Mod][-suffix]
```

规则: **MK0 和 Mod0 省略**, 两者都为 0 时仅保留 Name-Series

示例: `v6-pureRWKV-Delta1` (MK=0,Mod=0) / `v6-pureRWKV-Charlie1-MK4-Mod0` / `v6-pureRWKV-Delta1-pic`

| 层级 | 说明 | 由谁控制 |
|------|------|----------|
| **Name** | 核心构型名 (如 pureRWKV) | 用户指定 |
| **Series+Block** | 系列字母+批次 (如 Bravo1, Charlie1) | 用户指定 |
| **MK** | 模型大代际基础构型, 自动递增 | 自动 |
| **Mod** | 小幅局部升级, 自动递增 | 自动 |
| **suffix** | 可选标记 (code/pic/docs/train) | 用户指定 |

## 自动递增规则

每次 commit 前必须:

1. 读取 `docs/v6/VERSION` 获取当前版本
2. 分析 git diff (staged + unstaged)
3. 执行 `python3 scripts/bump_version.py [--mk | --mod | --no-bump]`

规则:

| 条件 | 动作 |
|------|------|
| 新增/删除 `models/modules/*` 文件, 或 `models/tfs_net.py` 数据流重构 | `--mk` (MK++, Mod 归零) |
| 已有模块内参数/功能调整, config 修改, train.py 修改 | `--mod` (Mod++) |
| 仅文档/图片/README 修改 | `--no-bump` |

## 示例

```
当前 VERSION: name=pureRWKV series=Charlie block=1 mk=3 mod=2

改动: mrpn.py 新增 blur_estimator → --mk
→ 下一版: v6-pureRWKV-Charlie1-MK4 (Mod0 省略)

改动: 调整学习率参数 → --mod
→ 下一版: v6-pureRWKV-Charlie1-MK3-Mod3
```

## commit 流程

```
1. git add <files>
2. python3 scripts/bump_version.py  # 自动检测并递增
3. git commit -m "v6-pureRWKV-Charlie1-MK4: 描述"
4. git push origin master
```

## 当前版本锚点

文件: `docs/v6/VERSION`
