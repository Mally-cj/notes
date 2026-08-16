---
title: 本地 Markdown 发布测试
date: 2026-08-16
---

# 本地 Markdown 发布测试

这篇文章直接在本地以 Markdown 文件编写，没有通过飞书同步。

发布流程如下：

1. 在仓库中创建 Markdown 文件。
2. 使用 VuePress 在本地完成构建检查。
3. 将文件提交并推送到 GitHub 的 `main` 分支。
4. GitHub Actions 构建网站并部署到 `gh-pages` 分支。

如果在线上看到本页面，说明本地 Markdown 发布链路工作正常。
