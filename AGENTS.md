# 项目说明

这是一个以仓库内 Markdown 为内容源的 VuePress 博客。发布链路为：

```text
Markdown -> main 分支 -> GitHub Actions -> gh-pages -> notes.mally.cc
```

文章只按主题分类，不按写作工具或内容来源分类。

## 内容位置

- `README.md`：网站首页。
- `博客/README.md`：文章统一入口和分类索引。
- `AIGC相关/`：AIGC 主题文章；同主题的新文章继续写入对应子目录。
- `<主题>/`：其他主题文章；新增主题时在仓库根目录创建分类，并提供 `README.md` 索引。
- `.vuepress/public/images/<主题>/`：文章图片，正文使用 `/images/<主题>/...` 引用。
- `.vuepress/navbar.ts`：顶部导航。
- `.vuepress/sidebar.ts` 和 `.vuepress/sidebars/`：侧边栏配置。

## 构建验证

```bash
npm ci
npm run docs:dev
npm run docs:build
```

提交前至少运行 `npm run docs:build`，确认没有断链警告。不要提交 `node_modules/`、`.vuepress/.cache/`、`.vuepress/.temp/` 或 `.vuepress/dist/`。

## 发布

推送 `main` 后，`.github/workflows/deploy-docs.yml` 自动构建并发布到 `gh-pages`。不要直接编辑 `gh-pages` 分支，也不要提交构建产物。
