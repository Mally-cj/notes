# 项目说明

这是一个只以本地 Markdown 为内容源的 VuePress 博客。发布链路为：

```text
本地 Markdown -> main 分支 -> GitHub Actions -> gh-pages -> notes.mally.cc
```

不要重新引入飞书同步、在线文档导出器或第二套内容源。

## 内容位置

- `README.md`：网站首页。
- `博客/README.md`：文章统一入口和分类索引。
- `博客/`：后续本地新写的文章。
- `AIGC相关/`：从飞书一次性迁移的历史文章；保留现有路径，避免旧 URL 失效。
- `.vuepress/public/images/`：新文章的图片，正文使用 `/images/...` 引用。
- `.vuepress/public/wiki/AIGC相关/`：历史文章图片，不移动或改名。
- `.vuepress/navbar.ts`：顶部导航。
- `.vuepress/sidebar.ts` 和 `.vuepress/sidebars/`：侧边栏配置。

## 本地验证

```bash
npm ci
npm run docs:dev
npm run docs:build
```

提交前至少运行 `npm run docs:build`，确认没有断链警告。不要提交 `node_modules/`、`.vuepress/.cache/`、`.vuepress/.temp/` 或 `.vuepress/dist/`。

## 发布

推送 `main` 后，`.github/workflows/deploy-docs.yml` 自动构建并发布到 `gh-pages`。不要直接编辑 `gh-pages` 分支，也不要提交构建产物。
