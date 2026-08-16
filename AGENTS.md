# 项目说明

这是一个以仓库内 Markdown 为内容源的 VuePress 博客。发布链路为：

```text
Markdown -> main 分支 -> GitHub Actions -> gh-pages -> notes.mally.cc
```

文章只按主题分类，不按写作工具或内容来源分类。

## 内容位置

- `README.md`：网站首页。
- `博客/<主题>/<分类>/`：文章内容；一级主题自动生成顶部导航，子目录自动生成目录页和侧边栏。
- `博客/<主题>/<分类>/_assets/`：分类内文章图片，正文使用相对路径引用。
- `.vuepress/content/discover.ts`：扫描内容目录，生成导航和侧边栏数据。
- `.vuepress/content/catalogPlugin.ts`：生成虚拟目录页、旧 URL 跳转和旧图片兼容路径。

不要手工创建目录 `README.md`，也不要维护静态 navbar 或 sidebar 文件。新增、移动或删除文章后，如果开发服务器已启动，需要重启一次以重新扫描目录结构；编辑已有文章可以正常热更新。

## 构建验证

```bash
npm ci
npm run docs:dev
npm run docs:build
```

提交前至少运行 `npm run docs:build`，确认没有断链警告。不要提交 `node_modules/`、`.vuepress/.cache/`、`.vuepress/.temp/` 或 `.vuepress/dist/`。

## 发布

推送 `main` 后，`.github/workflows/deploy-docs.yml` 自动构建并发布到 `gh-pages`。不要直接编辑 `gh-pages` 分支，也不要提交构建产物。
