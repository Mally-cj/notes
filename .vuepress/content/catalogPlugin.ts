import fs from "node:fs";
import path from "node:path";

import { createPage, type App, type Plugin } from "@vuepress/core";

import {
  flattenDirectories,
  type ContentCatalog,
  type ContentNode,
} from "./discover";

const legacyTopicPrefix = "/AIGC相关/";
const currentTopicPrefix = "/博客/文本大模型相关/";

const encodeLink = (route: string): string => encodeURI(route);

const escapeMarkdown = (value: string): string =>
  value.replace(/([\\[\]])/g, "\\$1");

const renderCatalog = (node: ContentNode): string => {
  const sections: string[] = [`# ${node.name}`, ""];

  if (node.children.length > 0) {
    sections.push("## 分类", "");
    for (const child of node.children) {
      sections.push(`- [${escapeMarkdown(child.name)}](${encodeLink(child.route)})`);
    }
    sections.push("");
  }

  if (node.articles.length > 0) {
    sections.push("## 文章", "");
    for (const article of node.articles) {
      sections.push(`- [${escapeMarkdown(article.name)}](${encodeLink(article.route)})`);
    }
    sections.push("");
  }

  return sections.join("\n");
};

const escapeHtml = (value: string): string =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

const outputPathForRoute = (destination: string, route: string): string => {
  const decodedRoute = decodeURI(route).replace(/^\//, "");
  return route.endsWith("/")
    ? path.join(destination, decodedRoute, "index.html")
    : path.join(destination, decodedRoute);
};

const writeRedirect = (destination: string, from: string, to: string): void => {
  const outputPath = outputPathForRoute(destination, from);
  const encodedTarget = encodeURI(decodeURI(to));
  const safeTarget = escapeHtml(encodedTarget);
  const html = `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=${safeTarget}">
    <link rel="canonical" href="${safeTarget}">
    <title>页面已移动</title>
  </head>
  <body>
    <script>location.replace(${JSON.stringify(encodedTarget)})</script>
    <a href="${safeTarget}">页面已移动</a>
  </body>
</html>
`;

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, html, "utf8");
};

const copyLegacyAssets = (catalog: ContentCatalog, app: App): void => {
  const topic = catalog.root.children.find((item) => item.name === "文本大模型相关");
  if (!topic) return;

  for (const node of flattenDirectories(topic)) {
    const assetsPath = path.join(node.absolutePath, "_assets");
    if (!fs.existsSync(assetsPath)) continue;

    const relativePath = path.relative(topic.absolutePath, node.absolutePath);
    for (const prefix of ["images", "wiki"]) {
      const legacyPath = path.join(
        app.dir.dest(),
        prefix,
        "AIGC相关",
        relativePath,
        "static",
      );
      fs.cpSync(assetsPath, legacyPath, { recursive: true });
    }
  }
};

export const contentCatalogPlugin = (catalog: ContentCatalog): Plugin => ({
  name: "content-catalog",

  async onInitialized(app) {
    const existingRoutes = new Set(app.pages.map((page) => page.path));

    for (const node of flattenDirectories(catalog.root)) {
      if (existingRoutes.has(node.route)) continue;
      app.pages.push(
        await createPage(app, {
          path: node.route,
          content: renderCatalog(node),
          frontmatter: { title: node.name },
        }),
      );
    }
  },

  onGenerated(app) {
    for (const page of app.pages) {
      const decodedPath = decodeURI(page.path);
      if (!decodedPath.startsWith(currentTopicPrefix)) continue;
      const legacyPath = decodedPath.replace(currentTopicPrefix, legacyTopicPrefix);
      writeRedirect(app.dir.dest(), legacyPath, page.path);
    }

    copyLegacyAssets(catalog, app);
  },
});
