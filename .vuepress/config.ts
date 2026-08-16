import { viteBundler } from "@vuepress/bundler-vite";
import { markdownMathPlugin } from "@vuepress/plugin-markdown-math";
import { defaultTheme } from "@vuepress/theme-default";
import { defineUserConfig } from "vuepress";

import { contentCatalogPlugin } from "./content/catalogPlugin";
import {
  createNavbar,
  createSidebar,
  discoverContent,
} from "./content/discover";

const catalog = discoverContent();

export default defineUserConfig({
  bundler: viteBundler(),
  plugins: [markdownMathPlugin(), contentCatalogPlugin(catalog)],
  markdown: {
    toc: {},
  },
  theme: defaultTheme({
    navbar: createNavbar(catalog),
    sidebar: createSidebar(catalog),
    logo: "/logo.jpg",
  }),
  lang: "zh-CN",
  title: "mally的技术笔记",
  description: "记录 AI 技术、科研学习与工程实践",
  head: [
    ["link", { rel: "icon", href: "/favicon.ico" }],
    [
      "meta",
      {
        name: "keywords",
        content: "mally, 技术笔记, 开发, 编程分享",
      },
    ],
  ],
});
