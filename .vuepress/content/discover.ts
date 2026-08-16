import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type {
  NavbarOptions,
  SidebarGroupOptions,
  SidebarOptions,
} from "@vuepress/theme-default";

export interface ContentArticle {
  filePath: string;
  name: string;
  route: string;
}

export interface ContentNode {
  absolutePath: string;
  articles: ContentArticle[];
  children: ContentNode[];
  name: string;
  relativePath: string;
  route: string;
}

export interface ContentCatalog {
  contentRoot: string;
  root: ContentNode;
}

const sourceRoot = fileURLToPath(new URL("../..", import.meta.url));
export const contentRoot = path.join(sourceRoot, "博客");

const compareNames = (left: string, right: string): number =>
  left.localeCompare(right, "zh-CN", { numeric: true, sensitivity: "base" });

const isVisibleDirectory = (entry: fs.Dirent): boolean =>
  entry.isDirectory() && !entry.name.startsWith(".") && !entry.name.startsWith("_");

const toRoutePath = (value: string): string =>
  value.split(path.sep).join("/").replaceAll(":", "_");

const directoryRoute = (relativePath: string): string => {
  const suffix = relativePath ? `${toRoutePath(relativePath)}/` : "";
  return `/博客/${suffix}`;
};

const articleRoute = (relativePath: string): string =>
  `/博客/${toRoutePath(relativePath).replace(/\.md$/i, ".html")}`;

const discoverNode = (absolutePath: string, relativePath: string): ContentNode => {
  const entries = fs.readdirSync(absolutePath, { withFileTypes: true });
  const articles = entries
    .filter(
      (entry) =>
        entry.isFile() &&
        entry.name.endsWith(".md") &&
        entry.name.toLowerCase() !== "readme.md",
    )
    .map((entry) => {
      const articleRelativePath = path.join(relativePath, entry.name);
      return {
        filePath: path.join(absolutePath, entry.name),
        name: entry.name.slice(0, -3),
        route: articleRoute(articleRelativePath),
      };
    })
    .sort((left, right) => compareNames(left.name, right.name));

  const children = entries
    .filter(isVisibleDirectory)
    .map((entry) => {
      const childRelativePath = path.join(relativePath, entry.name);
      return discoverNode(path.join(absolutePath, entry.name), childRelativePath);
    })
    .filter((node) => node.articles.length > 0 || node.children.length > 0)
    .sort((left, right) => compareNames(left.name, right.name));

  return {
    absolutePath,
    articles,
    children,
    name: relativePath ? path.basename(absolutePath) : "文章",
    relativePath,
    route: directoryRoute(relativePath),
  };
};

export const discoverContent = (): ContentCatalog => {
  if (!fs.existsSync(contentRoot)) {
    throw new Error(`博客内容目录不存在: ${contentRoot}`);
  }

  const root = discoverNode(contentRoot, "");
  const routes = new Set<string>();

  const assertUniqueRoutes = (node: ContentNode): void => {
    for (const route of [node.route, ...node.articles.map((article) => article.route)]) {
      if (routes.has(route)) {
        throw new Error(`检测到重复页面路径: ${route}`);
      }
      routes.add(route);
    }
    node.children.forEach(assertUniqueRoutes);
  };

  assertUniqueRoutes(root);
  return { contentRoot, root };
};

export const flattenDirectories = (root: ContentNode): ContentNode[] => [
  root,
  ...root.children.flatMap(flattenDirectories),
];

const toSidebarGroup = (node: ContentNode): SidebarGroupOptions => ({
  text: node.name,
  link: node.route,
  collapsible: node.relativePath.split(path.sep).length > 1,
  children: [
    ...node.children.map(toSidebarGroup),
    ...node.articles.map((article) => ({
      text: article.name,
      link: article.route,
    })),
  ],
});

export const createNavbar = (catalog: ContentCatalog): NavbarOptions => [
  ...catalog.root.children.map((topic) => ({
    text: topic.name,
    link: topic.route,
    activeMatch: `^${topic.route}`,
  })),
  {
    text: "关于作者",
    link: "/关于作者.html",
  },
];

export const createSidebar = (catalog: ContentCatalog): SidebarOptions => {
  const sidebar: Record<string, SidebarGroupOptions[] | "heading"> = {
    "/": "heading",
  };

  for (const topic of catalog.root.children) {
    sidebar[topic.route] = [toSidebarGroup(topic)];
  }

  return sidebar;
};
