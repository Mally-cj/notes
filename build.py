#!/usr/bin/env python3

from dataclasses import field, dataclass
import json
import os
import pathlib
import sys
import re
import traceback
import shutil

@dataclass
class WikiItem:
    name: str
    url: str

@dataclass
class NavItem:
    text: str
    link: str
    items: list["NavItem"]

@dataclass
class Config:
    max_depth: int = 10
    collapsible_depth: int = 2
    wiki: list[WikiItem] = field(default_factory=list)
    nav: list[NavItem] = field(default_factory=list)

class WikiGenerator:
    def __init__(self, root_dir: str, max_depth=10, collapsible_depth=2, nav: list[NavItem] = None):
        self.root_dir = root_dir
        # self.wiki_dir = os.path.join(root_dir, ".wiki")
        self.vuepress_dir = os.path.join(root_dir, ".vuepress")
        self.static_base_dir = os.path.join(self.vuepress_dir, "public", "wiki")

        self.roadmap_lists = []
        self.max_depth = max_depth
        self.collapsible_depth = collapsible_depth
        self.nav = [] if nav is None else nav

    def get_realtive_path_of_wiki(self, wiki_name, path: dict):
        base_path = self.get_wiki_dir(wiki_name)
        return str(pathlib.Path(os.path.join(self.root_dir, path)).relative_to(base_path))

    def _generate_roadmap_siderbar(self, wiki_name: str, data: dict, depth: int):
        res = []
        item_m = {}
        collapsible = depth >= self.collapsible_depth
        for item in data["md_files"]:
            # key = item["path"][:-3]
            # p = self.get_realtive_path_of_wiki(wiki_name, item["path"])
            if item["is_leaf"]:
                res.append(item["name"])
        for item in data["subdirs"]:
            children = item_m.setdefault(item["key"], {
                "text": item["name"],
                "collapsible": collapsible,
                "prefix": item["name"] + "/",
                "children": [""],
            })["children"]
            if depth >= self.max_depth:
                children.extend([
                    item["name"]
                    for item in item["md_files"]
                ])
            else:
                children.extend(self._generate_roadmap_siderbar(wiki_name, item, depth + 1))

        for key, value in item_m.items():
            res.append(value)
        return res

    def generate_roadmap_siderbar(self, wiki_name: str, data: dict):
        res = [""] + self._generate_roadmap_siderbar(wiki_name, data, 1)

        ts_code = f"export default {json.dumps(res, indent=4, ensure_ascii=False)};"

        # print(ts_code)
        file_path = os.path.join(self.vuepress_dir, 'sidebars', f"{wiki_name}Siderbar.ts")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(ts_code)
        self.roadmap_lists.append({
            "name": wiki_name,
            "key": wiki_name,
        })
        return res

    def generate_sidebar(self):
        import_str = ""
        items_str = ""
        for item in self.roadmap_lists:
            items_str += f"""
    "/{item["name"]}/": {item["key"]}Siderbar,
    """
            import_str += f"""import {item["key"]}Siderbar from "./sidebars/{item["key"]}Siderbar";\n"""
        data = f"""
import {{ SidebarOptions }} from "@vuepress/theme-default";

{import_str}
export default {{
{items_str}
    // 降级，默认根据文章标题渲染侧边栏
    '/': "heading" ,
}} as SidebarOptions;
    """
        file_path = os.path.join(self.vuepress_dir, "sidebar.ts")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)

    def generate_navbar(self):
        """
        生成navbar，从 self.nav 生成
        """
        ts_code = f'''import {{NavItem}} from "vuepress/config";

export default {json.dumps(self.nav, indent=4, ensure_ascii=False)} as NavItem[];'''
        file_path = os.path.join(self.vuepress_dir, 'navbar.ts')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(ts_code)

    def get_wiki_dir(self, wiki_name: str):
        return os.path.join(self.root_dir, wiki_name)

    def download_wiki(self, wiki_name: str, wiki_url: str):
        """
        下载wiki
        """
        shutil.rmtree(self.get_wiki_dir(wiki_name), ignore_errors=True)
        os.system(f"""./deploy/feishu2md dl --wiki -o output_directory "{wiki_url}" """)

    def generate_wiki(self, wiki_name: str, wiki_url: str):
        """
        生成一个wiki页面
        """
        wiki_dir = self.get_wiki_dir(wiki_name)

        self.download_wiki(wiki_name, wiki_url)
        self.move_static_directory(wiki_name)

        # self.change_directory_name(wiki_dir)

        data = self.generate_directory_listing(wiki_name, wiki_dir, 0)
        print(data)
        self.generate_roadmap_siderbar(wiki_name, data)



    def move_static_directory(self, wiki_name: str):
        """
        遍历所有子目录的 static 目录，将其按原本的结构拷贝到 .vuepress/public 下，
        注意从最深层开始处理，避免因目录嵌套导致覆盖问题
        """
        # 目标根目录：.vuepress/public/wiki/<wiki_name>
        static_base_dir = os.path.join(self.static_base_dir, wiki_name)
        os.makedirs(static_base_dir, exist_ok=True)
        source_wiki_dir = self.get_wiki_dir(wiki_name)

        # 从最深处开始遍历
        for root, dirs, files in os.walk(source_wiki_dir, topdown=False):
            if "static" in dirs:
                source_static_dir = os.path.join(root, "static")
                # 计算 source_static_dir 相对于 source_wiki_dir 的路径，
                # 这样可以在目标目录中还原原有的层级结构
                rel_path = os.path.relpath(root, start=source_wiki_dir)
                target_static_dir = os.path.join(static_base_dir, rel_path, "static")
                os.makedirs(os.path.dirname(target_static_dir), exist_ok=True)
                if os.path.exists(target_static_dir):
                    # 如果目标目录存在，先删除
                    shutil.rmtree(target_static_dir)
                os.rename(source_static_dir, target_static_dir)

    def get_realname_dir_name(self: str, directory: str):
        """
        Get the name of the directory
        """
        directory_name = os.path.basename(directory)
        return directory_name

    def change_directory_name(self, directory: str):
        """
        遍历所有的目录，将所有的目录名加一个.dir后缀
        """
        # 先收集所有需要重命名的目录，并按路径深度排序（从深到浅）
        dirs_to_rename = []
        for root, dirs, files in os.walk(directory):
            for dir_name in dirs:
                if dir_name.endswith(".dir") or dir_name == "static":
                    continue
                full_path = os.path.join(root, dir_name)
                # 计算路径深度
                depth = full_path.count(os.sep)
                dirs_to_rename.append((depth, root, dir_name))

        # 按深度降序排序，确保先处理深层目录
        dirs_to_rename.sort(reverse=True, key=lambda x: x[0])

        # 然后统一重命名
        for depth, root, dir_name in dirs_to_rename:
            old_dir_path = os.path.join(root, dir_name)
            new_dir_path = os.path.join(root, dir_name + ".dir")
            print(f"  Renaming: {old_dir_path} -> {new_dir_path}")  # 调试输出
            try:
                os.rename(old_dir_path, new_dir_path)
            except FileNotFoundError as e:
                print(f"  Warning: Failed to rename {old_dir_path}: {e}")

    def generate_directory_listing(self, wiki_name: str, directory: str, depth: int):
        """
        Generate directory listings for README.md files in the given directory and its subdirectories.
        """
        dir_name = self.get_realname_dir_name(directory)
        try:
            # Skip hidden directories
            if os.path.basename(directory).startswith('.'):
                return

            print(f"Processing directory: {directory}")

            subdirs = []

            # 如果某个文档有子目录，将这个文档移动到子目录中，并命名为README.md
            # Get directory content
            try:
                dir_content = os.listdir(directory)
            except Exception as e:
                print(f"  Error listing directory contents: {e}")
                return
            for item in dir_content:
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    if os.path.exists(item_path + ".md"):
                        os.rename(item_path + ".md", os.path.join(item_path, "README.md"))
                    subdirs.append(item)

            readme_path = os.path.join(directory, 'README.md')

            # Read existing README content if it exists
            existing_content = ""
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    print(f"  Found existing README.md: {len(existing_content)} bytes")

                    # Remove any existing directory listing section if it exists (modified regex to capture optional newline)
                    pattern = r"(?:\n)?## 目录内容[\s\S]*?(?=\n##|\Z)"
                    existing_content = re.sub(pattern, "", existing_content)
                    existing_content = existing_content.rstrip() + "\n\n"
                except Exception as e:
                    print(f"  Error reading README.md: {e}")
                    # Create a simple header if we can't read the file
                    dir_name = self.get_realname_dir_name(directory)
                    existing_content = f"# {dir_name}\n\n"
            else:
                # If README doesn't exist, create basic header
                dir_name = self.get_realname_dir_name(directory)
                existing_content = f"# {dir_name}\n\n"
                print(f"  No README.md found, will create one with title: {dir_name}")

            # Start building directory listing
            content_listing = f""
            # List subdirectories

            # if subdirs:
            #     content_listing += "### 子目录\n\n"
            #     for subdir in sorted(subdirs):
            #         content_listing += f"- [{subdir}](./{subdir}/)\n"
            #     content_listing += "\n"
            #     print(f"  Found {len(subdirs)} subdirectories")

            # List markdown files
            md_files = []
            for item in dir_content:
                if item.endswith('.md') and item != 'README.md':
                    md_files.append(item)

            if md_files:
                content_listing += "## 目录\n\n"
                for md_file in sorted(md_files):
                    title = os.path.splitext(md_file)[0]
                    content_listing += f"- [{title}](./{md_file})\n"
                content_listing += "\n"
                print(f"  Found {len(md_files)} markdown files")

            # If no items found, add a default message.
            # if not (subdirs or md_files):
            #     content_listing += "暂无内容\n\n"

            # Always update README.md file
            try:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    final_content = existing_content + content_listing
                    f.write(final_content)
                print(f"  Updated {readme_path} ({len(final_content)} bytes written)")
            except Exception as e:
                print(f"  Error writing to README.md: {e}")

            res = {
                "subdirs": [],
                "key": directory,
                "name": os.path.basename(directory),
                "md_files": [{"path": os.path.join(directory, md_file), "name": md_file[:-3], "is_leaf": md_file[:-3] not in subdirs } for md_file in md_files],
            }
            # Process subdirectories recursively
            for subdir in subdirs:
                subres = self.generate_directory_listing(wiki_name, os.path.join(directory, subdir), depth=1)
                res["subdirs"].append(subres)

            return res
        except Exception as e:
            print(f"Error processing directory {directory}: {e}")
            traceback.print_exc()

def read_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config = Config(**config)
    return config

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "."
    config = read_config()
    generator = WikiGenerator(root_dir,
        max_depth=config.max_depth,
        collapsible_depth=config.collapsible_depth,
        nav=config.nav)
    for item in config.wiki:
        wiki_name, wiki_url = item["name"], item["url"]
        generator.generate_wiki(wiki_name, wiki_url)
    generator.generate_sidebar()
    generator.generate_navbar()
