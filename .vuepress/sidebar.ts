
import { SidebarOptions } from "@vuepress/theme-default";

import AIGC相关Siderbar from "./sidebars/AIGC相关Siderbar";

export default {

    "/AIGC相关/": AIGC相关Siderbar,
    
    // 降级，默认根据文章标题渲染侧边栏
    '/': "heading" ,
} as SidebarOptions;
    