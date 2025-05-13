import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'; // 导入 Node.js 的 path 模块
import { prismjsPlugin } from 'vite-plugin-prismjs'
// https://vite.dev/config/
export default defineConfig({
    plugins: [vue(),
        prismjsPlugin({
            languages: 'all', // 语言
            plugins: ['line-numbers', 'show-language', 'copy-to-clipboard', 'inline-color'],
            theme: 'okaidia', // 主题
            css: true,
        })
    ],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, 'src'), // 设置 @ 别名指向 src 目录
        },
    },
})