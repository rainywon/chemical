import { createApp } from 'vue';
import './style.css';
import App from './App.vue';
import router from './router' // 导入路由配置
import axios from 'axios'
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

// 创建 Vue 应用实例
const app = createApp(App);

// 使用 ElementPlus 插件
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}
app.use(ElementPlus);
app.use(router)

// 配置 axios
app.config.globalProperties.$axios = axios;

// 将应用挂载到 DOM 节点上
app.mount('#app');