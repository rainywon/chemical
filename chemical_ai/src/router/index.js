import { createRouter, createWebHistory } from 'vue-router';
import Login from '../views/Login.vue'; // 导入你的当前页面组件
import Chat from '../views/Chat.vue';
import Register from '../views/Register.vue'
import Welcome from '../views/Welcome.vue'; // 导入新的欢迎页组件
import SaftyFiles from '@/views/SaftyFiles.vue'
import EmergencyFiles from '@/views/EmergencyFiles.vue'

// 导入管理系统布局组件
import AdminLayout from '../views/admin/AdminLayout.vue';
import AdminDashboard from '../views/admin/AdminDashboard.vue';

const routes = [
    {
        path: '/', // 根路径设置为欢迎页
        name: 'welcome',
        component: Welcome, // 使用欢迎页作为真正的首页
    },
    {
        path: '/chat',
        name: 'chat',
        component: Chat,
    },
    {
        path: '/login',
        name: 'Login',
        component: Login,
    },
    {
        path: '/register',
        name: 'Register',
        component: Register,
    },
    {
        path: '/files',
        name: 'Files',
        component: SaftyFiles,
        meta: {
            title: '安全资料库 - 化工安全AI助手'
        }
    },
    {
        path: '/emergency_files',
        name: 'EmergencyFiles',
        component: EmergencyFiles,
        meta: {
            title: '应急处理资料库 - 化工安全AI助手'
        }
    },
    // 管理系统路由
    {
        path: '/admin',
        component: AdminLayout,
        redirect: '/admin/dashboard',
        children: [
            {
                path: 'dashboard',
                name: 'AdminDashboard',
                component: AdminDashboard,
                meta: { title: '管理系统控制台', requiresAdmin: true }
            },
            // 系统监控
            {
                path: 'monitor/conversation',
                name: 'ConversationStats',
                component: () => import('../components/admin/monitor/ConversationStats.vue'),
                meta: { title: '对话数据统计', requiresAdmin: true }
            },
            {
                path: 'monitor/users',
                name: 'UserActivity',
                component: () => import('../components/admin/monitor/UserActivity.vue'),
                meta: { title: '用户活跃度监控', requiresAdmin: true }
            },
            // 用户管理
            {
                path: 'users/user-management',
                name: 'UserManagement',
                component: () => import('../components/admin/users/UserManagement.vue'), // 重用UserList组件或创建新的组件
                meta: { title: '用户查询与管理', requiresAdmin: true }
            },
            {
                path: 'users/login-history',
                name: 'LoginHistory',
                component: () => import('../components/admin/users/LoginHistory.vue'),
                meta: { title: '用户登录历史', requiresAdmin: true }
            },
            // 管理员管理
            {
                path: 'admins/admin-management',
                name: 'AdminManagement',
                component: () => import('../components/admin/admins/AdminManagement.vue'),
                meta: { title: '管理员与角色管理', requiresAdmin: true },
                alias: ['admins/list']
            },
            {
                path: 'admins/logs',
                name: 'OperationLogs',
                component: () => import('../components/admin/admins/OperationLogs.vue'),
                meta: { title: '管理员操作日志', requiresAdmin: true }
            },
            // 内容管理
            {
                path: 'content/knowledge-files',
                name: 'CategoryManager',
                component: () => import('../components/admin/content/CategoryManager.vue'),
                meta: { title: '知识库文件管理', requiresAdmin: true }
            },
            {
                path: 'content/safety-materials',
                name: 'DocumentManager',
                component: () => import('../components/admin/content/DocumentManager.vue'),
                meta: { title: '安全资料库', requiresAdmin: true }
            },
            {
                path: 'content/emergency-materials',
                name: 'EmergencyPlanManager',
                component: () => import('../components/admin/content/EmergencyPlanManager.vue'),
                meta: { title: '应急处理资料库', requiresAdmin: true }
            },
            // 反馈管理
            {
                path: 'feedback/list',
                name: 'FeedbackList',
                component: () => import('../components/admin/feedback/SystemFunctionFeedback.vue'),
                meta: { title: '系统功能反馈', requiresAdmin: true }
            },
            {
                path: 'feedback/ratings',
                name: 'ContentRating',
                component: () => import('../components/admin/feedback/AIContentFeedback.vue'),
                meta: { title: '生成内容反馈', requiresAdmin: true }
            },
            {
                path: 'settings/system-params',
                name: 'SystemParams',
                component: () => import('../components/admin/settings/SystemParams.vue'),
                meta: { title: '系统参数设置', requiresAdmin: true }
            }
        ]
    },
    // 可以继续添加其他路由...
];

const router = createRouter({
    history: createWebHistory(),
    routes,
});



export default router;