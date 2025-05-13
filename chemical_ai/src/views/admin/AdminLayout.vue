<template>
  <div class="admin-layout" :class="{ 'dark-mode': isDarkMode }">
    <!-- 侧边栏 -->
    <div class="sidebar" :class="{ 'collapsed': isCollapsed }">
      <div class="logo">
        <img src="../../assets/product.png" alt="Logo" />
        <span v-if="!isCollapsed">天工AI管理系统</span>
      </div>
      
      <!-- 折叠按钮 -->
      <div class="collapse-btn" @click="toggleCollapse">
        <el-icon v-if="isCollapsed"><Expand /></el-icon>
        <el-icon v-else><Fold /></el-icon>
      </div>
      
      <!-- 导航菜单 -->
      <el-menu
        :default-active="activeMenu"
        class="sidebar-menu"
        :background-color="isDarkMode ? '#1a1a1a' : '#001529'"
        :text-color="isDarkMode ? '#e0e0e0' : '#fff'"
        :active-text-color="isDarkMode ? '#66b1ff' : '#409EFF'"
        :collapse="isCollapsed"
        router
      >
        <!-- 系统监控 -->
        <el-sub-menu index="/admin/monitor">
          <template #title>
            <el-icon><Monitor /></el-icon>
            <span>系统监控</span>
          </template>
          <el-menu-item index="/admin/monitor/conversation">对话数据统计</el-menu-item>
          <el-menu-item index="/admin/monitor/users">用户活跃度监控</el-menu-item>
        </el-sub-menu>
        
        <!-- 用户管理 -->
        <el-sub-menu index="/admin/users">
          <template #title>
            <el-icon><User /></el-icon>
            <span>用户管理</span>
          </template>
          <el-menu-item index="/admin/users/user-management">用户查询与管理</el-menu-item>
          <el-menu-item index="/admin/users/login-history">用户登录历史</el-menu-item>
        </el-sub-menu>
        
        <!-- 管理员管理 -->
        <el-sub-menu index="/admin/admins">
          <template #title>
            <el-icon><Setting /></el-icon>
            <span>管理员管理</span>
          </template>
          <el-menu-item index="/admin/admins/admin-management">管理员与角色管理</el-menu-item>
          <el-menu-item index="/admin/admins/logs">管理员操作日志</el-menu-item>
        </el-sub-menu>
        
        <!-- 内容管理 -->
        <el-sub-menu index="/admin/content">
          <template #title>
            <el-icon><Document /></el-icon>
            <span>内容管理</span>
          </template>
          <el-menu-item index="/admin/content/knowledge-files">知识库文件管理</el-menu-item>
          <el-menu-item index="/admin/content/safety-materials">安全资料库</el-menu-item>
          <el-menu-item index="/admin/content/emergency-materials">事故案例库</el-menu-item>
        </el-sub-menu>
        
        <!-- 反馈管理 -->
        <el-sub-menu index="/admin/feedback">
          <template #title>
            <el-icon><Comment /></el-icon>
            <span>反馈管理</span>
          </template>
          <el-menu-item index="/admin/feedback/list">系统功能反馈</el-menu-item>
          <el-menu-item index="/admin/feedback/ratings">生成内容反馈</el-menu-item>
        </el-sub-menu>
        
        <!-- 系统设置 -->
        <el-sub-menu index="/admin/settings">
          <template #title>
            <el-icon><Tools /></el-icon>
            <span>系统设置</span>
          </template>
          <el-menu-item index="/admin/settings/system-params">系统参数设置</el-menu-item>
        </el-sub-menu>
        
        <!-- 返回前台 -->
        <el-menu-item index="/">
          <el-icon><Back /></el-icon>
          <span>返回前台</span>
        </el-menu-item>
      </el-menu>
    </div>
    
    <!-- 内容区域 -->
    <div class="content" :class="{ 'collapsed-content': isCollapsed }">
      <div class="header" :class="{ 'dark-header': isDarkMode }">
        <div class="header-left">
          <div class="breadcrumb">
            <el-breadcrumb separator="/">
              <el-breadcrumb-item :to="{ path: '/admin' }">首页</el-breadcrumb-item>
              <el-breadcrumb-item v-for="(item, index) in breadcrumbs" :key="index">
                {{ item }}
              </el-breadcrumb-item>
            </el-breadcrumb>
          </div>
        </div>
        
        <div class="header-right">
          
          <!-- 全屏切换按钮 -->
          <div class="fullscreen-toggle" @click="toggleFullScreen">
            <el-tooltip :content="isFullscreen ? '退出全屏' : '全屏显示'" placement="bottom">
              <el-icon size="20"><FullScreen v-if="!isFullscreen" /><Aim v-else /></el-icon>
            </el-tooltip>
          </div>
          

          
          <!-- 用户信息 -->
          <div class="user-info">
            <el-dropdown trigger="click">
              <div class="user-avatar-info">
                <el-avatar :size="32" src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png" />
                <span class="dropdown-link">
                  {{ adminInfo.full_name || '管理员' }}
                  <el-icon class="el-icon--right"><arrow-down /></el-icon>
                </span>
              </div>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item @click="showProfileDialog">
                    <el-icon><UserFilled /></el-icon>个人资料
                  </el-dropdown-item>
                  <el-dropdown-item @click="showPasswordDialog">
                    <el-icon><Lock /></el-icon>修改密码
                  </el-dropdown-item>
                  <el-dropdown-item divided @click="handleLogout">
                    <el-icon><SwitchButton /></el-icon>退出登录
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </div>
      </div>
      
      <div class="main-content">
        <router-view />
      </div>
    </div>
    
    <!-- 个人资料对话框 -->
    <el-dialog v-model="profileDialogVisible" title="个人资料" width="550px" destroy-on-close custom-class="profile-dialog">
      <el-skeleton :rows="4" animated :loading="!adminInfo.admin_id">
        <template #default>
          <el-card class="profile-card" shadow="never">
            <template #header>
              <div class="profile-header">
                <el-avatar :size="80" src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png" />
                <div class="profile-role">
                  <el-tag type="success" v-if="adminInfo.role === 'admin'" effect="dark">管理员</el-tag>
                  <el-tag v-else effect="dark">操作员</el-tag>
                </div>
              </div>
            </template>

            <el-form :model="profileForm" label-width="80px" :rules="profileRules" ref="profileFormRef" class="profile-form">
              <el-form-item label="手机号" prop="phone_number">
                <el-input v-model="profileForm.phone_number" placeholder="请输入手机号">
                  <template #prefix><el-icon><Phone /></el-icon></template>
                </el-input>
              </el-form-item>
              <el-form-item label="姓名" prop="full_name">
                <el-input v-model="profileForm.full_name" placeholder="请输入姓名">
                  <template #prefix><el-icon><UserFilled /></el-icon></template>
                </el-input>
              </el-form-item>
              <el-form-item label="邮箱" prop="email">
                <el-input v-model="profileForm.email" placeholder="请输入邮箱">
                  <template #prefix><el-icon><Message /></el-icon></template>
                </el-input>
              </el-form-item>
              <el-form-item label="创建时间" v-if="adminInfo.created_at">
                <el-input v-model="adminInfo.created_at" disabled>
                  <template #prefix><el-icon><Calendar /></el-icon></template>
                </el-input>
              </el-form-item>
            </el-form>
          </el-card>
        </template>
      </el-skeleton>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="profileDialogVisible = false">取消</el-button>
          <el-button type="primary" @click="updateProfile" :loading="loading" :disabled="!adminInfo.admin_id">
            <el-icon><Check /></el-icon> 保存
          </el-button>
        </span>
      </template>
    </el-dialog>
    
    <!-- 修改密码对话框 -->
    <el-dialog v-model="passwordDialogVisible" title="修改密码" width="550px" destroy-on-close custom-class="password-dialog">
      <el-card class="password-card" shadow="never">
        <template #header>
          <div class="password-header">
            <el-icon><Key /></el-icon>
            <span>安全设置</span>
          </div>
        </template>
        
        <el-form :model="passwordForm" label-width="100px" :rules="passwordRules" ref="passwordFormRef" class="password-form">
          <el-form-item label="旧密码" prop="old_password">
            <el-input v-model="passwordForm.old_password" type="password" show-password placeholder="请输入旧密码">
              <template #prefix><el-icon><Lock /></el-icon></template>
            </el-input>
          </el-form-item>
          <el-form-item label="新密码" prop="new_password">
            <el-input v-model="passwordForm.new_password" type="password" show-password placeholder="请输入新密码">
              <template #prefix><el-icon><Edit /></el-icon></template>
            </el-input>
            <div class="form-tip">密码至少需要6个字符</div>
          </el-form-item>
          <el-form-item label="确认新密码" prop="confirm_password">
            <el-input v-model="passwordForm.confirm_password" type="password" show-password placeholder="请再次输入新密码">
              <template #prefix><el-icon><Check /></el-icon></template>
            </el-input>
          </el-form-item>
        </el-form>
      </el-card>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="passwordDialogVisible = false">取消</el-button>
          <el-button type="primary" @click="updatePassword" :loading="loading">
            <el-icon><Check /></el-icon> 确认修改
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, onUnmounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import { API_BASE_URL } from '../../config';
import { ElMessage, ElMessageBox } from 'element-plus';
import { 
  Monitor, User, Setting, Document, Comment, Tools, Back, ArrowDown, 
  Fold, Expand, Search, Bell, Sunny, Moon, UserFilled, Lock, SwitchButton, FullScreen, Aim, Phone, Message, Calendar, Key, Edit, Check
} from '@element-plus/icons-vue';

const route = useRoute();
const router = useRouter();
const activeMenu = ref('');
const breadcrumbs = ref([]);
const isCollapsed = ref(false);
const isDarkMode = ref(false);
const searchQuery = ref('');
const isFullscreen = ref(false);
const profileDialogVisible = ref(false);
const passwordDialogVisible = ref(false);
const profileForm = ref({
  full_name: '',
  email: '',
  phone_number: ''
});
const passwordForm = ref({
  old_password: '',
  new_password: '',
  confirm_password: ''
});
const profileRules = ref({
  full_name: [{ required: true, message: '请输入姓名', trigger: 'blur' }],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '请输入正确的邮箱格式', trigger: 'blur' }
  ],
  phone_number: [
    { required: true, message: '请输入手机号', trigger: 'blur' },
    { pattern: /^1[3-9]\d{9}$/, message: '请输入正确的手机号格式', trigger: 'blur' }
  ]
});
const passwordRules = ref({
  old_password: [{ required: true, message: '请输入旧密码', trigger: 'blur' }],
  new_password: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 6, message: '密码长度至少6个字符', trigger: 'blur' }
  ],
  confirm_password: [
    { required: true, message: '请确认新密码', trigger: 'blur' },
    {
      validator: (rule, value, callback) => {
        if (value !== passwordForm.value.new_password) {
          callback(new Error('两次输入的密码不一致'));
        } else {
          callback();
        }
      },
      trigger: 'blur'
    }
  ]
});
const loading = ref(false);
const adminInfo = ref({});
const profileFormRef = ref(null);
const passwordFormRef = ref(null);

// 获取管理员信息
const fetchAdminInfo = async () => {
  try {
    const adminToken = localStorage.getItem('token');
    if (!adminToken) {
      ElMessage.warning('未识别出管理员，请重新登录');
      router.push('/admin/login');
      return;
    }
    
    const response = await axios.get(`${API_BASE_URL}/admin/profile`, {
      headers: {
        'Authorization': `Bearer ${adminToken}`
      }
    });
    
    if (response.data.code === 200) {
      adminInfo.value = response.data.data;
      // 初始化表单数据
      profileForm.value = {
        admin_id: adminInfo.value.admin_id,
        full_name: adminInfo.value.full_name || '',
        email: adminInfo.value.email || '',
        phone_number: adminInfo.value.phone_number || ''
      };
    } else {
      ElMessage.error(response.data.message || '获取管理员信息失败');
      if (response.data.code === 404) {
        router.push('/admin/login');
      }
    }
  } catch (error) {
    console.error('获取管理员信息失败:', error);
    ElMessage.error('获取管理员信息请求失败，请检查网络连接');
  }
};

// 切换侧边栏折叠状态
const toggleCollapse = () => {
  isCollapsed.value = !isCollapsed.value;
  localStorage.setItem('adminSidebarCollapsed', isCollapsed.value.toString());
};



// 切换全屏状态
const toggleFullScreen = () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(err => {
      console.error(`Error attempting to enable full-screen mode: ${err.message}`);
    });
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }
};

// 搜索处理
const handleSearch = () => {
  if (searchQuery.value.trim()) {
    // 这里可以实现搜索逻辑
  }
};

// 生命周期钩子
onMounted(() => {
  updateActiveMenu();
  
  // 从本地存储中恢复设置
  const savedCollapsed = localStorage.getItem('adminSidebarCollapsed');
  
  if (savedCollapsed) {
    isCollapsed.value = savedCollapsed === 'true';
  }
  
  // 获取管理员Token (登录成功后存储在localStorage中)
  const adminToken = localStorage.getItem('token');
  if (adminToken) {
    // 立即获取管理员信息
    fetchAdminInfo();
  } else {
    ElMessage.warning('请先登录');
    router.push('/admin/login');
  }
  
  // 添加全屏变化事件监听
  document.addEventListener('fullscreenchange', updateFullscreenStatus);
});

// 组件卸载时移除事件监听
onUnmounted(() => {
  document.removeEventListener('fullscreenchange', updateFullscreenStatus);
});

// 更新全屏状态
const updateFullscreenStatus = () => {
  isFullscreen.value = !!document.fullscreenElement;
};

// 获取当前路由对应的菜单项
const updateActiveMenu = () => {
  activeMenu.value = route.path;
  
  // 生成面包屑导航
  const pathParts = route.path.split('/').filter(Boolean);
  const crumbs = [];
  
  // 如果是管理后台首页(/admin 或 /admin/dashboard)，不添加额外的面包屑
  if (route.path === '/admin' || route.path === '/admin/dashboard') {
    breadcrumbs.value = [];
    return;
  }
  
  if (pathParts.length > 1) {
    pathParts.slice(1).forEach(part => {
      // 转换路径为可读性更好的文本
      let text = '';
      switch (part) {
        case 'dashboard': return; // 跳过dashboard，不添加到面包屑
        case 'monitor': text = '系统监控'; break;
        case 'conversation': text = '对话数据统计'; break;
        case 'users': 
          if (pathParts[1] === 'monitor') {
            text = '用户活跃度监控';
          } else {
            text = '用户管理';
          }
          break;
        case 'user-management': text = '用户查询与管理'; break;
        case 'login-history': text = '用户登录历史'; break;
        case 'admins': text = '管理员管理'; break;
        case 'admin-management': text = '管理员与角色管理'; break;
        case 'logs': text = '操作日志'; break;
        case 'content': text = '内容资源管理'; break;
        case 'knowledge-files': text = '知识库管理'; break;
        case 'safety-materials': text = '安全文档库'; break;
        case 'emergency-materials': text = '应急预案库'; break;
        case 'review': text = '内容审核'; break;
        case 'feedback': text = '反馈管理'; break;
        case 'list': 
          if (pathParts[1] === 'feedback') {
            text = '系统功能反馈';
          } else {
            text = 'list';
          }
          break;
        case 'ratings': 
          if (pathParts[1] === 'feedback') {
            text = '生成内容反馈';
          } else {
            text = '内容评价分析';
          }
          break;
        case 'statistics': text = '反馈分类统计'; break;
        case 'status': text = '反馈处理状态管理'; break;
        case 'settings': text = '系统设置'; break;
        case 'system-params': text = '系统参数'; break;
        default: text = part.charAt(0).toUpperCase() + part.slice(1);
      }
      crumbs.push(text);
    });
  }
  
  breadcrumbs.value = crumbs;
};

// 监听路由变化，实时更新菜单和面包屑
watch(() => route.path, (newPath) => {
  updateActiveMenu();
});

// 显示个人资料对话框
const showProfileDialog = () => {
  const adminToken = localStorage.getItem('token');
  if (!adminToken) {
    ElMessage.warning('请先登录');
    router.push('/admin/login');
    return;
  }
  
  // 如果管理员信息未加载，先加载信息
  if (!adminInfo.value || !adminInfo.value.admin_id) {
    fetchAdminInfo().then(() => {
      openProfileDialog();
    });
  } else {
    openProfileDialog();
  }
};

const openProfileDialog = () => {
  // 确保表单数据是最新的
  profileForm.value = {
    admin_id: adminInfo.value.admin_id,
    full_name: adminInfo.value.full_name || '',
    email: adminInfo.value.email || '',
    phone_number: adminInfo.value.phone_number || ''
  };
  profileDialogVisible.value = true;
};

// 显示修改密码对话框
const showPasswordDialog = () => {
  const adminToken = localStorage.getItem('token');
  if (!adminToken) {
    ElMessage.warning('请先登录');
    router.push('/admin/login');
    return;
  }
  
  passwordForm.value = {
    admin_id: adminInfo.value.admin_id,
    old_password: '',
    new_password: '',
    confirm_password: ''
  };
  passwordDialogVisible.value = true;
};

// 更新个人资料
const updateProfile = async () => {
  if (!profileFormRef.value) return;
  
  await profileFormRef.value.validate(async (valid) => {
    if (valid) {
      try {
        loading.value = true;
        const adminToken = localStorage.getItem('token');
        if (!adminToken) {
          ElMessage.error('管理员已退出，请重新登录');
          router.push('/admin/login');
          return;
        }
        
        const response = await axios.put(
          `${API_BASE_URL}/admin/profile`, 
          profileForm.value,
          {
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          }
        );
        
        if (response.data.code === 200) {
          // 更新本地管理员信息
          adminInfo.value = {
            ...adminInfo.value,
            full_name: profileForm.value.full_name,
            email: profileForm.value.email,
            phone_number: profileForm.value.phone_number
          };
          
          ElMessage({
            message: '个人资料更新成功',
            type: 'success',
            duration: 2000
          });
          
          profileDialogVisible.value = false;
        } else {
          ElMessage({
            message: response.data.message || '更新失败，请稍后重试',
            type: 'error',
            duration: 3000
          });
        }
      } catch (error) {
        console.error('更新个人资料失败:', error);
        
        if (error.response?.status === 401) {
          ElMessage.error('登录已过期，请重新登录');
          router.push('/admin/login');
        } else if (error.response?.status === 403) {
          ElMessage.error('您没有权限执行此操作');
        } else {
          ElMessage({
            message: error.response?.data?.message || '网络错误，请检查网络连接',
            type: 'error',
            duration: 3000
          });
        }
      } finally {
        loading.value = false;
      }
    }
  });
};

// 更新密码
const updatePassword = async () => {
  if (!passwordFormRef.value) return;
  
  await passwordFormRef.value.validate(async (valid) => {
    if (valid) {
      try {
        loading.value = true;
        const adminToken = localStorage.getItem('token');
        if (!adminToken) {
          ElMessage.error('管理员已退出，请重新登录');
          router.push('/login');
          return;
        }
        
        const response = await axios.put(
          `${API_BASE_URL}/admin/password`, 
          passwordForm.value,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          }
        );
        
        if (response.data.code === 200) {
          ElMessage.success('密码修改成功');
          passwordDialogVisible.value = false;
          
          // 清空密码表单
          passwordForm.value = {
            admin_id: adminInfo.value.admin_id,
            old_password: '',
            new_password: '',
            confirm_password: ''
          };
        } else {
          ElMessage.error(response.data.message || '修改失败');
        }
      } catch (error) {
        console.error('修改密码失败:', error);
        ElMessage.error(error.response?.data?.message || '修改密码失败');
      } finally {
        loading.value = false;
      }
    }
  });
};

// 登录状态检查和登出功能
const checkLoginStatus = () => {
  const adminToken = localStorage.getItem('token');
  if (!adminToken) {
    ElMessage.warning('您尚未登录或登录已过期，请重新登录');
    router.push('/admin/login');
    return false;
  }
  return true;
};

// 处理登出
const handleLogout = () => {
  ElMessageBox.confirm(
    '确定要退出登录吗?',
    '退出提示',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    }
  )
    .then(() => {
      // 清除本地存储中的登录信息
      localStorage.removeItem('token');
      
      // 提示用户已登出
      ElMessage.success('已成功退出登录');
      
      // 跳转到登录页
      router.push('/login');
    })
    .catch(() => {
      // 用户取消登出操作
    });
};
</script>

<style scoped>
.admin-layout {
  display: flex;
  min-height: 100vh;
  transition: all 0.3s;
}

.dark-mode {
  color: #e0e0e0;
  background-color: #121212;
}

.sidebar {
  width: 240px;
  background-color: #001529;
  color: white;
  overflow-y: auto;
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  z-index: 1000;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  scrollbar-width: none; /* Firefox */
  -ms-overflow-style: none; /* IE and Edge */
}

.sidebar.collapsed {
  width: 64px;
}

/* 隐藏滚动条但保留滚动功能 */
.sidebar::-webkit-scrollbar {
  width: 0;
  display: none; /* Chrome, Safari, Opera */
}

.logo {
  display: flex;
  align-items: center;
  padding: 16px;
  background-color: #002140;
  height: 64px;
  overflow: hidden;
  transition: all 0.3s;
}

.logo img {
  width: 32px;
  height: 32px;
  margin-right: 8px;
  transition: all 0.3s;
}

.logo span {
  font-size: 18px;
  font-weight: bold;
  color: white;
  white-space: nowrap;
  transition: opacity 0.3s;
}

.collapse-btn {
  text-align: center;
  line-height: 40px;
  cursor: pointer;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  color: rgba(255, 255, 255, 0.65);
  transition: all 0.3s;
}

.collapse-btn:hover {
  color: white;
  background-color: rgba(255, 255, 255, 0.08);
}

.sidebar-menu {
  border-right: none;
}

.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  margin-left: 240px;
  transition: all 0.3s;
}

.collapsed-content {
  margin-left: 64px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  height: 64px;
  background-color: white;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
  position: fixed;
  top: 0;
  left: 240px;
  right: 0;
  z-index: 999;
  transition: all 0.3s;
}

.dark-header {
  background-color: #1a1a1a;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
  color: #e0e0e0;
}

.collapsed-content .header {
  left: 64px;
}

.header-left, .header-right {
  display: flex;
  align-items: center;
}

.header-right {
  gap: 16px;
}

.breadcrumb {
  font-size: 14px;
}

.search-box {
  width: 200px;
  transition: width 0.3s;
}

.search-box:focus-within {
  width: 240px;
}

.notifications {
  position: relative;
  cursor: pointer;
}

.notification-badge :deep(.el-badge__content) {
  transform: translate(50%, -50%);
}

.theme-toggle, .fullscreen-toggle {
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
  border-radius: 50%;
  transition: all 0.3s;
  margin: 0 4px;
}

.theme-toggle:hover, .fullscreen-toggle:hover {
  background-color: rgba(0, 0, 0, 0.05);
  transform: scale(1.1);
}

.dark-mode .theme-toggle:hover, .dark-mode .fullscreen-toggle:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.fullscreen-toggle {
  color: #409EFF;
}

.dark-mode .fullscreen-toggle {
  color: #66b1ff;
}

.user-info {
  display: flex;
  align-items: center;
}

.user-avatar-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.user-avatar-info:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.dark-mode .user-avatar-info:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.dropdown-link {
  display: flex;
  align-items: center;
  cursor: pointer;
  color: #333;
  transition: color 0.3s;
}

.dark-mode .dropdown-link {
  color: #e0e0e0;
}

.main-content {
  flex: 1;
  padding: 20px;
  background-color: #f0f2f5;
  overflow: auto;
  margin-top: 64px;
  transition: background-color 0.3s;
}

.dark-mode .main-content {
  background-color: #1e1e1e;
}

/* 深色模式下menu样式调整 */
:deep(.el-menu) {
  border-right: none;
}

:deep(.el-sub-menu__title) {
  &:hover {
    background-color: #001f3f !important;
  }
}

:deep(.el-menu-item) {
  &:hover {
    background-color: #001f3f !important;
  }
}

.dark-mode :deep(.el-sub-menu__title) {
  &:hover {
    background-color: #252525 !important;
  }
}

.dark-mode :deep(.el-menu-item) {
  &:hover {
    background-color: #252525 !important;
  }
}

/* 对话框样式 */
:deep(.profile-dialog), :deep(.password-dialog) {
  border-radius: 8px;
}

:deep(.el-dialog__header) {
  margin-bottom: 10px;
  padding-bottom: 15px;
  border-bottom: 1px solid #ebeef5;
}

:deep(.el-dialog__footer) {
  border-top: 1px solid #ebeef5;
  padding-top: 15px;
}

.profile-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 0;
}

.profile-role {
  margin-top: 10px;
}

.password-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: bold;
  font-size: 16px;
  color: #409EFF;
}

.profile-form, .password-form {
  margin-top: 15px;
}

.form-tip {
  font-size: 12px;
  color: #909399;
  line-height: 1.2;
  padding-top: 4px;
}

:deep(.el-card__header) {
  padding: 15px;
  background-color: #f8f9fa;
}

:deep(.el-input.is-disabled .el-input__wrapper) {
  background-color: #f5f7fa;
}
</style> 