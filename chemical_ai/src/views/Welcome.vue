<template>
  <div class="welcome-container">
    <div class="welcome-content" :class="{ 'content-loaded': isLoaded }">
      <!-- 系统概览区域 -->
      <div class="system-overview" :class="{ 'fade-in-up': isLoaded }">
        <AppHeader 
          :current-theme="currentTheme" 
          @toggle-theme="toggleTheme" 
          @show-feedback="showFeedback = true" 
          @logout="logout" 
        />
        <SystemStatusCard />
      </div>
      
      <!-- 功能模块区域 -->
      <div class="features-grid">
        <!-- 安全资料库模块 -->
        <FileLibraryCard :class="{ 'fade-in-left': isLoaded }" />
        
        <!-- 应急处理模块 -->
        <EmergencyResponseCard :class="{ 'fade-in-right': isLoaded }" />
        
        <!-- 智能问答模块 -->
        <AIFeatureCard :class="{ 'fade-in-right-delay': isLoaded }" />
      </div>
      
      <!-- 反馈弹窗 -->
      <ModalFeedback 
        :show="showFeedback" 
        @close="showFeedback = false" 
        @submit-success="handleFeedbackSuccess" 
      />
      
      <!-- 退出登录确认弹窗 -->
      <ModalConfirm
        :show="showConfirmModal"
        :message="confirmMessage"
        @confirm="confirmLogout"
        @cancel="showConfirmModal = false"
      />
      
      <!-- 反馈提交成功通知 -->
      <NotificationToast
        :show="showSuccessNotification"
        title="提交成功"
        message="感谢您的反馈！我们会认真考虑您的意见。"
        type="success"
        :duration="3000"
        @close="showSuccessNotification = false"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import { API_BASE_URL } from "../config";
// 导入组件
import AppHeader from '@/components/common/AppHeader.vue';
import SystemStatusCard from '@/components/common/SystemStatusCard.vue';
import AIFeatureCard from '@/components/common/AIFeatureCard.vue';
import FileLibraryCard from '@/components/common/FileLibraryCard.vue';
import EmergencyResponseCard from '@/components/common/EmergencyResponseCard.vue';
import ModalFeedback from '@/components/common/ModalFeedback.vue';
import ModalConfirm from '@/components/common/ModalConfirm.vue';
import NotificationToast from '@/components/common/NotificationToast.vue';

// 路由
const router = useRouter();

// 状态变量
const showFeedback = ref(false);
const showConfirmModal = ref(false);
const confirmMessage = ref('');
const currentTheme = ref('light');
const showSuccessNotification = ref(false);
const isLoaded = ref(false);

// 检查并应用已保存的主题
onMounted(() => {
  // 检查本地存储中的主题设置
  const savedTheme = localStorage.getItem('appTheme');
  if (savedTheme) {
    currentTheme.value = savedTheme;
    applyTheme(savedTheme);
  }
  
  // 检查系统首选主题
  if (!savedTheme && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    currentTheme.value = 'dark';
    applyTheme('dark');
  }
  
  // 添加页面加载动画
  setTimeout(() => {
    isLoaded.value = true;
  }, 300);
});

// 应用主题
const applyTheme = (theme) => {
  if (theme === 'dark') {
    document.body.classList.add('dark-theme');
  } else {
    document.body.classList.remove('dark-theme');
  }
  // 保存主题设置到本地存储
  localStorage.setItem('appTheme', theme);
};

// 切换主题
const toggleTheme = () => {
  const newTheme = currentTheme.value === 'light' ? 'dark' : 'light';
  currentTheme.value = newTheme;
  applyTheme(newTheme);
};

// 处理反馈成功
const handleFeedbackSuccess = () => {
  showSuccessNotification.value = true;
};

// 显示确认对话框
const showConfirm = (message) => {
  confirmMessage.value = message;
  showConfirmModal.value = true;
};

// 确认操作
const confirmLogout = async () => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      // 如果没有token，直接清除本地存储并跳转到登录页
      clearLocalStorage();
      router.push("/login");
      return;
    }

    // 调用后端退出登录 API
    const response = await fetch(`${API_BASE_URL}/logout/`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    // 无论响应如何，都清除本地存储
    clearLocalStorage();

    if (!response.ok) {
      // 如果是401错误，说明token已经无效，直接跳转到登录页
      if (response.status === 401) {
        router.push("/login");
        return;
      }
      const data = await response.json();
      throw new Error(data.message || '退出登录失败');
    }

    // 跳转到登录页面
    setTimeout(() => {
      router.push("/login");
    }, 200);
  } catch (error) {
    console.error("退出登录失败:", error);
    ElMessage.error(error.message || "退出登录失败，请重试");
  }
};

// 清除本地存储的辅助函数
const clearLocalStorage = () => {
  localStorage.removeItem("token");
  localStorage.removeItem("mobile");
};

// 退出登录
const logout = () => {
  showConfirm('确定要退出登录吗？');
};
</script>

<style scoped>
.welcome-container {
  height: 100vh;
  background: linear-gradient(135deg, #f0f4ff 0%, #e6eaff 50%, #dde2ff 100%);
  padding: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  box-sizing: border-box;
  position: relative;
}

/* 添加背景装饰元素 */
.welcome-container::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    radial-gradient(circle at 10% 20%, rgba(79, 70, 229, 0.03) 0%, transparent 40%),
    radial-gradient(circle at 80% 70%, rgba(59, 130, 246, 0.03) 0%, transparent 30%);
  z-index: 0;
}

.welcome-content {
  max-width: 1400px;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 20px;
  max-height: calc(100vh - 40px);
  position: relative;
  z-index: 1;
  opacity: 0;
  transform: translateY(10px);
  transition: opacity 0.5s ease, transform 0.5s ease;
}

.content-loaded {
  opacity: 1;
  transform: translateY(0);
}

/* 系统概览样式 */
.system-overview {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(233, 236, 239, 0.7);
  backdrop-filter: blur(8px);
  transition: all 0.3s ease, transform 0.6s ease, opacity 0.6s ease;
  opacity: 0;
  transform: translateY(20px);
}

.system-overview:hover {
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.fade-in-up {
  opacity: 1;
  transform: translateY(0);
}

/* 功能模块网格样式 */
.features-grid {
  display: grid;
  grid-template-columns: 3fr 3fr 4fr;
  grid-template-areas: "file emergency ai";
  gap: 20px;
  overflow: hidden;
  flex: 1;
  height: calc(100vh - 250px);
  max-height: 600px;
}

.features-grid > .file-library-card {
  grid-area: file;
}

.features-grid > .emergency-response-card {
  grid-area: emergency;
}

.features-grid > .ai-feature-card {
  grid-area: ai;
}

.fade-in-left {
  opacity: 0;
  transform: translateX(-20px);
  transition: all 0.5s ease 0.2s;
}

.fade-in-left.fade-in-left {
  opacity: 1;
  transform: translateX(0);
}

.fade-in-right {
  opacity: 0;
  transform: translateX(20px);
  transition: all 0.5s ease 0.2s;
}

.fade-in-right.fade-in-right {
  opacity: 1;
  transform: translateX(0);
}

.fade-in-right-delay {
  opacity: 0;
  transform: translateX(20px);
  transition: all 0.5s ease 0.4s;
}

.fade-in-right-delay.fade-in-right-delay {
  opacity: 1;
  transform: translateX(0);
}

.right-side {
  display: none; /* 不再需要这个容器 */
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .features-grid {
    grid-template-columns: 1fr 1fr;
    grid-template-areas: 
      "file emergency"
      "ai ai";
  }
}

@media (max-width: 900px) {
  .features-grid {
    grid-template-columns: 1fr;
    height: auto;
    grid-template-areas: 
      "file"
      "emergency"
      "ai";
  }
  
  .fade-in-right, .fade-in-right-delay {
    transition: all 0.5s ease 0.2s;
  }
}

/* 暗色模式适配 */
body.dark-theme {
  background: #212529;
  color: #f8f9fa;
}

body.dark-theme .welcome-container {
  background: linear-gradient(135deg, #1e2c4f 0%, #1a1f36 50%, #121827 100%);
}

body.dark-theme .welcome-container::before {
  background: 
    radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
    radial-gradient(circle at 80% 70%, rgba(79, 70, 229, 0.06) 0%, transparent 30%);
}

body.dark-theme .system-overview {
  background: rgba(31, 41, 55, 0.8);
  border-color: rgba(55, 65, 81, 0.5);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
</style> 