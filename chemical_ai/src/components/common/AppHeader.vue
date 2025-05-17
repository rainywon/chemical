<template>
  <div class="header">
    <div class="logo-container">
      <img src="@/assets/product.png" alt="应用Logo" class="logo fade-in" />
    </div>
    <div class="title-container">
      <h1 class="title fade-in">天工AI智能助手</h1>
      <p class="subtitle fade-in-delay">您的专业化工安全AI对话助手，提供实时咨询与安全指导</p>
    </div>
    <div class="status-section">
      <div class="status-badge fade-in-delay-2" :class="systemStatus">
        <span>{{ statusText }}</span>
      </div>
      <div class="settings-dropdown fade-in-delay-2" ref="settingsDropdown">
        <button class="settings-button" @click="toggleSettings">
          <span class="settings-icon">设置</span>
        </button>
        
        <teleport to="body">
          <div class="dropdown-menu-container" v-show="showSettings" :style="dropdownPosition">
            <div class="dropdown-menu">
              <div class="dropdown-item" @click="handleToggleTheme">
                <span>切换主题 ({{ currentTheme === 'light' ? '浅色' : '深色' }})</span>
              </div>
              <div class="dropdown-item" @click="handleShowFeedback">
                <span>信息反馈</span>
              </div>
              <div class="dropdown-item" @click="handleLogout">
                <span>退出登录</span>
              </div>
            </div>
          </div>
        </teleport>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { API_BASE_URL } from '../../config';

// Props
const props = defineProps({
  currentTheme: {
    type: String,
    required: true
  }
});

// Emits
const emit = defineEmits(['toggle-theme', 'show-feedback', 'logout']);

// State
const showSettings = ref(false);
const systemStatus = ref('normal');
const settingsDropdown = ref(null);
const dropdownPosition = ref({});
const statusTimer = ref(null);

// 获取系统状态
const fetchSystemStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/system/status`);
    if (!response.ok) {
      throw new Error('获取系统状态失败');
    }
    const data = await response.json();
    systemStatus.value = data.system_status;
  } catch (error) {
    console.error('获取系统状态失败:', error);
    systemStatus.value = 'error';
  }
};

// 状态文本
const statusText = computed(() => {
  switch (systemStatus.value) {
    case 'normal':
      return '系统正常';
    case 'warning':
      return '系统警告';
    case 'error':
      return '系统异常';
    default:
      return '系统正常';
  }
});

// Methods
const handleToggleTheme = () => {
  showSettings.value = false;
  emit('toggle-theme');
};

const handleShowFeedback = () => {
  showSettings.value = false;
  emit('show-feedback');
};

const handleLogout = () => {
  showSettings.value = false;
  emit('logout');
};

const calculateDropdownPosition = () => {
  if (!settingsDropdown.value) return;
  
  const rect = settingsDropdown.value.getBoundingClientRect();
  
  dropdownPosition.value = {
    position: 'fixed',
    top: `${rect.bottom}px`,
    right: `${window.innerWidth - rect.right}px`
  };
};

const toggleSettings = () => {
  showSettings.value = !showSettings.value;
  
  if (showSettings.value) {
    calculateDropdownPosition();
  }
};

const handleClickOutside = (event) => {
  if (showSettings.value && settingsDropdown.value && !settingsDropdown.value.contains(event.target)) {
    const dropdownMenus = document.querySelectorAll('.dropdown-menu-container');
    let clickedOnMenu = false;
    
    dropdownMenus.forEach(menu => {
      if (menu.contains(event.target)) {
        clickedOnMenu = true;
      }
    });
    
    if (!clickedOnMenu) {
      showSettings.value = false;
    }
  }
};

// 组件挂载时获取系统状态
onMounted(() => {
  fetchSystemStatus();
  // 将轮询间隔从30秒增加到120秒（2分钟）
  statusTimer.value = setInterval(fetchSystemStatus, 120000);
  
  // 添加点击外部关闭事件监听
  document.addEventListener('click', handleClickOutside);
  
  // 添加窗口大小变化监听，以更新下拉菜单位置
  window.addEventListener('resize', () => {
    if (showSettings.value) {
      calculateDropdownPosition();
    }
  });
});

// 组件卸载时清除定时器和事件监听
onUnmounted(() => {
  if (statusTimer.value) {
    clearInterval(statusTimer.value);
  }
  document.removeEventListener('click', handleClickOutside);
  window.removeEventListener('resize', () => {
    if (showSettings.value) {
      calculateDropdownPosition();
    }
  });
});
</script>

<style scoped>
/* 添加渐入动画效果 */
.fade-in {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
}

.fade-in-delay {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
  animation-delay: 0.3s;
}

.fade-in-delay-2 {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
  animation-delay: 0.6s;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.logo-container {
  padding-right: 20px;
}

.logo {
  width: 48px;
  height: 48px;
  object-fit: contain;
}

.title-container {
  flex-grow: 1;
  text-align: center;
}

.title {
  font-size: 1.6rem;
  font-weight: 700;
  color: #1a1f36;
  margin: 0 0 4px 0;
  line-height: 1.2;
  text-align: center;
}

.subtitle {
  font-size: 0.9rem;
  color: #4a5568;
  margin: 0;
  line-height: 1.4;
  text-align: center;
}

.status-section {
  display: flex;
  align-items: center;
  gap: 10px;
}

.status-badge {
  display: flex;
  align-items: center;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
}

.status-badge.normal {
  background-color: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

.status-badge.warning {
  background-color: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

.status-badge.error {
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
}

/* 设置按钮和下拉菜单 */
.settings-dropdown {
  position: relative;
}

.settings-button {
  background: transparent;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.3s ease;
  font-size: 0.9rem;
  color: #4a5568;
}

.settings-button:hover {
  background: rgba(0, 0, 0, 0.05);
}

.dropdown-menu-container {
  position: fixed;
  z-index: 20000;
  filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
}

.dropdown-menu {
  background: white;
  border-radius: 8px;
  overflow: hidden;
  min-width: 160px;
  animation: dropdownFadeIn 0.2s ease;
}

@keyframes dropdownFadeIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.dropdown-item {
  padding: 10px 15px;
  cursor: pointer;
  transition: background 0.2s ease;
  font-size: 0.85rem;
}

.dropdown-item:hover {
  background: rgba(0, 0, 0, 0.05);
}

/* Dark theme styles */
body.dark-theme .title {
  color: #f3f4f6;
}

body.dark-theme .subtitle {
  color: #d1d5db;
}

body.dark-theme .dropdown-menu {
  background: #1f2937;
}

body.dark-theme .dropdown-item {
  color: #e5e7eb;
}

body.dark-theme .dropdown-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

body.dark-theme .settings-button {
  color: #d1d5db;
}

body.dark-theme .settings-button:hover {
  background: rgba(255, 255, 255, 0.1);
}
</style> 