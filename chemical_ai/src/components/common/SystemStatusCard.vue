<template>
  <div class="info-card">
    <div class="info-grid">
      <div class="info-item" @mouseenter="showTooltip('knowledge', $event)" @mouseleave="hideTooltip">
        <span class="info-label">知识库版本</span>
        <span class="info-value">{{ systemStatus.knowledge_base }}</span>
      </div>
      <div class="info-item" @mouseenter="showTooltip('date', $event)" @mouseleave="hideTooltip">
        <span class="info-label">知识库更新日期</span>
        <span class="info-value">{{ systemStatus.update_date }}</span>
      </div>
      <div class="info-item" @mouseenter="showTooltip('sources', $event)" @mouseleave="hideTooltip">
        <span class="info-label">数据源</span>
        <span class="info-value">{{ systemStatus.data_sources }}</span>
      </div>
      <div class="info-item" @mouseenter="showTooltip('response', $event)" @mouseleave="hideTooltip">
        <span class="info-label">响应时间</span>
        <span class="info-value">{{ systemStatus.response_time }}</span>
      </div>
    </div>
    
    <!-- 使用teleport将tooltip传送到body下 -->
    <teleport to="body">
      <div v-if="activeTooltip" class="global-tooltip" :style="tooltipStyle">
        <div class="tooltip-content">
          <p v-if="activeTooltip === 'knowledge'">当前系统知识库版本，包含化工安全相关的全部专业知识和操作规范。</p>
          <p v-if="activeTooltip === 'knowledge'">知识库定期更新，确保数据的准确性和时效性。</p>
          
          <p v-if="activeTooltip === 'date'">知识库的最近一次更新时间，系统定期同步最新的化工安全标准和法规。</p>
          <p v-if="activeTooltip === 'date'">更新频率：每月更新一次基础数据，紧急安全通知实时更新。</p>
          
          <p v-if="activeTooltip === 'sources'">系统整合的专业数据源数量，包括国内外权威化工安全标准、MSDS数据库及企业内部安全操作规范。</p>
          <p v-if="activeTooltip === 'sources'">所有数据源均经过专业审核，确保信息准确可靠。</p>
          
          <p v-if="activeTooltip === 'response'">系统平均响应时间，反映了AI处理问题的速度。</p>
          <p v-if="activeTooltip === 'response'">响应时间受网络状况、问题复杂度和系统负载影响，通常保持在5秒以内。</p>
        </div>
        <div class="tooltip-arrow" :style="arrowStyle"></div>
      </div>
    </teleport>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { API_BASE_URL } from '../../config'

const systemStatus = ref({
  knowledge_base: '加载中...',
  update_date: '加载中...',
  data_sources: '加载中...',
  response_time: '加载中...'
})

// 从全局状态中获取系统状态 - 避免重复请求
let useGlobalStatus = false;
let statusTimer = null;
const activeTooltip = ref(null);
const tooltipStyle = ref({});
const arrowStyle = ref({});

const showTooltip = (type, event) => {
  activeTooltip.value = type;
  
  // 获取触发元素的位置
  if (event && event.currentTarget) {
    const rect = event.currentTarget.getBoundingClientRect();
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    
    // 计算基本位置
    let left = rect.left + (rect.width / 2);
    let top = rect.bottom + 15; // 默认显示在下方
    let position = 'bottom';
    
    // 检查下方的空间是否足够
    const tooltipHeight = 180; // 估计的tooltip高度
    if (top + tooltipHeight > viewportHeight) {
      // 如果下方空间不足，显示在上方
      top = rect.top - 15;
      position = 'top';
    }
    
    // 检查左右空间是否足够
    const tooltipWidth = 240; // tooltip宽度
    if (left - tooltipWidth/2 < 0) {
      left = tooltipWidth/2 + 10; // 左侧边缘偏移
    } else if (left + tooltipWidth/2 > viewportWidth) {
      left = viewportWidth - tooltipWidth/2 - 10; // 右侧边缘偏移
    }
    
    // 设置tooltip位置
    tooltipStyle.value = {
      left: `${left}px`,
      top: position === 'bottom' ? `${top}px` : 'auto',
      bottom: position === 'top' ? `${viewportHeight - top}px` : 'auto',
    };
    
    // 设置箭头位置
    if (position === 'bottom') {
      arrowStyle.value = {
        top: '-8px',
        bottom: 'auto',
        borderBottom: '8px solid rgba(0, 0, 0, 0.9)',
        borderTop: 'none'
      };
    } else {
      arrowStyle.value = {
        top: 'auto',
        bottom: '-8px',
        borderTop: '8px solid rgba(0, 0, 0, 0.9)',
        borderBottom: 'none'
      };
    }
  }
};

const hideTooltip = () => {
  activeTooltip.value = null;
};

const fetchSystemStatus = async () => {
  try {
    // 检查是否有其他组件已获取状态 - 实际项目中可使用状态管理库
    const cachedStatus = sessionStorage.getItem('systemStatus');
    const cachedTime = sessionStorage.getItem('systemStatusTime');
    
    // 如果缓存状态存在且未过期（3分钟内的缓存有效）
    if (cachedStatus && cachedTime && (Date.now() - parseInt(cachedTime)) < 180000) {
      systemStatus.value = JSON.parse(cachedStatus);
      return;
    }
    
    const response = await fetch(`${API_BASE_URL}/system/status`)
    if (!response.ok) {
      throw new Error('获取系统状态失败')
    }
    const data = await response.json()
    systemStatus.value = data
    
    // 缓存系统状态和时间戳
    sessionStorage.setItem('systemStatus', JSON.stringify(data));
    sessionStorage.setItem('systemStatusTime', Date.now().toString());
  } catch (error) {
    console.error('获取系统状态失败:', error)
    systemStatus.value = {
      knowledge_base: '获取失败',
      update_date: '获取失败',
      data_sources: '获取失败',
      response_time: '获取失败'
    }
  }
}

onMounted(() => {
  fetchSystemStatus()
  // 设置5分钟的轮询间隔
  statusTimer = setInterval(fetchSystemStatus, 300000);
})

// 组件卸载时清除定时器
onUnmounted(() => {
  if (statusTimer) clearInterval(statusTimer);
})
</script>

<style>
/* 全局提示样式 (不使用scoped，确保传送后的样式正确) */
.global-tooltip {
  position: fixed;
  width: 240px;
  background: rgba(0, 0, 0, 0.9);
  color: #fff;
  padding: 12px;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  z-index: 10000;
  transform: translateX(-50%);
  animation: fadeIn 0.2s ease;
  pointer-events: none;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(5px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}

.tooltip-content {
  font-size: 0.8rem;
  line-height: 1.4;
  margin: 0;
}

.tooltip-content p {
  margin: 0 0 8px 0;
}

.tooltip-content p:last-child {
  margin-bottom: 0;
}

.tooltip-arrow {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  width: 0;
  height: 0;
  border-left: 8px solid transparent;
  border-right: 8px solid transparent;
}

@media (max-width: 768px) {
  .global-tooltip {
    width: 200px;
  }
}

@media (max-width: 480px) {
  .global-tooltip {
    width: 180px;
  }
}
</style>

<style scoped>
.info-card {
  background: transparent;
  border-radius: 14px;
  padding: 16px;
  box-shadow: none;
  border: none;
  transition: all 0.3s ease;
  position: relative;
}

/* 移除装饰背景 */
.info-card::before {
  display: none;
}

.info-card:hover {
  transform: none;
  box-shadow: none;
}

.info-card-header {
  margin-bottom: 14px;
  text-align: center;
  position: relative;
  z-index: 1;
}

.info-card h3 {
  font-size: 1.1rem;
  margin: 0;
  font-weight: 600;
  background: linear-gradient(135deg, #4f46e5, #6366f1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  letter-spacing: 0.5px;
  position: relative;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  position: relative;
  z-index: 1;
}

.info-item {
  display: flex;
  flex-direction: column;
  padding: 8px;
  border-radius: 10px;
  transition: all 0.2s ease;
  background: transparent;
  border: none;
  position: relative;
  cursor: pointer;
}

.info-item:hover {
  background: rgba(255, 255, 255, 0.1);
  transform: none;
  box-shadow: none;
}

.info-label {
  font-size: 0.75rem;
  color: #6366f1;
  margin-bottom: 4px;
  font-weight: 500;
  text-align: center;
}

.info-value {
  font-size: 0.9rem;
  font-weight: 600;
  color: #1f2937;
  text-align: center;
  position: relative;
  transition: all 0.2s ease;
}

.info-value::after {
  content: "";
  position: absolute;
  bottom: -2px;
  left: 50%;
  width: 0;
  height: 1px;
  background: #6366f1;
  transition: all 0.3s ease;
  transform: translateX(-50%);
}

.info-item:hover .info-value::after {
  width: 40%;
}

/* Dark theme styles */
body.dark-theme .info-card {
  background: transparent;
  border: none;
  box-shadow: none;
}

body.dark-theme .info-card::before {
  display: none;
}

body.dark-theme .info-card h3 {
  background: linear-gradient(135deg, #a5b4fc, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

body.dark-theme .info-item {
  background: transparent;
  border: none;
}

body.dark-theme .info-item:hover {
  background: rgba(255, 255, 255, 0.05);
}

body.dark-theme .info-label {
  color: #a5b4fc;
}

body.dark-theme .info-value {
  color: #e5e7eb;
}

body.dark-theme .info-value::after {
  background: #a5b4fc;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .info-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .info-grid {
    grid-template-columns: 1fr;
  }
}
</style> 