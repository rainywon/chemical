<template>
  <div>
    <!-- 文档参考小方框 -->
    <div v-if="Array.isArray(references) ? references.length > 0 : Object.keys(references).length > 0" class="references-summary" @click="toggleDetails">
      <svg class="reference-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M7 7H17" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        <path d="M7 12H17" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        <path d="M7 17H13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg>
      <span>参考了 {{ Array.isArray(references) ? references.length : Object.keys(references).length }} 篇文档</span>
    </div>
  </div>

  <!-- 使用 Teleport 将内容移至 body 下以避免嵌套问题 -->
  <Teleport to="body">
    <!-- 遮罩层 -->
    <div v-if="showDetails" class="overlay" @click="closeDetails"></div>

    <!-- 显示详细参考信息 -->
    <transition name="modal">
      <div v-if="showDetails" class="references-detail">
        <div class="references-header">
          <h3>参考文档详情</h3>
          <button class="close-button" @click="closeDetails" aria-label="关闭">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
        </div>
        <div class="references-content">
          <div class="reference-card" v-for="(reference, index) in referencesArray" :key="index">
            <div class="reference-header">
              <div class="reference-source">
                <svg class="file-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M14 2V8H20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>{{ reference.file }}</span>
              </div>
              <div class="reference-score">
                <div class="score-indicator" :style="{ width: `${reference.score * 100}%` }"></div>
                <span>相关性 {{ (reference.score * 100).toFixed(0) }}%</span>
              </div>
            </div>
            <div class="reference-content">
              {{ reference.content }}
            </div>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<script setup>
import { ref, computed } from "vue";

const props = defineProps({
  references: {
    type: [Array, Object],
    required: true,
  },
});

// Computed property to handle both array and object references
const referencesArray = computed(() => {
  return Array.isArray(props.references) 
    ? props.references 
    : Object.values(props.references);
});

const showDetails = ref(false); // 控制显示详细信息

// 切换详细信息的显示
const toggleDetails = () => {
  showDetails.value = true;
  // 显示弹窗时禁止页面滚动
  document.body.style.overflow = 'hidden';
};

// 关闭详细信息的显示
const closeDetails = () => {
  showDetails.value = false;
  // 关闭弹窗时恢复页面滚动
  document.body.style.overflow = '';
};
</script>

<style scoped>
/* 文档参考小方框样式 */
.references-summary {
  background: linear-gradient(135deg, #4f46e5, #7c3aed);
  color: white;
  padding: 10px 16px;
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  box-shadow: 0px 3px 12px rgba(124, 58, 237, 0.2), 0px 1px 4px rgba(124, 58, 237, 0.1);
  transition: all 0.2s ease;
  backdrop-filter: blur(5px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  user-select: none;
}

.references-summary:hover {
  transform: translateY(-2px);
  box-shadow: 0px 6px 16px rgba(124, 58, 237, 0.25);
  background: linear-gradient(135deg, #4338ca, #6d28d9);
}

.references-summary:active {
  transform: translateY(0);
  background: linear-gradient(135deg, #3730a3, #5b21b6);
}

.reference-icon {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
}

/* 遮罩层 */
.overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(4px);
  z-index: 9999;
  cursor: pointer;
  animation: fadeIn 0.2s ease;
}

/* 显示详细参考信息的样式 */
.references-detail {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 85%;
  max-width: 900px;
  height: 75%;
  max-height: 700px;
  background-color: white;
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15), 0 6px 12px rgba(0, 0, 0, 0.1);
  z-index: 10000;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid rgba(226, 232, 240, 0.8);
}

.references-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.8);
}

.references-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
}

.references-content {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
  background: #f8fafc;
}

.close-button {
  background: transparent;
  border: none;
  cursor: pointer;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #64748b;
  transition: all 0.2s ease;
}

.close-button:hover {
  background: #f1f5f9;
  color: #1e293b;
}

.close-button svg {
  width: 22px;
  height: 22px;
}

/* 卡片样式 */
.reference-card {
  background-color: white;
  border-radius: 16px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  overflow: hidden;
  transition: all 0.2s ease;
  border: 1px solid rgba(226, 232, 240, 0.8);
  position: relative;
}

.reference-card:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.reference-header {
  padding: 16px 20px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.6);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.reference-source {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 600;
  color: #334155;
}

.file-icon {
  width: 18px;
  height: 18px;
  color: #6366f1;
}

.reference-score {
  position: relative;
  height: 8px;
  width: 120px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  align-items: center;
}

.score-indicator {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #4f46e5, #8b5cf6);
  border-radius: 4px;
}

.reference-score span {
  position: absolute;
  font-size: 12px;
  color: #64748b;
  font-weight: 500;
  bottom: -20px;
  right: 0;
  white-space: nowrap;
}

.reference-content {
  padding: 16px 20px;
  font-size: 15px;
  line-height: 1.6;
  color: #475569;
  white-space: pre-line;
  max-height: 300px;
  overflow-y: auto;
}

/* 动画效果 */
.modal-enter-active, .modal-leave-active {
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.modal-enter-from {
  opacity: 0;
  transform: translate(-50%, -48%) scale(0.96);
}

.modal-leave-to {
  opacity: 0;
  transform: translate(-50%, -48%) scale(0.96);
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .references-detail {
    width: 95%;
    height: 85%;
  }
  
  .reference-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .reference-score {
    width: 100%;
  }
}

/* 适配深色模式（如果需要） */
@media (prefers-color-scheme: dark) {
  .references-summary {
    background: linear-gradient(135deg, #4c1d95, #6d28d9);
  }
  
  .references-detail {
    background-color: #1e293b;
    border-color: rgba(51, 65, 85, 0.6);
  }
  
  .references-header {
    border-color: rgba(51, 65, 85, 0.6);
  }
  
  .references-header h3 {
    color: #e2e8f0;
  }
  
  .references-content {
    background: #0f172a;
  }
  
  .close-button {
    color: #94a3b8;
  }
  
  .close-button:hover {
    background: #334155;
    color: #f1f5f9;
  }
  
  .reference-card {
    background-color: #1e293b;
    border-color: rgba(51, 65, 85, 0.6);
  }
  
  .reference-header {
    border-color: rgba(51, 65, 85, 0.6);
  }
  
  .reference-source {
    color: #e2e8f0;
  }
  
  .reference-content {
    color: #cbd5e1;
  }
  
  .reference-score {
    background: #334155;
  }
}
</style>
