<template>
  <div class="feature-card small-feature file-library-card">
    <div class="feature-header">
      <h2 class="fade-in">安全资料库</h2>
    </div>
    <p class="feature-desc fade-in-delay">浏览与下载各类化工安全相关文档与标准</p>
    <div class="feature-tags fade-in-delay-2">
      <span class="tag file-tag">安全手册</span>
      <span class="tag file-tag">标准文件</span>
      <span class="tag file-tag">危化品MSDS</span>
    </div>
    <div class="file-list fade-in-delay-3">
      <div class="file-item" v-for="file in latestFiles.slice(0, 8)" :key="file.id">
        <span class="file-name" :title="file.name">{{ file.name }}</span>
      </div>
    </div>
    <div class="button-container fade-in-delay-3">
      <router-link to="/files" class="action-button file-button">
        浏览文件
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { API_BASE_URL } from '../../config';

const latestFiles = ref([]);

// 获取最新的文件列表
const fetchLatestFiles = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/safety_files/?page=1&page_size=10`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`,
      }
    });
    
    if (!response.ok) {
      throw new Error('获取文件列表失败');
    }
    
    const data = await response.json();
    if (data.code === 200) {
      latestFiles.value = data.data;
    }
  } catch (error) {
    console.error('获取文件列表失败:', error);
  }
};

onMounted(() => {
  fetchLatestFiles();
});
</script>

<style scoped>
.feature-card {
  background: white;
  border-radius: 16px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  overflow: hidden;
}

.small-feature {
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 200px);
}

.feature-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-bottom: 15px;
}

.feature-icon {
  font-size: 1.5rem;
}

.feature-card h2 {
  margin: 0;
  font-size: 1.3rem;
  color: #1a1f36;
}

.feature-desc {
  font-size: 0.9rem;
  color: #4a5568;
  margin: 0 0 15px 0;
  line-height: 1.4;
}

/* 标签样式 */
.feature-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.tag {
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(79, 70, 229, 0.05));
  color: #4f46e5;
  font-size: 0.7rem;
  padding: 4px 10px;
  border-radius: 6px;
  font-weight: 500;
  border: 1px solid rgba(79, 70, 229, 0.15);
  transition: all 0.2s ease;
}

.tag:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 5px rgba(79, 70, 229, 0.1);
}

.file-tag {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.05));
  color: #10b981;
  border: 1px solid rgba(16, 185, 129, 0.15);
}

.file-tag:hover {
  box-shadow: 0 2px 5px rgba(16, 185, 129, 0.1);
}

/* 文件列表样式 */
.file-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
  flex: 1;
  overflow-y: hidden;
  width: 100%;
  box-sizing: border-box;
  border-radius: 8px;
  padding: 12px;
  border: 1px solid rgba(16, 185, 129, 0.15);
  box-shadow: 0 2px 10px rgba(16, 185, 129, 0.05);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(247, 255, 250, 0.9));
  height: 175px;
}

.file-item {
  display: flex;
  justify-content: flex-start;
  align-items: center;
  padding: 10px 14px;
  border-radius: 8px;
  transition: all 0.2s ease;
  width: 100%;
  box-sizing: border-box;
  background: white;
  cursor: pointer;
  border-left: 2px solid transparent;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
  margin-bottom: 0;
}

.file-item:hover {
  border-left: 2px solid #10b981;
  transform: translateX(2px);
  box-shadow: 0 2px 5px rgba(16, 185, 129, 0.1);
}

.file-name {
  display: block;
  font-size: 0.8rem;
  font-weight: normal;
  color: #4a5568;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  width: 100%;
  position: relative;
  box-sizing: border-box;
  padding-right: 5px;
  text-align: left;
}

.file-name:hover::after {
  content: attr(title);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  padding: 4px 8px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  border-radius: 4px;
  font-size: 0.75rem;
  white-space: nowrap;
  z-index: 1000;
  margin-bottom: 4px;
}

/* 按钮容器 */
.button-container {
  margin-top: 10px;
  display: flex;
  justify-content: center;
  width: 100%;
  position: relative;
  bottom: 5px;
}

/* 按钮样式 */
.action-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #4f46e5;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 10px 16px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.2s ease;
  width: 75%;
  text-align: center;
}

.action-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
}

.file-button {
  background: linear-gradient(135deg, #10b981, #34d399);
}

/* Dark theme styles */
body.dark-theme .feature-card {
  background: #1f2937;
}

body.dark-theme .feature-card h2 {
  color: #f3f4f6;
}

body.dark-theme .feature-desc, 
body.dark-theme .file-item {
  color: #d1d5db;
}

body.dark-theme .file-list {
  background: linear-gradient(135deg, rgba(31, 41, 55, 0.7), rgba(17, 24, 39, 0.7));
  border-color: rgba(16, 185, 129, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

body.dark-theme .file-item {
  background: rgba(31, 41, 55, 0.9);
  border: none;
  border-left: 2px solid transparent;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

body.dark-theme .file-item:hover {
  background: rgba(31, 41, 55, 0.95);
  border-left: 2px solid #10b981;
}

body.dark-theme .tag {
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(79, 70, 229, 0.1));
  border-color: rgba(79, 70, 229, 0.2);
}

body.dark-theme .file-tag {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.1));
  border-color: rgba(16, 185, 129, 0.2);
}

/* Responsive styles */
@media (max-width: 900px) {
  .small-feature {
    max-height: none;
  }
  
  .action-button {
    width: 100%;
  }
}

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

.fade-in-delay-3 {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
  animation-delay: 0.9s;
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
</style> 