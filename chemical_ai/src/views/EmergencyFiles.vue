<template>
  <div class="files-container">
    <!-- 左侧边栏 -->
    <div class="sidebar">
      <!-- 返回首页按钮 -->
      <div class="back-button" @click="goToWelcome">
        <span class="back-icon">←</span>
        <span class="back-text">返回首页</span>
      </div>
      
      <!-- 分类列表 -->
      <div class="category-section">
        <h2 class="sidebar-title">文档分类</h2>
        <ul class="category-list">
          <li class="category-item active">
            所有分类
          </li>
        </ul>
      </div>
    </div>
    
    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 顶部标题和搜索 -->
      <div class="content-header">
        <h1 class="page-title">事故案例库</h1>
        <div class="search-area">
          <div class="search-box">
            <input type="text" v-model="searchQuery" placeholder="搜索文件..." class="search-input" />
            <button class="search-btn">搜索</button>
          </div>
        </div>
      </div>
      
      <!-- 文件列表 -->
      <div class="files-content">
        <div class="files-table">
          <div class="file-row" v-for="file in paginatedFiles" :key="file.id">
            <div class="file-name" @click="viewFile(file)">{{ file.name }}</div>
            <div class="file-actions">
              <button class="download-btn" @click="downloadFile(file)">下载</button>
            </div>
          </div>
          
          <!-- 无结果显示 -->
          <div class="no-results" v-if="filteredFiles.length === 0">
            <div class="no-results-icon">🔍</div>
            <p>未找到匹配的文件</p>
          </div>
        </div>
        
        <!-- 分页控件 -->
        <div class="pagination" v-if="totalPages > 1">
          <button 
            class="page-btn prev" 
            :disabled="currentPage === 1"
            @click="currentPage--">
            上一页
          </button>
          
          <div class="page-numbers">
            <div 
              v-for="page in displayedPages" 
              :key="page"
              :class="['page-number', { active: currentPage === page }]"
              @click="currentPage = page">
              {{ page }}
            </div>
          </div>
          
          <button 
            class="page-btn next" 
            :disabled="currentPage === totalPages"
            @click="currentPage++">
            下一页
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { API_BASE_URL } from '@/config';
import { ElMessage } from 'element-plus'; // 导入提示组件

const router = useRouter();

// 返回欢迎页面
const goToWelcome = () => {
  router.push('/');
};

// 搜索和过滤状态
const searchQuery = ref('');
const selectedCategory = ref('');

// 分页状态
const currentPage = ref(1);
const filesPerPage = 10; // 每页显示的文件数量
const totalPages = ref(1);
const totalFiles = ref(0);

// 文件列表数据
const fileList = ref([]);

// 获取文件列表
const fetchFiles = async () => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      ElMessage.error('登录已过期，请重新登录');
      router.push('/login');
      return;
    }

    const response = await fetch(`${API_BASE_URL}/emergency_files/?page=${currentPage.value}&page_size=${filesPerPage}&search=${searchQuery.value}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.status === 401) {
      ElMessage.error('登录已过期，请重新登录');
      // 清除本地存储的无效token
      localStorage.removeItem('token');
      localStorage.removeItem('mobile');
      router.push('/login');
      return;
    }
    
    if (!response.ok) {
      throw new Error('获取文件列表失败');
    }
    
    const data = await response.json();
    if (data.code === 200) {
      fileList.value = data.data;
      totalPages.value = data.total_pages;
      totalFiles.value = data.total;
    } else {
      throw new Error(data.message);
    }
  } catch (error) {
    console.error('获取文件列表失败:', error);
    ElMessage.error(error.message || '获取文件列表失败');
  }
};

// 下载文件
const downloadFile = async (file) => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      ElMessage.error('登录已过期，请重新登录');
      router.push('/login');
      return;
    }

    const response = await fetch(`${API_BASE_URL}/emergency_files/download/${file.id}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.status === 401) {
      ElMessage.error('登录已过期，请重新登录');
      // 清除本地存储的无效token
      localStorage.removeItem('token');
      localStorage.removeItem('mobile');
      router.push('/login');
      return;
    }
    
    if (!response.ok) {
      throw new Error('下载文件失败');
    }
    
    // 获取文件名
    const contentDisposition = response.headers.get('content-disposition');
    const filename = contentDisposition
      ? contentDisposition.split('filename=')[1].replace(/"/g, '')
      : file.name;
    
    // 创建下载链接
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    console.error('下载文件失败:', error);
    // 这里可以添加错误提示
  }
};

// 监听搜索和分页变化
watch([searchQuery, currentPage], () => {
  fetchFiles();
});

// 初始化时获取文件列表
onMounted(() => {
  // 检查token是否存在
  const token = localStorage.getItem('token');
  if (!token) {
    ElMessage.error('请先登录系统');
    router.push('/login');
    return;
  }
  
  fetchFiles();
});

// 获取筛选后的文件列表
const filteredFiles = computed(() => {
  return fileList.value;
});

// 当前页显示的文件
const paginatedFiles = computed(() => {
  return filteredFiles.value;
});

// 显示的页码范围 (始终显示5个页码)
const displayedPages = computed(() => {
  if (totalPages.value <= 5) {
    return Array.from({ length: totalPages.value }, (_, i) => i + 1);
  }
  
  const halfWindow = 2;
  let start = Math.max(currentPage.value - halfWindow, 1);
  let end = Math.min(start + 4, totalPages.value);
  
  // 调整开始值，确保显示5个页码
  if (end - start < 4) {
    start = Math.max(end - 4, 1);
  }
  
  return Array.from({ length: end - start + 1 }, (_, i) => start + i);
});

// 文件操作函数
const viewFile = (file) => {
  // 实现文件预览逻辑，可能打开新窗口或使用预览组件
};
</script>

<style scoped>
.files-container {
  width: 100%;
  height: 100vh;
  display: flex;
  box-sizing: border-box;
  background: #f0f4f8;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

/* 左侧边栏样式 */
.sidebar {
  width: 280px;
  flex-shrink: 0;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #e2e8f0;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.03);
  position: relative;
  transition: all 0.3s ease;
}

.back-button {
  display: flex;
  align-items: center;
  padding: 8px 24px;
  height: 40px;
  border-bottom: 1px solid #e2e8f0;
  background-color: #ffffff;
  color: #1a56db;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-size: 13px;
  font-weight: 600;
  position: relative;
  overflow: hidden;
  letter-spacing: 0.3px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.back-button:hover {
  background-color: #f8fafc;
  color: #1e40af;
  transform: translateX(2px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.back-icon {
  font-size: 16px;
  margin-right: 8px;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: rgba(26, 86, 219, 0.1);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.back-button:hover .back-icon {
  background: rgba(26, 86, 219, 0.2);
  transform: translateX(-2px) scale(1.05);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.back-text {
  line-height: 1.4;
  position: relative;
  top: 1px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  text-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}

.category-section {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  background: #ffffff;
}

.sidebar-title {
  font-size: 16px;
  color: #1e293b;
  padding-bottom: 16px;
  margin-bottom: 16px;
  border-bottom: 1px solid #e2e8f0;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.category-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.category-item {
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  color: #475569;
  margin-bottom: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  position: relative;
}

.category-item::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 0;
  background: #1a56db;
  border-radius: 0 2px 2px 0;
  transition: height 0.2s ease;
}

.category-item:hover {
  background: #f1f5f9;
  color: #1a56db;
  transform: translateX(4px);
}

.category-item.active {
  background: #eff6ff;
  color: #1a56db;
  font-weight: 500;
}

.category-item.active::before {
  height: 24px;
}

/* 主内容区样式 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  height: 100%;
}

/* 顶部标题和搜索 */
.content-header {
  padding: 8px 24px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e2e8f0;
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

.page-title {
  font-size: 16px;
  color: #1e293b;
  margin: 0;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.search-area {
  width: 360px;
}

.search-box {
  display: flex;
  width: 100%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border-radius: 8px;
  overflow: hidden;
}

.search-input {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px 0 0 8px;
  font-size: 14px;
  color: #1e293b;
  transition: all 0.2s ease;
  background: #f8fafc;
}

.search-input:focus {
  outline: none;
  border-color: #1a56db;
  box-shadow: 0 0 0 3px rgba(26, 86, 219, 0.1);
  background: #ffffff;
}

.search-btn {
  background: #1a56db;
  color: white;
  border: none;
  padding: 0 20px;
  border-radius: 0 8px 8px 0;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
}

.search-btn:hover {
  background: #1e40af;
}

/* 文件列表样式 */
.files-content {
  flex: 1;
  padding: 12px 24px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  height: calc(100vh - 40px - 40px - 40px - 40px); /* 总高度减去顶部标题、底部分页、边距和分页控件 */
}

.files-table {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  padding-right: 8px;
  height: 100%;
  overflow: hidden;
  min-height: 400px; /* 设置最小高度，确保即使数据少也能保持固定行高 */
}

.file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  height: 36px; /* 固定行高 */
  border-radius: 6px;
  background: #f8fafc;
  transition: all 0.2s ease;
  border: 1px solid #e2e8f0;
}

.file-name {
  font-size: 14px;
  color: #1a56db;
  cursor: pointer;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 500;
  padding-right: 12px;
}

.file-name:hover {
  text-decoration: underline;
  color: #1e40af;
}

.file-actions {
  margin-left: 12px;
  display: flex;
  gap: 6px;
}

.download-btn {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  border: none;
  background: #1a56db;
  color: white;
  display: flex;
  align-items: center;
  gap: 4px;
}

.download-btn:hover {
  background: #1e40af;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.no-results {
  text-align: center;
  padding: 48px 0;
  color: #64748b;
  background: #f8fafc;
  border-radius: 10px;
  margin: 20px 0;
}

.no-results-icon {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.8;
}

/* 分页控件 */
.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid #e2e8f0;
}

.page-btn {
  padding: 10px 20px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #475569;
  font-weight: 500;
}

.page-btn:hover:not(:disabled) {
  background: #f1f5f9;
  border-color: #cbd5e1;
  color: #1e293b;
  transform: translateY(-1px);
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-numbers {
  display: flex;
  gap: 8px;
  margin: 0 16px;
}

.page-number {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  color: #475569;
  transition: all 0.2s ease;
  font-weight: 500;
}

.page-number:hover {
  background: #f1f5f9;
  color: #1e293b;
  transform: translateY(-1px);
}

.page-number.active {
  background: #1a56db;
  color: white;
  box-shadow: 0 2px 4px rgba(26, 86, 219, 0.2);
}

/* 响应式布局 */
@media (max-width: 1024px) {
  .sidebar {
    width: 240px;
  }
  
  .content-header {
    padding: 16px 24px;
  }
  
  .search-area {
    width: 300px;
  }
  
  .files-content {
    padding: 16px 24px;
  }
}

@media (max-width: 768px) {
  .files-container {
    height: 100%;
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid #e2e8f0;
  }
  
  .category-section {
    padding: 16px;
  }
  
  .content-header {
    flex-direction: column;
    gap: 16px;
    padding: 6px 12px;
    height: auto;
  }
  
  .search-area {
    width: 100%;
  }
  
  .files-content {
    height: calc(100vh - 40px - 40px - 32px - 40px);
    padding: 8px 16px;
  }
  
  .files-table {
    min-height: 360px; /* 移动端调整最小高度 */
  }
  
  .file-row {
    height: 32px; /* 移动端固定行高 */
    padding: 6px 10px;
  }
  
  .file-name {
    font-size: 13px;
    padding-right: 8px;
  }
  
  .download-btn {
    padding: 3px 6px;
    font-size: 12px;
  }
  
  .back-button {
    padding: 6px 12px;
    height: auto;
  }
  
  .page-title {
    font-size: 15px;
  }
}

@media (max-width: 480px) {
  .page-btn {
    padding: 8px 12px;
    font-size: 13px;
  }
  
  .page-number {
    width: 32px;
    height: 32px;
    font-size: 13px;
  }
}

@media (max-height: 600px) {
  .files-content {
    height: calc(100vh - 32px - 32px - 24px - 32px);
  }
  
  .files-table {
    min-height: 320px; /* 小屏幕设备调整最小高度 */
  }
  
  .file-row {
    height: 28px; /* 小屏幕设备固定行高 */
    padding: 4px 8px;
  }
  
  .file-name {
    font-size: 12px;
  }
  
  .download-btn {
    padding: 2px 6px;
    font-size: 12px;
  }
}
</style>
