<template>
  <div class="knowledge-files-container">
    <div class="page-header">
      <div class="title-section">
        <el-icon class="title-icon"><FolderOpened /></el-icon>
        <h2>事故案例库</h2>
      </div>
      <div class="header-actions">
        <el-button type="primary" class="action-button" @click="uploadDialogVisible = true">
          <el-icon><UploadFilled /></el-icon>
          上传事故案例
        </el-button>
        <el-button type="danger" class="action-button" @click="handleBatchDelete" :disabled="multipleSelection.length === 0">
          <el-icon><Delete /></el-icon>
          批量删除
        </el-button>
      </div>
    </div>

    <!-- 文件上传对话框 -->
    <el-dialog
      v-model="uploadDialogVisible"
      title="上传事故案例文件"
      width="60%"
      :close-on-click-modal="false"
      class="upload-file-dialog"
      top="5vh"
      :fullscreen="false"
      append-to-body
    >
      <div class="upload-dialog-content">
        <div class="upload-explanation">
          <el-alert
            title="事故案例文件上传说明"
            type="info"
            :closable="false"
          >
            <div class="upload-instructions">
              <ul class="instruction-list">
                <li><el-icon class="instruction-icon"><InfoFilled /></el-icon> 支持上传的文件格式：PDF, DOC, DOCX</li>
                <li><el-icon class="instruction-icon"><InfoFilled /></el-icon> 单个文件大小不超过50MB</li>
                <li><el-icon class="instruction-icon"><InfoFilled /></el-icon> 文件名应当清晰表明事故案例类型和内容</li>
              </ul>
            </div>
          </el-alert>
        </div>
        
        <div class="upload-area">
          <el-upload
            class="file-uploader"
            action="#"
            :auto-upload="false"
            :on-change="handleFileChange"
            :limit="5"
            multiple
            drag
          >
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              将文件拖到此处，或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                请确保上传的文件格式正确，并且文件大小在限制范围内。
              </div>
            </template>
          </el-upload>
        </div>
        
        <div class="selected-files-container">
          <div v-if="selectedFiles.length" class="selected-files">
            <h4>已选择 {{ selectedFiles.length }} 个文件:</h4>
            <el-scrollbar class="files-scrollbar">
              <el-table :data="selectedFiles" style="width: 100%" size="small">
                <el-table-column prop="name" label="文件名" min-width="120" show-overflow-tooltip></el-table-column>
                <el-table-column label="大小" width="80">
                  <template #default="scope">
                    {{ formatFileSize(scope.row.size) }}
                  </template>
                </el-table-column>
                <el-table-column label="操作" width="60">
                  <template #default="scope">
                    <el-button 
                      type="danger" 
                      link 
                      @click="removeSelectedFile(scope.$index)"
                    >
                      删除
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-scrollbar>
          </div>
          <div v-else class="no-files-selected">
            <el-empty description="暂无选择文件" :image-size="80"></el-empty>
          </div>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="uploadDialogVisible = false">取消</el-button>
          <el-button 
            type="success" 
            :icon="UploadFilled" 
            @click="uploadSelectedFiles" 
            :disabled="!selectedFiles.length"
          >
            上传到事故案例库
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 筛选卡片 -->
    <el-card class="filter-card">
      <div class="filter-header">
        <div class="filter-title">
          <el-icon><Filter /></el-icon>
          <span>筛选条件</span>
        </div>
        <el-button type="info" plain size="small" @click="resetFilters" class="reset-button">
          <el-icon><RefreshLeft /></el-icon> 重置筛选
        </el-button>
      </div>
      <el-divider class="filter-divider" />
      <div class="filter-container">
        <div class="filter-item">
          <div class="filter-label">文件名称</div>
          <el-input
            v-model="searchQuery"
            placeholder="搜索文件名称"
            class="search-input"
            :prefix-icon="Search"
            clearable
            @clear="handleSearch"
            @input="handleSearchInput"
          />
        </div>

        <div class="filter-item">
          <div class="filter-label">文件类型</div>
          <el-select v-model="fileTypeFilter" placeholder="文件类型" clearable @change="filterFiles" class="filter-select">
            <el-option label="全部类型" value="" />
            <el-option label="PDF(.pdf)" value="pdf" />
            <el-option label="Word(.doc)" value="doc" />
            <el-option label="Word(.docx)" value="docx" />
          </el-select>
        </div>

        <div class="filter-item">
          <div class="filter-label">日期范围</div>
          <el-date-picker
            v-model="dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            format="YYYY-MM-DD"
            value-format="YYYY-MM-DD"
            @change="filterFiles"
            class="date-picker"
          />
        </div>

        <div class="filter-item">
          <div class="filter-label">排序方式</div>
          <el-select v-model="sortOption" placeholder="排序方式" @change="sortFiles" class="filter-select">
            <el-option label="按名称升序" value="name-asc" />
            <el-option label="按名称降序" value="name-desc" />
            <el-option label="按大小升序" value="size-asc" />
            <el-option label="按大小降序" value="size-desc" />
            <el-option label="按日期升序" value="date-asc" />
            <el-option label="按日期降序" value="date-desc" />
          </el-select>
        </div>

        <el-button 
          type="primary" 
          :icon="RefreshRight" 
          @click="refreshFileList" 
          class="refresh-button"
        >
          刷新列表
        </el-button>
      </div>
    </el-card>

    <!-- 文件列表表格 -->
    <el-card class="table-card" :body-style="{ padding: '0px' }">
      <div class="table-header">
        <h3>事故案例文件列表</h3>
        <div class="table-stats">共 {{ totalFiles }} 个文件</div>
      </div>
      <el-table
        v-loading="loading"
        :data="paginatedFiles"
        style="width: 100%"
        @selection-change="handleSelectionChange"
        row-key="id"
        class="file-table"
        :border="true"
        :stripe="true"
        highlight-current-row
        row-class-name="file-table-row"
        header-row-class-name="file-table-header"
      >
        <el-table-column type="selection" width="55" align="center" />
        <el-table-column label="文件名" min-width="240" prop="fileName" show-overflow-tooltip>
          <template #default="scope">
            <div class="file-name-cell">
              <el-icon :color="getFileIconColor(scope.row.fileType)" class="file-icon">
                <Document />
              </el-icon>
              <span class="file-name">{{ scope.row.fileName }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="类型" width="120" align="center">
          <template #default="scope">
            <el-tag 
              :type="getTagType(scope.row.fileType)"
              effect="light"
              size="small"
              class="file-type-tag"
            >
              {{ scope.row.fileType.toUpperCase() }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="大小" width="120" prop="fileSize" align="center">
          <template #default="scope">
            <span class="file-size">{{ formatFileSize(scope.row.fileSize) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="修改时间" width="180" prop="lastModified" align="center">
          <template #default="scope">
            <span class="date-time">{{ formatDate(scope.row.lastModified) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" fixed="right" width="240" align="center">
          <template #default="scope">
            <div class="action-buttons">
              <el-button link type="primary" size="small" @click="previewFile(scope.row)">
                <el-icon><View /></el-icon> 预览
              </el-button>
              <el-button link type="success" size="small" @click="downloadFile(scope.row)">
                <el-icon><Download /></el-icon> 下载
              </el-button>
              <el-popconfirm
                title="确定要删除这个文件吗？"
                @confirm="deleteFile(scope.row)"
                confirm-button-text="确定"
                cancel-button-text="取消"
                confirm-button-type="danger"
                icon-color="#ff4949"
              >
                <template #reference>
                  <el-button link type="danger" size="small">
                    <el-icon><Delete /></el-icon> 删除
                  </el-button>
                </template>
              </el-popconfirm>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页控件 -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          :total="totalFiles"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
          background
          class="custom-pagination"
        />
      </div>
    </el-card>

    <!-- 文件预览对话框 -->
    <el-dialog
      v-model="previewDialogVisible"
      title="文件预览"
      width="80%"
      top="5vh"
      class="preview-dialog"
    >
      <div v-loading="previewLoading" class="preview-container">
        <div v-if="selectedFile" class="file-info">
          <h3>{{ selectedFile.fileName }}</h3>
          <div class="file-meta">
            <div class="meta-item">
              <span class="meta-label">文件类型:</span>
              <el-tag :type="getTagType(selectedFile.fileType)">{{ selectedFile.fileType.toUpperCase() }}</el-tag>
            </div>
            <div class="meta-item">
              <span class="meta-label">文件大小:</span>
              <span>{{ formatFileSize(selectedFile.fileSize) }}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">创建时间:</span>
              <span>{{ formatDate(selectedFile.lastModified) }}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">修改时间:</span>
              <span>{{ formatDate(selectedFile.lastModified) }}</span>
            </div>
          </div>
          <div class="preview-message">
            <el-alert
              type="info"
              :closable="false"
              show-icon
            >
              {{ selectedFile.fileType.toLowerCase() === '.pdf' ? 'PDF文件内容预览' : 'Word文档内容无法直接预览，请下载后查看' }}
            </el-alert>
          </div>
          <div class="preview-actions">
            <el-button type="primary" :icon="Download" @click="downloadFile(selectedFile)">下载文件</el-button>
          </div>
          <div v-if="selectedFile.fileType.toLowerCase() === '.pdf' && previewUrl" class="pdf-preview">
            <iframe :src="previewUrl" width="100%" height="500" frameborder="0"></iframe>
          </div>
        </div>
      </div>
    </el-dialog>

    <!-- 批量删除确认对话框 -->
    <el-dialog
      v-model="batchDeleteDialogVisible"
      title="批量删除确认"
      width="30%"
    >
      <div class="batch-delete-content">
        <el-alert
          title="删除操作不可恢复"
          type="warning"
          show-icon
          :closable="false"
        />
        <p class="delete-confirm-text">
          您确定要删除选中的 {{ multipleSelection.length }} 个文件吗？
        </p>
        <ul class="delete-file-list">
          <li v-for="(file, index) in multipleSelection.slice(0, 3)" :key="index">
            {{ file.fileName }}
          </li>
          <li v-if="multipleSelection.length > 3">
            ...等 {{ multipleSelection.length }} 个文件
          </li>
        </ul>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="batchDeleteDialogVisible = false">取消</el-button>
          <el-button type="danger" @click="confirmBatchDelete">确认删除</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';
import { 
  Search, 
  Upload, 
  UploadFilled, 
  Delete, 
  Document, 
  RefreshRight,
  View,
  Download,
  FolderOpened,
  Filter,
  RefreshLeft,
  InfoFilled
} from '@element-plus/icons-vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import axios from 'axios';
import { API_BASE_URL } from '../../../config.js';

// 加载状态
const loading = ref(false);
const previewLoading = ref(false);

// 文件列表状态
const fileList = ref([]);
const filteredFiles = ref([]);
const multipleSelection = ref([]);
const selectedFiles = ref([]);

// 分页状态
const currentPage = ref(1);
const pageSize = ref(20);
const totalFiles = ref(0);

// 搜索和筛选状态
const searchQuery = ref('');
const fileTypeFilter = ref('');
const dateRange = ref([]);
const sortOption = ref('name-asc');

// 对话框状态
const previewDialogVisible = ref(false);
const batchDeleteDialogVisible = ref(false);
const selectedFile = ref(null);
const previewUrl = ref('');

// 文件上传对话框
const uploadDialogVisible = ref(false);

// API 基础路径
const baseApiUrl = API_BASE_URL;

// 根据分页设置计算当前页的文件
const paginatedFiles = computed(() => fileList.value);

// 监听上传对话框状态
watch(uploadDialogVisible, (newVal) => {
  if (newVal === true) {
    // 当对话框打开时清空已选择的文件
    selectedFiles.value = [];
  }
});

// 加载文件列表数据
onMounted(() => {
  loadFileList();
});

// 加载文件列表（从后端API获取）
const loadFileList = async () => {
  loading.value = true;
  
  try {
    // 获取token
    const token = localStorage.getItem('token');
    
    // 构建查询参数
    const params = {
      page: currentPage.value,
      page_size: pageSize.value
    };
    
    // 添加搜索条件
    if (searchQuery.value) {
      params.search_query = searchQuery.value;
    }
    
    // 添加文件类型过滤条件
    if (fileTypeFilter.value) {
      params.file_type = fileTypeFilter.value;
    }
    
    // 添加日期范围过滤条件
    if (dateRange.value && dateRange.value.length === 2) {
      params.start_date = dateRange.value[0];
      params.end_date = dateRange.value[1];
    }
    
    // 添加排序条件
    if (sortOption.value) {
      params.sort_by = sortOption.value;
    }
    
    // 发送请求到后端API
    const response = await axios.get(`${baseApiUrl}/admin/content/emergency-plans`, { 
      params,
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.data.success) {
      fileList.value = response.data.data.files;
      totalFiles.value = response.data.data.total;
    } else {
      ElMessage.error('加载文件列表失败');
    }
  } catch (error) {
    console.error('加载文件列表出错:', error);
    ElMessage.error(`加载文件列表失败: ${error.response?.data?.detail || error.message}`);
  } finally {
    loading.value = false;
  }
};

// 格式化文件大小
const formatFileSize = (size) => {
  if (size < 1024) {
    return size + ' B';
  } else if (size < 1024 * 1024) {
    return (size / 1024).toFixed(2) + ' KB';
  } else if (size < 1024 * 1024 * 1024) {
    return (size / (1024 * 1024)).toFixed(2) + ' MB';
  } else {
    return (size / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }
};

// 格式化日期
const formatDate = (dateString) => {
  const date = new Date(dateString);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
};

// 处理表格选择变化
const handleSelectionChange = (selection) => {
  multipleSelection.value = selection;
};

// 处理文件选择
const handleFileChange = (file, fileList) => {
  selectedFiles.value = fileList;
};

// 从选中的文件列表中移除文件
const removeSelectedFile = (index) => {
  selectedFiles.value.splice(index, 1);
};

// 上传选中的文件
const uploadSelectedFiles = async () => {
  if (selectedFiles.value.length === 0) {
    ElMessage.warning('请先选择要上传的文件');
    return;
  }
  
  try {
    // 获取token
    const token = localStorage.getItem('token');
    
    // 创建FormData对象
    const formData = new FormData();
    selectedFiles.value.forEach(file => {
      formData.append('files', file.raw);
    });
    
    // 发送上传请求
    const response = await axios.post(`${baseApiUrl}/admin/content/emergency-plans/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.data.success) {
      ElMessage.success(response.data.message);
      // 上传成功后刷新文件列表
      loadFileList();
      // 清空选择的文件
      selectedFiles.value = [];
      // 关闭上传对话框
      uploadDialogVisible.value = false;
    } else {
      ElMessage.error(response.data.message || '上传文件失败');
    }
  } catch (error) {
    console.error('上传文件出错:', error);
    ElMessage.error(`上传文件失败: ${error.response?.data?.detail || error.message}`);
  }
};

// 搜索文件
const handleSearch = () => {
  loadFileList();
};

const handleSearchInput = () => {
  handleSearch();
};

// 筛选文件
const filterFiles = () => {
  loadFileList();
};

// 排序文件
const sortFiles = () => {
  loadFileList();
};

// 刷新文件列表
const refreshFileList = () => {
  loadFileList();
  ElMessage.success('文件列表已刷新');
};

// 预览文件
const previewFile = async (file) => {
  selectedFile.value = file;
  previewDialogVisible.value = true;
  previewLoading.value = true;
  
  try {
    // 获取token
    const token = localStorage.getItem('token');
    
    // 只有PDF文件才能直接预览
    if (file.fileType.toLowerCase() === '.pdf') {
      // 构建预览URL
      let url = `${baseApiUrl}/admin/content/emergency-plans/view/${file.fileName}`;
      
      // 添加token参数用于预览认证
      url += `?token=${encodeURIComponent(token)}`;
      
      previewUrl.value = url;
    } else {
      // 非PDF文件不提供预览
      previewUrl.value = '';
    }
  } catch (error) {
    console.error('预览文件出错:', error);
    ElMessage.error(`预览文件失败: ${error.response?.data?.detail || error.message}`);
  } finally {
    previewLoading.value = false;
  }
};

// 下载文件
const downloadFile = (file) => {
  if (!file) return;
  
  try {
    // 获取token
    const token = localStorage.getItem('token');
    
    // 使用XMLHttpRequest进行下载，以便设置Authorization请求头
    const xhr = new XMLHttpRequest();
    xhr.open('GET', `${baseApiUrl}/admin/content/emergency-plans/download/${file.fileName}`, true);
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    
    xhr.onload = function() {
      if (this.status === 200) {
        const blob = new Blob([this.response]);
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = file.fileName;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        ElMessage.success(`开始下载文件: ${file.fileName}`);
      } else {
        ElMessage.error('下载文件失败');
      }
    };
    
    xhr.onerror = function() {
      ElMessage.error('下载文件失败');
    };
    
    xhr.send();
  } catch (error) {
    console.error('下载文件出错:', error);
    ElMessage.error(`下载文件失败: ${error.message}`);
  }
};

// 删除单个文件
const deleteFile = async (file) => {
  try {
    // 获取token
    const token = localStorage.getItem('token');
    
    // 发送删除请求
    const response = await axios.delete(`${baseApiUrl}/admin/content/emergency-plans/${file.fileName}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.data.success) {
      ElMessage.success(response.data.message);
      // 删除成功后刷新文件列表
      loadFileList();
    } else {
      ElMessage.error(response.data.message || '删除文件失败');
    }
  } catch (error) {
    console.error('删除文件出错:', error);
    ElMessage.error(`删除文件失败: ${error.response?.data?.detail || error.message}`);
  }
};

// 处理批量删除
const handleBatchDelete = () => {
  if (multipleSelection.value.length === 0) {
    ElMessage.warning('请先选择要删除的文件');
    return;
  }
  
  batchDeleteDialogVisible.value = true;
};

// 确认批量删除
const confirmBatchDelete = async () => {
  try {
    // 获取token
    const token = localStorage.getItem('token');
    
    // 获取所有选中文件的ID
    const fileIds = multipleSelection.value.map(file => file.id);
    
    // 构建请求数据
    const requestData = {
      file_ids: fileIds
    };
    
    // 发送批量删除请求
    const response = await axios.post(`${baseApiUrl}/admin/content/emergency-plans/batch-delete`, requestData, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.data.success) {
      ElMessage.success(response.data.message);
      // 删除成功后刷新文件列表
      loadFileList();
      // 关闭对话框并清空选择
      batchDeleteDialogVisible.value = false;
      multipleSelection.value = [];
    } else {
      ElMessage.error(response.data.message || '批量删除文件失败');
    }
  } catch (error) {
    console.error('批量删除文件出错:', error);
    ElMessage.error(`批量删除文件失败: ${error.response?.data?.detail || error.message}`);
  }
};

// 分页处理
const handleSizeChange = (size) => {
  pageSize.value = size;
  loadFileList();
};

const handleCurrentChange = (page) => {
  currentPage.value = page;
  loadFileList();
};

// 获取文件图标颜色
const getFileIconColor = (fileType) => {
  if (!fileType) return '#909399';
  
  const typeMap = {
    '.pdf': '#FF5252',   // 红色
    '.doc': '#2196F3',   // 蓝色
    '.docx': '#2196F3',  // 蓝色
  };
  
  return typeMap[fileType.toLowerCase()] || '#909399';
};

// 获取标签类型
const getTagType = (fileType) => {
  if (!fileType) return '';
  
  const typeMap = {
    '.pdf': 'danger',
    '.doc': 'primary',
    '.docx': 'primary',
  };
  
  return typeMap[fileType.toLowerCase()] || 'info';
};

// 重置所有筛选条件
const resetFilters = () => {
  searchQuery.value = '';
  fileTypeFilter.value = '';
  dateRange.value = [];
  sortOption.value = 'name-asc';
  loadFileList();
  ElMessage.success('已重置所有筛选条件');
};
</script>

<style scoped>
/* 确保所有按钮点击时没有黑边 */
:deep(.el-button:focus),
:deep(.el-button:active),
:deep(.el-button:focus-visible) {
  outline: none !important;
  box-shadow: none !important;
  border-color: var(--el-button-hover-border-color);
}

:deep(.el-upload-dragger:focus),
:deep(.el-upload-dragger:active) {
  outline: none !important;
  border-color: #409eff !important;
  box-shadow: none !important;
}

.knowledge-files-container {
  padding: 20px;
  position: relative;
  z-index: 1;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  background-color: #fff;
  padding: 20px 24px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
}

.title-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.title-icon {
  font-size: 24px;
  color: #409eff;
}

.page-header h2 {
  margin: 0;
  font-size: 22px;
  color: #303133;
  font-weight: 600;
}

.header-actions {
  display: flex;
  gap: 16px;
  align-items: center;
}

.file-upload {
  display: flex;
  align-items: center;
}

:deep(.file-upload .el-upload) {
  display: flex;
  align-items: center;
}

.action-button {
  height: 40px;
  font-weight: 500;
  border-radius: 4px;
  padding: 0 16px;
}

:deep(.el-upload__tip) {
  color: #909399;
  font-size: 12px;
  margin-top: 8px;
  line-height: 1.4;
}

.ml-2 {
  margin-left: 8px;
}

.filter-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.filter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background-color: #f9fafc;
  border-bottom: 1px solid #ebeef5;
}

.filter-title {
  display: flex;
  align-items: center;
  gap: 10px;
}

.filter-title span {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}

.reset-button {
  margin-left: auto;
}

.filter-divider {
  margin: 0;
}

.filter-container {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  align-items: flex-end;
  padding: 16px 20px;
  justify-content: space-between;
}

.filter-item {
  display: flex;
  flex-direction: column;
  flex: 1 0 auto;
  min-width: 180px;
  max-width: 250px;
  margin-bottom: 10px;
}

.filter-label {
  font-weight: 600;
  margin-bottom: 5px;
  color: var(--el-text-color-secondary);
}

.search-input {
  width: 100%;
}

.filter-select,
.date-picker {
  width: 100%;
}

.refresh-button {
  height: 40px;
  margin-bottom: 10px;
  flex: 0 0 auto;
}

/* 表格卡片样式 */
.table-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background-color: #f9fafc;
  border-bottom: 1px solid #ebeef5;
}

.table-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #303133;
}

.table-stats {
  color: #606266;
  font-size: 14px;
}

.file-table-header {
  background-color: #f5f7fa !important;
  color: #606266;
  font-weight: 600;
  height: 50px;
}

.file-table-row {
  transition: all 0.3s;
}

.file-table-row:hover {
  background-color: #f0f9ff !important;
  cursor: pointer;
}

.file-name-cell {
  display: flex;
  align-items: center;
  gap: 10px;
}

.file-icon {
  font-size: 18px;
}

.file-name {
  font-weight: 500;
  color: #303133;
  flex-grow: 1;
}

.file-type-tag {
  font-weight: 600;
  padding: 4px 8px;
}

.file-size {
  color: #606266;
  font-family: 'Courier New', monospace;
}

.date-time {
  color: #606266;
  font-size: 13px;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 8px;
}

.pagination-container {
  margin-top: 5px;
  padding: 15px 20px;
  display: flex;
  justify-content: flex-end;
  background-color: #f9fafc;
  border-top: 1px solid #ebeef5;
}

.custom-pagination {
  margin-right: 20px;
}

.preview-container {
  min-height: 300px;
}

.file-info {
  margin-bottom: 20px;
}

.file-info h3 {
  margin: 0 0 10px 0;
  font-size: 18px;
  font-weight: 600;
}

.file-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin-bottom: 15px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.meta-label {
  font-weight: 600;
  color: #606266;
}

.preview-message {
  margin: 20px 0;
}

.preview-actions {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

/* 批量删除对话框样式 */
.batch-delete-content {
  padding: 10px 0;
}

.delete-confirm-text {
  margin: 15px 0;
  font-weight: bold;
}

.delete-file-list {
  padding-left: 20px;
  max-height: 100px;
  overflow-y: auto;
  color: #666;
  font-size: 14px;
}

.upload-dialog-content {
  padding: 20px;
  display: flex;
  flex-direction: column;
  width: 100%;
  box-sizing: border-box;
}

.upload-explanation {
  margin-bottom: 20px;
  width: 100%;
  box-sizing: border-box;
  border-radius: 4px;
  overflow: hidden;
}

.upload-explanation :deep(.el-alert--info) {
  background-color: #f0f6ff;
  border: 1px solid #d1e3ff;
}

.upload-explanation :deep(.el-alert__title) {
  font-weight: 600;
  font-size: 16px;
  color: #1e88e5;
}

.upload-explanation :deep(.el-alert__icon) {
  color: #1e88e5;
  font-size: 18px;
}

.upload-instructions {
  padding: 15px;
  border-radius: 4px;
  font-size: 14px;
  line-height: 1.6;
  margin-top: 10px;
}

.instruction-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.instruction-list li {
  display: flex;
  align-items: flex-start;
  margin-bottom: 10px;
  color: #444;
}

.instruction-list li:last-child {
  margin-bottom: 0;
}

.instruction-icon {
  color: #409EFF;
  margin-right: 8px;
  font-size: 16px;
  flex-shrink: 0;
}

.upload-area {
  margin-bottom: 20px;
  width: 100%;
  box-sizing: border-box;
}

.file-uploader {
  width: 100%;
  box-sizing: border-box;
}

:deep(.file-uploader .el-upload) {
  width: 100%;
}

:deep(.file-uploader .el-upload-dragger) {
  width: 100%;
  height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed #409eff;
  background-color: rgba(64, 158, 255, 0.05);
  transition: all 0.3s;
  border-radius: 8px;
}

:deep(.file-uploader .el-upload-dragger:hover) {
  border-color: #79bbff;
  background-color: rgba(64, 158, 255, 0.1);
  box-shadow: 0 0 10px rgba(64, 158, 255, 0.1);
}

:deep(.file-uploader .el-icon--upload) {
  font-size: 48px;
  color: #409eff;
  margin-bottom: 10px;
}

:deep(.file-uploader .el-upload__text) {
  color: #606266;
  font-size: 16px;
  text-align: center;
  padding: 0 10px;
}

:deep(.file-uploader .el-upload__text em) {
  color: #409eff;
  font-style: normal;
  font-weight: bold;
}

.selected-files-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  overflow: hidden;
  margin-top: 20px;
  min-height: 200px;
  max-height: 300px;
  width: 100%;
  box-sizing: border-box;
}

.selected-files {
  padding: 15px;
  background-color: #f8f9fa;
  height: 100%;
  overflow-y: auto;
  width: 100%;
  box-sizing: border-box;
}

.selected-files h4 {
  margin-top: 0;
  margin-bottom: 15px;
  font-weight: 600;
  color: #303133;
}

.no-files-selected {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  background-color: #f8f9fa;
}

/* 修复对话框滚动条和按钮黑边问题 */
:deep(.el-dialog__body) {
  padding: 0 !important;
  overflow: hidden !important;
  height: auto !important;
  max-height: none !important;
}

:deep(.upload-file-dialog .el-dialog) {
  margin-top: 5vh !important;
  margin-bottom: 5vh !important;
  height: auto !important;
  max-height: 90vh !important;
  display: flex !important;
  flex-direction: column !important;
  width: 80% !important;
  max-width: 900px !important;
}

:deep(.upload-file-dialog .el-dialog__body) {
  flex: 1 !important;
  overflow-y: auto !important;
  max-height: calc(90vh - 120px) !important;
}

@media screen and (max-width: 1200px) {
  :deep(.upload-file-dialog .el-dialog) {
    width: 90% !important;
  }
}

@media screen and (max-width: 768px) {
  :deep(.upload-file-dialog .el-dialog) {
    width: 95% !important;
    margin: 10px auto !important;
  }
  
  :deep(.file-uploader .el-upload-dragger) {
    height: 100px;
  }
  
  :deep(.file-uploader .el-icon--upload) {
    font-size: 36px;
  }

  :deep(.file-uploader .el-upload__text) {
    font-size: 14px;
  }
  
  .instruction-list li {
    align-items: flex-start;
  }
  
  .instruction-icon {
    margin-top: 2px;
  }
  
  .selected-files-container {
    max-height: 250px;
  }
}

@media screen and (min-width: 769px) and (max-width: 1200px) {
  .filter-item {
    min-width: 160px;
  }
  
  .refresh-button {
    margin-top: 0;
  }
}
</style> 