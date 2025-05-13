<template>
  <div class="knowledge-files-container">
    <div class="page-header">
      <div class="title-section">
        <el-icon class="title-icon"><FolderOpened /></el-icon>
        <h2>知识库文件管理</h2>
      </div>
      <div class="header-actions">
        <el-button 
          type="primary" 
          :icon="Upload" 
          class="action-button" 
          @click="uploadDialogVisible = true"
        >
          上传文件
        </el-button>
        <el-button 
          type="danger" 
          :icon="Delete" 
          :disabled="!multipleSelection.length" 
          @click="handleBatchDelete"
          class="action-button"
        >
          批量删除
        </el-button>
      </div>
    </div>

    <!-- 文件上传对话框 -->
    <el-dialog
      v-model="uploadDialogVisible"
      title="上传文件到知识库"
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
            title="文件上传说明"
            type="info"
            :closable="false"
          >
            <div class="upload-instructions">
              <ul class="instruction-list">
                <li><el-icon class="instruction-icon"><InfoFilled /></el-icon> 支持上传Excel文件(.xlsx, .xls)，单个文件不超过10MB</li>
                <li><el-icon class="instruction-icon"><InfoFilled /></el-icon> 最多可选择5个文件同时上传</li>
                <li><el-icon class="instruction-icon"><InfoFilled /></el-icon> 上传前请检查文件格式是否正确</li>
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
                只能上传Excel文件(.xlsx, .xls)，且不超过10MB
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
            上传到知识库
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 搜索和筛选区域 -->
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
            <el-option label="Excel(.xlsx)" value=".xlsx" />
            <el-option label="Excel(.xls)" value=".xls" />
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
        <h3>知识库文件列表</h3>
        <div class="table-stats">共 {{ totalFiles }} 个文件</div>
      </div>
      
      <el-table
        v-loading="loading"
        :data="paginatedFiles"
        style="width: 100%"
        @selection-change="handleSelectionChange"
        :border="true"
        :stripe="true"
        highlight-current-row
        row-class-name="file-table-row"
        header-row-class-name="file-table-header"
      >
        <el-table-column type="selection" width="55" align="center" />
        
        <el-table-column prop="fileName" label="文件名" min-width="250" show-overflow-tooltip>
          <template #default="scope">
            <div class="file-name-cell">
              <el-icon class="file-icon" :style="{ color: getFileIconColor(scope.row.fileType) }">
                <Document />
              </el-icon>
              <span class="file-name">{{ scope.row.fileName }}</span>
            </div>
          </template>
        </el-table-column>
        
        <el-table-column prop="fileType" label="类型" width="120" align="center">
          <template #default="scope">
            <el-tag 
              :type="scope.row.fileType === '.xlsx' ? 'success' : 'primary'"
              effect="light"
              size="small"
              class="file-type-tag"
            >
              {{ scope.row.fileType.toUpperCase() }}
            </el-tag>
          </template>
        </el-table-column>
        
        <el-table-column prop="fileSize" label="大小" width="120" align="center">
          <template #default="scope">
            <span class="file-size">{{ formatFileSize(scope.row.fileSize) }}</span>
          </template>
        </el-table-column>
        
        <el-table-column prop="createdTime" label="创建时间" width="180" sortable align="center">
          <template #default="scope">
            <span class="date-time">{{ formatDate(scope.row.createdTime) }}</span>
          </template>
        </el-table-column>
        
        <el-table-column prop="lastModified" label="最后修改" width="180" sortable align="center">
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
          :total="totalFiles"
          layout="total, sizes, prev, pager, next, jumper"
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
      width="60%"
      append-to-body
    >
      <div v-loading="previewLoading" class="preview-container">
        <template v-if="!previewLoading && selectedFile">
          <div class="preview-header">
            <h3>{{ selectedFile.fileName }}</h3>
            <div class="preview-info">
              <span>类型: {{ selectedFile.fileType.toUpperCase() }}</span>
              <span>大小: {{ formatFileSize(selectedFile.fileSize) }}</span>
              <span>修改时间: {{ formatDate(selectedFile.lastModified) }}</span>
            </div>
          </div>
          
          <div class="preview-content">
            <!-- 这里展示Excel预览内容，实际集成时可能需要专门的Excel预览组件 -->
            <div class="excel-preview-placeholder">
              <el-table :data="previewData" border style="width: 100%">
                <el-table-column v-for="(col, index) in previewColumns" :key="index" :prop="col.prop" :label="col.label" />
              </el-table>
              <div class="preview-note" v-if="previewData.length">
                注: 仅显示前5行数据，完整内容请下载查看
              </div>
              <el-empty v-else description="预览内容暂不可用" />
            </div>
          </div>
        </template>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="previewDialogVisible = false">关闭</el-button>
          <el-button type="primary" @click="downloadFile(selectedFile)">下载文件</el-button>
        </span>
      </template>
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

// 文件预览数据
const previewData = ref([]);
const previewColumns = ref([]);

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
    const response = await axios.get(`${baseApiUrl}/admin/content/knowledge-files`, { 
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
    const response = await axios.post(`${baseApiUrl}/admin/content/knowledge-files/upload`, formData, {
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
  // 通过重新加载文件列表来应用搜索条件
  loadFileList();
};

const handleSearchInput = () => {
  // 防抖处理，可以在这里添加
  handleSearch();
};

// 筛选文件
const filterFiles = () => {
  // 通过重新加载文件列表来应用筛选条件
  loadFileList();
};

// 排序文件
const sortFiles = () => {
  // 通过重新加载文件列表来应用排序条件
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
    
    // 构建查询参数
    const params = { max_rows: 5 };
    
    // 获取文件预览内容
    const response = await axios.get(`${baseApiUrl}/admin/content/knowledge-files/preview/${file.fileName}`, {
      params,
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.data.success) {
      previewColumns.value = response.data.columns;
      previewData.value = response.data.data;
    } else {
      ElMessage.warning(response.data.message || '无法预览文件');
      previewData.value = [];
      previewColumns.value = [];
    }
  } catch (error) {
    console.error('预览文件出错:', error);
    ElMessage.error(`预览文件失败: ${error.response?.data?.detail || error.message}`);
    previewData.value = [];
    previewColumns.value = [];
  } finally {
    previewLoading.value = false;
  }
};

// 下载文件
const downloadFile = (file) => {
  try {
    // 获取token
    const token = localStorage.getItem('token');
    
    // 使用XMLHttpRequest进行下载，以便设置Authorization请求头
    const xhr = new XMLHttpRequest();
    xhr.open('GET', `${baseApiUrl}/admin/content/knowledge-files/download/${file.fileName}`, true);
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
    const response = await axios.delete(`${baseApiUrl}/admin/content/knowledge-files/${file.fileName}`, {
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
    const response = await axios.post(`${baseApiUrl}/admin/content/knowledge-files/batch-delete`, requestData, {
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
  switch(fileType) {
    case '.xlsx':
      return '#1f883d'; // 绿色
    case '.xls':
      return '#1a73e8'; // 蓝色
    default:
      return '#606266'; // 默认灰色
  }
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

.filter-select {
  width: 100%;
}

.search-input {
  width: 100%;
}

.date-picker {
  width: 100%;
}

.refresh-button {
  height: 40px;
  margin-bottom: 10px;
  flex: 0 0 auto;
}

.table-card {
  margin-bottom: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
  border-radius: 8px;
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
  color: #303133;
  font-weight: 600;
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

.preview-header {
  margin-bottom: 20px;
}

.preview-header h3 {
  margin: 0 0 10px 0;
}

.preview-info {
  display: flex;
  gap: 20px;
  font-size: 14px;
  color: #666;
}

.excel-preview-placeholder {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 20px;
  background-color: #f8f9fa;
}

.preview-note {
  margin-top: 10px;
  font-size: 12px;
  color: #999;
  text-align: center;
}

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

:deep(.el-dialog__wrapper) {
  position: fixed !important;
  top: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  left: 0 !important;
  overflow: auto !important;
  margin: 0 !important;
  z-index: 3000 !important;
}

:deep(.el-dialog) {
  position: relative !important;
  margin: 15vh auto 50px !important;
  z-index: 3001 !important;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1) !important;
}

:deep(.el-upload-dragger:focus),
:deep(.el-upload-dragger:active) {
  outline: none !important;
  border-color: #409eff !important;
}

:deep(.el-button:focus),
:deep(.el-button:active) {
  outline: none !important;
  box-shadow: none !important;
}

:deep(.file-uploader .el-upload-list) {
  max-height: 0;
  overflow: hidden;
}

:deep(.selected-files .el-scrollbar__wrap) {
  overflow-x: hidden;
}

/* 添加files-scrollbar样式 */
.files-scrollbar {
  flex: 1;
  overflow: hidden;
}

:deep(.el-upload__tip) {
  width: 100%;
  box-sizing: border-box;
  word-break: break-word;
}

:deep(.date-picker) {
  max-width: 100%;
  box-sizing: border-box;
}

:deep(.el-table) {
  width: 100% !important;
  table-layout: fixed;
}

:deep(.el-table__body) {
  width: 100% !important;
}

:deep(.el-table .cell) {
  word-break: break-word;
}

/* 添加响应式布局支持 */
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
  
  .search-input,
  .filter-select,
  .date-picker {
    width: 100%;
    max-width: 250px;
  }
  
  .filter-container {
    flex-direction: column;
    align-items: stretch;
  }

  .filter-item {
    max-width: 100%;
  }
  
  .refresh-button {
    width: 100%;
    margin-top: 10px;
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
</style>
