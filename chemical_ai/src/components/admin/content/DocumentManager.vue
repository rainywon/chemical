<template>
    <div class="knowledge-files-container">
      <div class="page-header">
        <div class="title-section">
          <el-icon class="title-icon"><Document /></el-icon>
          <h2>安全资料库</h2>
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
        title="上传文件到安全资料库"
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
                  <li><el-icon class="instruction-icon"><InfoFilled /></el-icon> 支持上传PDF和Word文件(.pdf, .doc, .docx)，单个文件不超过20MB</li>
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
                  只能上传PDF和Word文件(.pdf, .doc, .docx)，且不超过20MB
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
              上传到安全资料库
            </el-button>
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

      <!-- 文件预览对话框 -->
      <el-dialog
        v-model="previewDialogVisible"
        title="文件预览"
        width="80%"
        top="8vh"
        class="preview-dialog"
        :append-to-body="true"
        destroy-on-close
        :close-on-click-modal="false"
      >
        <div v-loading="previewLoading" class="preview-container">
          <div v-if="selectedFile" class="file-info">
            <h3>{{ selectedFile.fileName }}</h3>
            <div class="file-meta">
              <div class="meta-item">
                <span class="meta-label">文件类型:</span>
                <el-tag :type="getTagType(selectedFile.fileType)">{{ selectedFile.fileType }}</el-tag>
              </div>
              <div class="meta-item">
                <span class="meta-label">文件大小:</span>
                <span>{{ formatFileSize(selectedFile.fileSize) }}</span>
              </div>
              <div class="meta-item">
                <span class="meta-label">创建时间:</span>
                <span>{{ formatDate(selectedFile.createdTime) }}</span>
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
                {{ selectedFile.message || 'PDF和Word文档内容无法直接预览，请下载后查看' }}
              </el-alert>
            </div>
            <div class="preview-actions">
              <el-button type="primary" :icon="Download" @click="downloadFile(selectedFile)">下载文件</el-button>
            </div>
          </div>
        </div>
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
              <el-option label="PDF(.pdf)" value=".pdf" />
              <el-option label="Word(.doc)" value=".doc" />
              <el-option label="Word(.docx)" value=".docx" />
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
          <h3>安全资料文件列表</h3>
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
          <el-table-column label="上传时间" width="180" prop="lastModified" align="center">
            <template #default="scope">
              <span class="date-time">{{ formatDate(scope.row.lastModified) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="操作" fixed="right" width="240" align="center">
            <template #default="scope">
              <div class="action-buttons">
                <el-button link type="primary" size="small" @click="previewFile(scope.row)">
                  <el-icon><Document /></el-icon> 预览
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
    </div>
  </template>
  
<script setup>
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage, ElLoading, ElMessageBox } from 'element-plus'
import axios from 'axios'
import {
  Delete,
  Download,
  Upload,
  UploadFilled,
  Search,
  Filter,
  Document,
  InfoFilled,
  RefreshLeft,
  RefreshRight
} from '@element-plus/icons-vue'
import { API_BASE_URL } from '../../../config.js'

// 状态管理 - 使用localStorage直接获取adminId
// const userStore = useUserStore()
// const adminId = computed(() => userStore.userId)
const adminId = computed(() => localStorage.getItem('admin_id') || '')

// 文件列表状态
const fileList = ref([])
const loading = ref(false)
const totalFiles = ref(0)

// API 基础路径
const baseApiUrl = API_BASE_URL

// 搜索和筛选状态
const searchQuery = ref('')
const fileTypeFilter = ref('')
const dateRange = ref([])
const sortOption = ref('date-desc')

// 分页状态
const currentPage = ref(1)
const pageSize = ref(20)
const paginatedFiles = computed(() => {
  const startIndex = (currentPage.value - 1) * pageSize.value
  const endIndex = startIndex + pageSize.value
  return filteredFiles.value.slice(startIndex, endIndex)
})

// 计算过滤后的文件列表
const filteredFiles = computed(() => {
  // 应用过滤器
  let result = fileList.value

  // 按文件名搜索
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(file => file.fileName.toLowerCase().includes(query))
  }

  // 按文件类型过滤
  if (fileTypeFilter.value) {
    result = result.filter(file => file.fileType === fileTypeFilter.value)
  }

  // 按日期范围过滤
  if (dateRange.value && dateRange.value.length === 2) {
    const startDate = new Date(dateRange.value[0])
    const endDate = new Date(dateRange.value[1])
    endDate.setHours(23, 59, 59, 999) // 设置到当天结束

    result = result.filter(file => {
      const fileDate = new Date(file.lastModified)
      return fileDate >= startDate && fileDate <= endDate
    })
  }

  // 排序
  result = sortFileArray(result, sortOption.value)

  return result
})

// 上传文件状态
const uploadDialogVisible = ref(false)
const selectedFiles = ref([])

// 多选状态
const multipleSelection = ref([])

// 预览状态
const previewDialogVisible = ref(false)
const previewLoading = ref(false)
const selectedFile = ref(null)

// 批量删除状态
const batchDeleteDialogVisible = ref(false)

// 初始化
onMounted(() => {
  fetchFileList()
})

// 监听筛选条件变化，重置到第一页
watch([searchQuery, fileTypeFilter, dateRange, sortOption], () => {
  currentPage.value = 1
})

// 文件列表相关函数
async function fetchFileList() {
  loading.value = true
  try {
    const response = await axios.get(`${baseApiUrl}/admin/content/safety-documents`, {
      params: {
        search_query: searchQuery.value,
        file_type: fileTypeFilter.value,
        start_date: dateRange.value?.[0] || '',
        end_date: dateRange.value?.[1] || '',
        sort_by: sortOption.value,
        page: currentPage.value,
        page_size: pageSize.value
      },
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    
    // 检查接口返回的数据结构
    if (response.data.success && response.data.data) {
      fileList.value = response.data.data.files.map(file => ({
        ...file,
        fileType: getFileExtension(file.fileName)
      }))
      
      // 如果后端返回了总数，使用后端的值
      if (response.data.data.total) {
        totalFiles.value = response.data.data.total
      } else {
        totalFiles.value = fileList.value.length
      }
    } else {
      // 后端可能返回了旧格式
      if (response.data.files) {
        fileList.value = response.data.files.map(file => ({
          ...file,
          fileType: getFileExtension(file.fileName)
        }))
        
        // 更新总数
        if (response.data.total) {
          totalFiles.value = response.data.total
        } else {
          totalFiles.value = fileList.value.length
        }
      } else {
        ElMessage.error('获取文件列表失败：服务器返回格式错误')
        console.error('服务器返回格式错误:', response.data)
      }
    }
    
  } catch (error) {
    console.error('获取文件列表失败:', error)
    ElMessage.error('获取文件列表失败，请重试')
  } finally {
    loading.value = false
  }
}

// 搜索和筛选函数
function handleSearch() {
  filterFiles()
}

// 防抖处理搜索输入
const searchTimeout = ref(null)
function handleSearchInput() {
  if (searchTimeout.value) {
    clearTimeout(searchTimeout.value)
  }
  searchTimeout.value = setTimeout(() => {
    filterFiles()
  }, 500)
}

function filterFiles() {
  fetchFileList()
}

function sortFiles() {
  fetchFileList()
}

function sortFileArray(files, sortOption) {
  return [...files].sort((a, b) => {
    const [property, direction] = sortOption.split('-')
    const multiplier = direction === 'asc' ? 1 : -1
    
    if (property === 'name') {
      return multiplier * a.fileName.localeCompare(b.fileName)
    } else if (property === 'size') {
      return multiplier * (a.fileSize - b.fileSize)
    } else if (property === 'date') {
      return multiplier * (new Date(a.lastModified) - new Date(b.lastModified))
    }
    return 0
  })
}

function refreshFileList() {
  fetchFileList()
}

// 文件预览
async function previewFile(file) {
  selectedFile.value = file
  previewDialogVisible.value = true
  previewLoading.value = true
  
  try {
    // 获取文件信息
    const response = await axios.get(`${baseApiUrl}/admin/content/safety-documents/preview/${encodeURIComponent(file.fileName)}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    
    // 更新文件信息
    selectedFile.value = {
      ...selectedFile.value,
      ...response.data
    }
    
  } catch (error) {
    console.error('获取文件预览信息失败:', error)
    ElMessage.error('获取文件预览信息失败，请重试')
  } finally {
    previewLoading.value = false
  }
}

// 文件下载
function downloadFile(file) {
  if (!file) return
  
  // 创建带有授权头的URL
  const downloadUrl = `${baseApiUrl}/admin/content/safety-documents/download/${encodeURIComponent(file.fileName)}`
  
  // 使用Fetch API进行下载，以便添加授权头
  fetch(downloadUrl, {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('token')}`
    }
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('下载失败')
    }
    return response.blob()
  })
  .then(blob => {
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = file.fileName
    document.body.appendChild(link)
    link.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(link)
    ElMessage.success(`正在下载: ${file.fileName}`)
  })
  .catch(error => {
    console.error('下载文件失败:', error)
    ElMessage.error('下载文件失败，请重试')
  })
}

// 文件删除
async function deleteFile(file) {
  try {
    const response = await axios.delete(`${baseApiUrl}/admin/content/safety-documents/${encodeURIComponent(file.fileName)}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    
    ElMessage.success(response.data.message || '文件删除成功')
    fetchFileList()
  } catch (error) {
    console.error('删除文件失败:', error)
    ElMessage.error('删除文件失败，请重试')
  }
}

// 批量删除相关函数
function handleBatchDelete() {
  if (multipleSelection.value.length === 0) {
    ElMessage.warning('请先选择要删除的文件')
    return
  }
  
  batchDeleteDialogVisible.value = true
}

async function confirmBatchDelete() {
  try {
    // 提取文件名列表
    const fileNames = multipleSelection.value.map(file => file.id || file.fileName)
    
    const response = await axios.post(`${baseApiUrl}/admin/content/safety-documents/batch-delete`, {
      file_ids: fileNames
    }, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    
    ElMessage.success(response.data.message || `成功删除 ${fileNames.length} 个文件`)
    
    // 清除选择并刷新列表
    multipleSelection.value = []
    batchDeleteDialogVisible.value = false
    fetchFileList()
  } catch (error) {
    console.error('批量删除文件失败:', error)
    ElMessage.error('批量删除文件失败，请重试')
  }
}

// 文件上传相关函数
function handleFileChange(file, fileList) {
  // 验证文件类型
  const allowedTypes = ['.pdf', '.doc', '.docx']
  const fileExt = getFileExtension(file.name).toLowerCase()
  
  if (!allowedTypes.includes(fileExt)) {
    ElMessage.error(`不支持的文件类型: ${fileExt}，只支持 PDF 和 Word 文档`)
    // 从文件列表中移除无效文件
    const index = fileList.findIndex(f => f.uid === file.uid)
    if (index !== -1) {
      fileList.splice(index, 1)
    }
    return
  }
  
  // 验证文件大小 (20MB = 20 * 1024 * 1024)
  const maxSize = 20 * 1024 * 1024
  if (file.size > maxSize) {
    ElMessage.error(`文件过大: ${file.name}，文件大小不能超过 20MB`)
    // 从文件列表中移除过大文件
    const index = fileList.findIndex(f => f.uid === file.uid)
    if (index !== -1) {
      fileList.splice(index, 1)
    }
    return
  }
  
  // 更新选择的文件列表
  selectedFiles.value = fileList.map(f => f.raw || f)
}

function removeSelectedFile(index) {
  selectedFiles.value.splice(index, 1)
}

async function uploadSelectedFiles() {
  if (selectedFiles.value.length === 0) {
    ElMessage.warning('请先选择要上传的文件')
    return
  }
  
  const formData = new FormData()
  
  // 添加所有选中的文件
  selectedFiles.value.forEach(file => {
    formData.append('files', file)
  })
  
  // 创建加载指示器
  const loadingInstance = ElLoading.service({
    lock: true,
    text: '正在上传文件...',
    background: 'rgba(0, 0, 0, 0.7)'
  })
  
  try {
    const response = await axios.post(`${baseApiUrl}/admin/content/safety-documents/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    
    ElMessage.success(response.data.message || '文件上传成功')
    
    // 关闭对话框并刷新文件列表
    uploadDialogVisible.value = false
    selectedFiles.value = []
    fetchFileList()
  } catch (error) {
    console.error('上传文件失败:', error)
    ElMessage.error(error.response?.data?.detail || '上传文件失败，请重试')
  } finally {
    loadingInstance.close()
  }
}

// 表格多选相关函数
function handleSelectionChange(selection) {
  multipleSelection.value = selection
}

// 分页相关函数
function handleSizeChange(size) {
  pageSize.value = size
  fetchFileList()
}

function handleCurrentChange(page) {
  currentPage.value = page
  fetchFileList()
}

// 重置筛选条件
function resetFilters() {
  searchQuery.value = ''
  fileTypeFilter.value = ''
  dateRange.value = []
  sortOption.value = 'date-desc'
  currentPage.value = 1
  fetchFileList()
}

// 工具函数
function formatDate(dateString) {
  if (!dateString) return '-'
  const date = new Date(dateString)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function formatFileSize(size) {
  if (!size) return '0 B'
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let unitIndex = 0
  let fileSize = size
  
  while (fileSize >= 1024 && unitIndex < units.length - 1) {
    fileSize /= 1024
    unitIndex++
  }
  
  return `${fileSize.toFixed(2)} ${units[unitIndex]}`
}

function getFileExtension(fileName) {
  if (!fileName) return ''
  const lastDotIndex = fileName.lastIndexOf('.')
  return lastDotIndex === -1 ? '' : fileName.substring(lastDotIndex)
}

function getFileIconColor(fileType) {
  if (!fileType) return '#909399'
  
  const typeMap = {
    '.pdf': '#FF5252',   // 红色
    '.doc': '#2196F3',   // 蓝色
    '.docx': '#2196F3',  // 蓝色
  }
  
  return typeMap[fileType.toLowerCase()] || '#909399'
}

function getTagType(fileType) {
  if (!fileType) return ''
  
  const typeMap = {
    '.pdf': 'danger',
    '.doc': 'primary',
    '.docx': 'primary',
  }
  
  return typeMap[fileType.toLowerCase()] || 'info'
}
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

.preview-dialog {
  display: flex;
  align-items: center;
  justify-content: center;
}

:deep(.preview-dialog .el-dialog) {
  margin: 0;
  margin-top: 8vh !important;
  max-height: 85vh;
  display: flex !important;
  flex-direction: column !important;
}

:deep(.preview-dialog .el-dialog__body) {
  flex: 1;
  overflow: auto;
  padding: 20px !important;
}

.preview-container {
  min-height: 300px;
  max-height: calc(80vh - 180px);
  overflow-y: auto;
  padding: 10px;
}

.file-info {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.file-info h3 {
  font-size: 18px;
  font-weight: 600;
  margin: 0;
  word-break: break-word;
}

.file-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  padding: 15px;
  background-color: #f9fafc;
  border-radius: 8px;
  border: 1px solid #ebeef5;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.meta-label {
  font-weight: 600;
  color: #606266;
}

.preview-message {
  margin: 15px 0;
}

.preview-actions {
  display: flex;
  justify-content: center;
  margin-top: 20px;
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
  
  