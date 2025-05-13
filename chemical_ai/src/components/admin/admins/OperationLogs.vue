<template>
  <div class="operation-logs-container">
    
    <!-- 搜索过滤区域 -->
    <div class="search-container">
      <el-form :inline="true" :model="searchParams" class="search-form">
        <el-form-item label="操作人ID">
          <el-input v-model="searchParams.adminId" placeholder="请输入管理员ID" clearable></el-input>
        </el-form-item>
        <el-form-item label="操作类型">
          <el-select v-model="searchParams.operationType" placeholder="请选择操作类型" clearable>
            <el-option value="登录" label="登录"></el-option>
            <el-option value="登出" label="登出"></el-option>
            <el-option value="查询" label="查询"></el-option>
            <el-option value="创建" label="创建"></el-option>
            <el-option value="更新" label="更新"></el-option>
            <el-option value="删除" label="删除"></el-option>
            <el-option value="上传文件" label="上传文件"></el-option>
            <el-option value="下载文件" label="下载文件"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="日期范围">
          <el-date-picker
            v-model="dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            format="YYYY-MM-DD"
            value-format="YYYY-MM-DD"
            clearable
          ></el-date-picker>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="searchLogs">
            <i class="el-icon-search"></i> 搜索
          </el-button>
          <el-button @click="resetSearchParams">
            <i class="el-icon-refresh"></i> 重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <!-- 操作日志表格 -->
    <div class="table-container">
      <el-table
        :data="operationLogs"
        border
        stripe
        style="width: 100%"
        v-loading="loading"
        :header-cell-style="{background:'#f5f7fa', color:'#606266', fontWeight: 'bold'}"
        :cell-style="{ padding: '12px 8px', fontSize: '14px', whiteSpace: 'nowrap' }"
      >
        <el-table-column prop="log_id" label="日志ID" min-width="80" align="center"></el-table-column>
        <el-table-column prop="admin_id" label="管理员ID" min-width="100" align="center"></el-table-column>
        <el-table-column prop="operation_type" label="操作类型" min-width="120" align="center">
          <template #default="scope">
            <div class="tag-container">
              <el-tag :type="getOperationTypeTag(scope.row.operation_type)" class="log-tag">
                {{ scope.row.operation_type || '未知' }}
              </el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="operation_desc" label="操作描述" min-width="300" align="left">
          <template #default="scope">
            <span class="ellipsis-cell" :title="scope.row.operation_desc">
              {{ scope.row.operation_desc || '无描述' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="ip_address" label="IP地址" min-width="130" align="center">
          <template #default="scope">
            <span class="ellipsis-cell" :title="scope.row.ip_address">
              {{ scope.row.ip_address || '未知' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="操作时间" min-width="180" align="center">
          <template #default="scope">
            <span class="ellipsis-cell" :title="scope.row.created_at">
              {{ formatDateTime(scope.row.created_at) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="详情" width="80" align="center">
          <template #default="scope">
            <el-button
              size="small"
              type="primary"
              @click="showLogDetail(scope.row)"
            >
              详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <!-- 分页 -->
      <div class="pagination-container">
        <el-pagination
          background
          layout="total, sizes, prev, pager, next, jumper"
          :total="totalItems"
          :page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :current-page="currentPage"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        ></el-pagination>
      </div>
    </div>

    <!-- 日志详情对话框 -->
    <el-dialog 
      title="日志详情" 
      v-model="detailDialogVisible" 
      width="600px" 
      :close-on-click-modal="false"
    >
      <div v-if="selectedLog" class="log-detail">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="日志ID">{{ selectedLog.log_id }}</el-descriptions-item>
          <el-descriptions-item label="管理员ID">{{ selectedLog.admin_id }}</el-descriptions-item>
          <el-descriptions-item label="操作类型">
            <el-tag :type="getOperationTypeTag(selectedLog.operation_type)">
              {{ selectedLog.operation_type }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="操作描述">{{ selectedLog.operation_desc }}</el-descriptions-item>
          <el-descriptions-item label="IP地址">{{ selectedLog.ip_address }}</el-descriptions-item>
          <el-descriptions-item label="用户代理">{{ selectedLog.user_agent || '未记录' }}</el-descriptions-item>
          <el-descriptions-item label="操作时间">{{ formatDateTime(selectedLog.created_at) }}</el-descriptions-item>
        </el-descriptions>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="detailDialogVisible = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed, watch, nextTick } from 'vue';
import { ElMessage } from 'element-plus';
import { API_BASE_URL } from '../../../config';

// 状态变量
const loading = ref(false);
const operationLogs = ref([]);
const totalItems = ref(0);
const currentPage = ref(1);
const pageSize = ref(10);
const detailDialogVisible = ref(false);
const selectedLog = ref(null);
const dateRange = ref([]);

// 日期格式化函数
const formatDateTime = (dateTimeStr) => {
  if (!dateTimeStr) return '未记录';
  
  const date = new Date(dateTimeStr);
  if (isNaN(date.getTime())) return dateTimeStr;
  
  return `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
};

// API基础URL，确保没有尾斜杠
const apiBaseUrl = computed(() => {
  if (!API_BASE_URL) return '';
  return API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
});

// 搜索参数
const searchParams = reactive({
  adminId: '',
  operationType: '',
  startDate: '',
  endDate: ''
});

// 监听日期范围变化，更新搜索参数
watch(dateRange, (newValue) => {
  if (newValue && Array.isArray(newValue) && newValue.length === 2) {
    searchParams.startDate = newValue[0];
    searchParams.endDate = newValue[1];
  } else {
    searchParams.startDate = '';
    searchParams.endDate = '';
  }
});

// 获取操作日志记录
const fetchOperationLogs = async () => {
  if (!apiBaseUrl.value) {
    ElMessage.error('API基础URL未配置');
    return;
  }
  
  loading.value = true;
  try {
    // 构建查询参数
    const params = new URLSearchParams();
    params.append('page', currentPage.value.toString());
    params.append('page_size', pageSize.value.toString());
    
    if (searchParams.adminId && searchParams.adminId.trim()) {
      const adminIdNum = parseInt(searchParams.adminId.trim());
      if (!isNaN(adminIdNum)) {
        params.append('admin_id', adminIdNum.toString());
      }
    }
    
    if (searchParams.operationType && searchParams.operationType.trim()) {
      params.append('operation_type', searchParams.operationType.trim());
    }
    
    if (searchParams.startDate && searchParams.endDate) {
      params.append('start_date', searchParams.startDate);
      params.append('end_date', searchParams.endDate);
    }
    
    // 获取认证 token
    const token = localStorage.getItem('token');
    if (!token) {
      ElMessage.error('您的登录已过期，请重新登录');
      loading.value = false;
      return;
    }
    
    // 发送请求
    const response = await fetch(`${apiBaseUrl.value}/admin/operation-logs?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
    
    // 检查响应状态
    if (!response.ok) {
      const errorText = await response.text();
      console.error('API错误响应:', response.status, errorText);
      throw new Error(`API请求失败: ${response.status} ${errorText}`);
    }
    
    const data = await response.json();
    
    if (data.code === 200) {
      operationLogs.value = data.data.logs;
      totalItems.value = data.data.total;
    } else {
      ElMessage.error(data.message || '获取操作日志失败');
    }
  } catch (error) {
    console.error('获取操作日志失败:', error);
    ElMessage.error('网络连接异常，请稍后再试');
  } finally {
    loading.value = false;
  }
};

// 根据操作类型返回不同的标签类型
const getOperationTypeTag = (type) => {
  if (!type) return 'info';
  
  const typeMap = {
    '登录': 'success',
    '登出': 'info',
    '查询': 'primary',
    '创建': 'success',
    '更新': 'warning',
    '删除': 'danger'
  };
  
  return typeMap[type] || 'info';
};

// 搜索日志
const searchLogs = () => {
  currentPage.value = 1;
  fetchOperationLogs();
};

// 重置搜索参数
const resetSearchParams = () => {
  Object.keys(searchParams).forEach(key => {
    searchParams[key] = '';
  });
  dateRange.value = [];
  currentPage.value = 1;
  fetchOperationLogs();
};

// 处理分页大小变化
const handleSizeChange = (size) => {
  pageSize.value = size;
  fetchOperationLogs();
};

// 处理当前页变化
const handleCurrentChange = (page) => {
  currentPage.value = page;
  fetchOperationLogs();
};

// 显示日志详情
const showLogDetail = async (log) => {
  try {
    // 获取认证 token
    const token = localStorage.getItem('token');
    if (!token) {
      ElMessage.error('您的登录已过期，请重新登录');
      return;
    }
    
    // 查询日志详情
    const response = await fetch(`${apiBaseUrl.value}/admin/operation-logs/${log.log_id}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
    
    // 检查响应状态
    if (!response.ok) {
      if (response.status === 403) {
        ElMessage.error('权限不足，仅管理员可查看操作日志详情');
        return;
      }
      const errorText = await response.text();
      throw new Error(`API请求失败: ${response.status} ${errorText}`);
    }
    
    const data = await response.json();
    
    if (data.code === 200) {
      selectedLog.value = data.data;
      detailDialogVisible.value = true;
    } else if (data.code === 403) {
      ElMessage.error(data.message || '权限不足，仅管理员可查看操作日志详情');
    } else if (data.code === 404) {
      ElMessage.error('日志不存在');
    } else {
      ElMessage.error(data.message || '获取日志详情失败');
    }
  } catch (error) {
    console.error('获取日志详情失败:', error);
    ElMessage.error('网络连接异常，请稍后再试');
  }
};

// 生命周期钩子
onMounted(() => {

  
  // 加载操作日志记录
  fetchOperationLogs();
});
</script>

<style scoped>
.operation-logs-container {
  padding: 20px;
}

.page-title {
  margin-bottom: 24px;
  font-size: 24px;
  font-weight: 500;
  color: #333;
}

.search-container {
  background-color: #fff;
  padding: 20px;
  border-radius: 4px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.table-container {
  background-color: #fff;
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

/* 添加表格单元格文本省略样式 */
.ellipsis-cell {
  display: block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 鼠标悬停时显示完整内容的提示 */
.ellipsis-cell:hover {
  cursor: pointer;
}

/* 标签容器样式 */
.tag-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}

/* 操作类型标签样式 */
.log-tag {
  min-width: 60px;
  text-align: center;
  padding: 2px 10px;
  font-weight: 500;
  letter-spacing: 1px;
  border-radius: 12px;
  white-space: nowrap;
  font-size: 13px;
}

.pagination-container {
  margin-top: 20px;
  text-align: right;
}

.log-detail {
  padding: 10px;
}

.operation-desc {
  line-height: 1.5;
  word-break: break-all;
}

/* 修复按钮点击后出现的黑色边框 */
:deep(.el-button:focus),
:deep(.el-button:focus-visible) {
  outline: none !important;
  box-shadow: none !important;
  border-color: initial;
}

/* 改善表格内容样式 */
:deep(.el-table__header-wrapper th) {
  font-weight: bold;
}

:deep(.el-pagination) {
  margin-top: 20px;
  justify-content: flex-end;
}
</style> 