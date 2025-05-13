<template>
  <div class="login-history-container">
    
    <!-- 搜索过滤区域 -->
    <div class="search-container">
      <el-form :inline="true" :model="searchParams" class="search-form">
        <el-form-item label="用户ID">
          <el-input v-model="searchParams.userId" placeholder="请输入用户ID" clearable></el-input>
        </el-form-item>
        <el-form-item label="手机号">
          <el-input v-model="searchParams.mobile" placeholder="请输入手机号" clearable></el-input>
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
          <el-button type="primary" @click="searchLoginHistory">
            <i class="el-icon-search"></i> 搜索
          </el-button>
          <el-button @click="resetSearchParams">
            <i class="el-icon-refresh"></i> 重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <!-- 登录历史表格 -->
    <div class="table-container">
      <el-table
        :data="loginHistory"
        border
        stripe
        style="width: 100%"
        v-loading="loading"
        :header-cell-style="{background:'#f5f7fa', color:'#606266', fontWeight: 'bold'}"
        :cell-style="{ padding: '12px 8px' }"
      >
        <el-table-column prop="id" label="ID" min-width="80" align="center"></el-table-column>
        <el-table-column prop="user_id" label="用户ID" min-width="80" align="center"></el-table-column>
        <el-table-column prop="mobile" label="手机号" min-width="120" align="center"></el-table-column>
        <el-table-column prop="login_time" label="登录时间" min-width="160" align="center"></el-table-column>
        <el-table-column prop="status" label="登录状态" min-width="100" align="center">
          <template #default="scope">
            <div class="status-tag-container">
              <el-tag 
                :type="scope.row.status === '有效' ? 'success' : 'info'"
                class="status-tag"
                effect="dark"
                size="small"
              >
                {{ scope.row.status }}
              </el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="expire_at" label="过期时间" min-width="160" align="center"></el-table-column>
        <el-table-column prop="device_info" label="设备信息" min-width="120" align="center">
          <template #default="scope">
            {{ scope.row.device_info || '未知' }}
          </template>
        </el-table-column>
        <el-table-column prop="ip_address" label="IP地址" min-width="120" align="center">
          <template #default="scope">
            {{ scope.row.ip_address || '未知' }}
          </template>
        </el-table-column>
        <el-table-column label="操作" min-width="120" align="center">
          <template #default="scope">
            <el-button
              size="small"
              type="primary"
              @click="viewUserDetails(scope.row.user_id)"
            >
              用户详情
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

    <!-- 用户详情对话框 -->
    <el-dialog 
      title="用户详情" 
      v-model="userDialogVisible" 
      width="600px" 
      :close-on-click-modal="false"
    >
      <div v-if="selectedUser" class="user-detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="用户ID">{{ selectedUser.user_id }}</el-descriptions-item>
          <el-descriptions-item label="手机号">{{ selectedUser.mobile }}</el-descriptions-item>
          <el-descriptions-item label="注册时间">{{ selectedUser.register_time || '未记录' }}</el-descriptions-item>
          <el-descriptions-item label="最后登录">{{ selectedUser.last_login_time || '未记录' }}</el-descriptions-item>
          <el-descriptions-item label="账户状态">
            <el-tag :type="selectedUser.status === 1 ? 'success' : 'danger'">
              {{ selectedUser.status === 1 ? '正常' : '禁用' }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="主题偏好">
            {{ selectedUser.theme_preference === 'light' ? '浅色' : '深色' }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="userDialogVisible = false">关闭</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed, watch } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { API_BASE_URL } from '../../../config';

// 状态变量
const loading = ref(false);
const loginHistory = ref([]);
const totalItems = ref(0);
const currentPage = ref(1);
const pageSize = ref(10);
const userDialogVisible = ref(false);
const selectedUser = ref(null);
const dateRange = ref([]);

// API基础URL，确保没有尾斜杠
const apiBaseUrl = computed(() => {
  if (API_BASE_URL.endsWith('/')) {
    return API_BASE_URL.slice(0, -1);
  }
  return API_BASE_URL;
});

// 搜索参数
const searchParams = reactive({
  userId: '',
  mobile: '',
  dateRange: [],
});

// 监听日期范围变化，更新搜索参数
watch(dateRange, (newValue) => {
  if (newValue && newValue.length === 2) {
    searchParams.startDate = newValue[0];
    searchParams.endDate = newValue[1];
  } else {
    searchParams.startDate = null;
    searchParams.endDate = null;
  }
});

// 获取登录历史记录
const fetchLoginHistory = async () => {
  loading.value = true;
  try {
    // 构建查询参数
    const params = new URLSearchParams();
    params.append('page', currentPage.value.toString());
    params.append('page_size', pageSize.value.toString());
    
    if (searchParams.userId && searchParams.userId.trim()) {
      const userIdNum = parseInt(searchParams.userId.trim());
      if (!isNaN(userIdNum)) {
        params.append('user_id', userIdNum.toString());
      }
    }
    
    if (searchParams.mobile && searchParams.mobile.trim()) {
      params.append('mobile', searchParams.mobile.trim());
    }
    
    // 修复日期处理
    if (dateRange.value && dateRange.value.length === 2) {
      if (dateRange.value[0]) params.append('start_date', dateRange.value[0]);
      if (dateRange.value[1]) params.append('end_date', dateRange.value[1]);
    }
    
    // 使用新的API路径
    const response = await fetch(`${apiBaseUrl.value}/admin/login-history/all?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
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
      loginHistory.value = data.data.logins;
      totalItems.value = data.data.total;
    } else {
      ElMessage.error(data.message || '获取登录历史失败');
    }
  } catch (error) {
    console.error('获取登录历史失败:', error);
    ElMessage.error('网络连接异常，请稍后再试');
  } finally {
    loading.value = false;
  }
};

// 搜索登录历史
const searchLoginHistory = () => {
  currentPage.value = 1;
  fetchLoginHistory();
};

// 重置搜索参数 - 确保清空为null而非空字符串
const resetSearchParams = () => {
  Object.keys(searchParams).forEach(key => {
    searchParams[key] = null;
  });
  dateRange.value = [];
  currentPage.value = 1;
  fetchLoginHistory();
};

// 处理分页大小变化
const handleSizeChange = (size) => {
  pageSize.value = size;
  fetchLoginHistory();
};

// 处理当前页变化
const handleCurrentChange = (page) => {
  currentPage.value = page;
  fetchLoginHistory();
};

// 查看用户详情
const viewUserDetails = async (userId) => {
  loading.value = true;
  try {
    // 确保userId是整数
    const userIdNum = parseInt(userId);
    if (isNaN(userIdNum)) {
      ElMessage.error('无效的用户ID');
      loading.value = false;
      return;
    }
    
    const response = await fetch(`${apiBaseUrl.value}/admin/users/${userIdNum}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
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
      selectedUser.value = data.data;
      userDialogVisible.value = true;
    } else {
      ElMessage.error(data.message || '获取用户详情失败');
    }
  } catch (error) {
    console.error('获取用户详情失败:', error);
    ElMessage.error('网络连接异常，请稍后再试');
  } finally {
    loading.value = false;
  }
};

// 生命周期钩子
onMounted(() => {
  // 加载登录历史记录
  fetchLoginHistory();
});
</script>

<style scoped>
.login-history-container {
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

/* 添加搜索表单样式 */
.search-form {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.search-form :deep(.el-form-item) {
  margin-bottom: 10px;
  margin-right: 10px;
}

.search-form :deep(.el-date-editor.el-input__wrapper),
.search-form :deep(.el-date-editor--daterange) {
  width: 320px;
}

.search-form :deep(.el-input__wrapper) {
  width: 200px;
}

/* 登录状态标签样式 */
.status-tag-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}

:deep(.status-tag) {
  min-width: 90px;
  text-align: center;
  padding: 2px 12px;
  font-weight: 500;
  letter-spacing: 1px;
  border-radius: 12px;
  white-space: nowrap;
}

:deep(.el-tag--success.status-tag) {
  background-color: rgba(103, 194, 58, 0.9);
}

:deep(.el-tag--info.status-tag) {
  background-color: rgba(144, 147, 153, 0.9);
}

.table-container {
  background-color: #fff;
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.pagination-container {
  margin-top: 20px;
  text-align: right;
}

.user-detail {
  padding: 10px;
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