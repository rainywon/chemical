<template>
  <div class="system-feedback-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1>系统功能反馈</h1>
      <div class="status-overview">
        <el-radio-group v-model="activeStatus" size="small" @change="handleStatusChange">
          <el-radio-button label="">全部</el-radio-button>
          <el-radio-button label="pending">
            待处理 <el-badge :value="statusCounts.pending || 0" type="danger" v-if="statusCounts.pending" />
          </el-radio-button>
          <el-radio-button label="processing">
            处理中 <el-badge :value="statusCounts.processing || 0" type="warning" v-if="statusCounts.processing" />
          </el-radio-button>
          <el-radio-button label="resolved">
            已解决 <el-badge :value="statusCounts.resolved || 0" type="success" v-if="statusCounts.resolved" />
          </el-radio-button>
          <el-radio-button label="rejected">
            已拒绝 <el-badge :value="statusCounts.rejected || 0" type="info" v-if="statusCounts.rejected" />
          </el-radio-button>
        </el-radio-group>
      </div>
    </div>

    <!-- 反馈概览卡片 -->
    <div class="feedback-overview">
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :md="6" v-for="(item, index) in overviewCards" :key="index">
          <el-card shadow="hover" class="overview-card" :class="item.class">
            <div class="card-icon">
              <i :class="item.icon"></i>
            </div>
            <div class="card-info">
              <div class="card-title">{{ item.title }}</div>
              <div class="card-value">{{ item.value }}</div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 搜索和筛选区域 -->
    <div class="search-container">
      <el-form :inline="true" :model="searchForm" class="search-form">
        <el-form-item label="反馈类型">
          <el-select v-model="searchForm.feedbackType" placeholder="请选择反馈类型" clearable>
            <el-option v-for="type in feedbackTypes" :key="type.value" :label="type.label" :value="type.value">
              <span class="feedback-type-option">{{ type.icon }} {{ type.label }}</span>
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="提交日期">
          <el-date-picker
            v-model="searchForm.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            value-format="YYYY-MM-DD"
          ></el-date-picker>
        </el-form-item>
        <el-form-item>
          <el-input
            v-model="searchForm.keyword"
            placeholder="搜索反馈内容"
            prefix-icon="el-icon-search"
            clearable
          ></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="searchFeedbacks">搜索</el-button>
          <el-button @click="resetSearch">重置</el-button>
        </el-form-item>
      </el-form>
    </div>

    <!-- 反馈列表 -->
    <div class="feedback-list-container">
      <el-table
        :data="feedbackList"
        style="width: 100%"
        border
        v-loading="loading"
        :header-cell-style="{background:'#f5f7fa', color:'#606266', fontWeight: 'bold'}"
      >
        <el-table-column prop="feedback_id" label="ID" width="70" align="center"></el-table-column>
        <el-table-column prop="user_id" label="用户ID" width="90" align="center">
          <template #default="scope">
            <el-tooltip 
              content="查看用户详情" 
              placement="top" 
              v-if="scope.row.user_id"
            >
              <el-button 
                type="text" 
                @click="handleViewUser(scope.row.user_id)"
              >
                {{ scope.row.user_id }}
              </el-button>
            </el-tooltip>
            <span v-else>匿名</span>
          </template>
        </el-table-column>
        <el-table-column label="反馈类型" width="120" align="center">
          <template #default="scope">
            <el-tag :type="getFeedbackTypeTag(scope.row.feedback_type)">
              {{ getFeedbackTypeName(scope.row.feedback_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="feedback_content" label="反馈内容" min-width="250" show-overflow-tooltip>
          <template #default="scope">
            <div class="feedback-content-preview">{{ scope.row.feedback_content }}</div>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="提交时间" width="170" align="center">
          <template #default="scope">
            {{ formatDateTime(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="处理状态" width="120" align="center">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">
              {{ getStatusName(scope.row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button 
              size="small" 
              type="primary" 
              plain
              @click="handleViewDetail(scope.row)"
            >
              查看
            </el-button>
            <el-button 
              size="small" 
              type="success" 
              plain
              @click="handleProcess(scope.row)"
              :disabled="scope.row.status === 'resolved' || scope.row.status === 'rejected'"
            >
              回复处理
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-container">
        <el-pagination
          background
          layout="total, sizes, prev, pager, next, jumper"
          :total="total"
          :page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :current-page="currentPage"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        ></el-pagination>
      </div>
    </div>

    <!-- 反馈详情对话框 -->
    <el-dialog
      title="反馈详情"
      v-model="detailDialogVisible"
      width="650px"
      :close-on-click-modal="false"
    >
      <div class="feedback-detail" v-if="currentFeedback">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="反馈ID">{{ currentFeedback.feedback_id }}</el-descriptions-item>
          <el-descriptions-item label="用户ID">
            <el-button 
              type="text" 
              @click="handleViewUser(currentFeedback.user_id)"
              v-if="currentFeedback.user_id"
            >
              {{ currentFeedback.user_id }}
            </el-button>
            <span v-else>匿名</span>
          </el-descriptions-item>
          <el-descriptions-item label="反馈类型">
            <el-tag :type="getFeedbackTypeTag(currentFeedback.feedback_type)">
              {{ getFeedbackTypeName(currentFeedback.feedback_type) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="反馈内容">
            <div class="feedback-content">
              {{ currentFeedback.feedback_content }}
            </div>
          </el-descriptions-item>
          <el-descriptions-item label="提交时间">
            {{ formatDateTime(currentFeedback.created_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="处理状态">
            <el-tag :type="getStatusType(currentFeedback.status)">
              {{ getStatusName(currentFeedback.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="管理员回复" v-if="currentFeedback.admin_reply">
            <div class="admin-reply">
              {{ currentFeedback.admin_reply }}
            </div>
          </el-descriptions-item>
          <el-descriptions-item label="回复时间" v-if="currentFeedback.replied_at">
            {{ formatDateTime(currentFeedback.replied_at) }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="detailDialogVisible = false">关闭</el-button>
          <el-button 
            type="primary" 
            @click="handleProcess(currentFeedback)"
            v-if="currentFeedback && (currentFeedback.status === 'pending' || currentFeedback.status === 'processing')"
          >
            回复处理
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 处理反馈对话框 -->
    <el-dialog
      title="回复处理反馈"
      v-model="processDialogVisible"
      width="650px"
      :close-on-click-modal="false"
    >
      <div class="process-form" v-if="currentFeedback">
        <div class="feedback-info">
          <div class="feedback-header">
            <span class="id">ID: {{ currentFeedback.feedback_id }}</span>
            <el-tag :type="getFeedbackTypeTag(currentFeedback.feedback_type)">
              {{ getFeedbackTypeName(currentFeedback.feedback_type) }}
            </el-tag>
          </div>
          <div class="feedback-content-box">
            {{ currentFeedback.feedback_content }}
          </div>
          <div class="feedback-meta">
            <span>用户ID: {{ currentFeedback.user_id || '匿名' }}</span>
            <span>提交时间: {{ formatDateTime(currentFeedback.created_at) }}</span>
          </div>
        </div>

        <el-form :model="processForm" ref="processFormRef" :rules="processRules" label-width="80px">
          <el-form-item label="处理状态" prop="status">
            <el-select v-model="processForm.status" placeholder="请选择状态">
              <el-option label="处理中" value="processing"></el-option>
              <el-option label="已解决" value="resolved"></el-option>
              <el-option label="已拒绝" value="rejected"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="回复内容" prop="adminReply">
            <el-input
              v-model="processForm.adminReply"
              type="textarea"
              rows="4"
              placeholder="请输入回复内容"
            ></el-input>
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="processDialogVisible = false">取消</el-button>
          <el-button type="primary" @click="submitProcess">提交</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { API_BASE_URL } from '../../../config';

// 状态变量
const loading = ref(false);
const feedbackList = ref([]);
const total = ref(0);
const currentPage = ref(1);
const pageSize = ref(10);
const detailDialogVisible = ref(false);
const processDialogVisible = ref(false);
const currentFeedback = ref(null);
const statusCounts = ref({});
const activeStatus = ref('');

// 反馈类型列表
const feedbackTypes = [
  { label: '功能建议', value: 'suggestion', icon: '💡', description: '对系统功能的建议和意见' },
  { label: '问题反馈', value: 'bug', icon: '🐛', description: '系统问题和错误反馈' },
  { label: '内容改进', value: 'content', icon: '📝', description: '对内容的改进建议' },
  { label: '其他', value: 'other', icon: '✨', description: '其他类型的反馈' }
];

// 搜索表单
const searchForm = reactive({
  feedbackType: '',
  status: '',
  dateRange: [],
  keyword: ''
});

// 处理表单
const processForm = reactive({
  status: 'processing',
  adminReply: ''
});

// 处理表单验证规则
const processRules = {
  status: [
    { required: true, message: '请选择处理状态', trigger: 'change' }
  ],
  adminReply: [
    { required: true, message: '请输入回复内容', trigger: 'blur' },
    { min: 2, max: 500, message: '回复内容长度应在2到500个字符之间', trigger: 'blur' }
  ]
};

// 统计卡片数据
const overviewCards = computed(() => [
  {
    title: '总反馈数',
    value: total.value || 0,
    icon: 'el-icon-data-analysis',
    class: 'card-total'
  },
  {
    title: '待处理',
    value: statusCounts.value.pending || 0,
    icon: 'el-icon-warning-outline',
    class: 'card-pending'
  },
  {
    title: '处理中',
    value: statusCounts.value.processing || 0,
    icon: 'el-icon-loading',
    class: 'card-processing'
  },
  {
    title: '已解决',
    value: statusCounts.value.resolved || 0,
    icon: 'el-icon-check',
    class: 'card-resolved'
  }
]);

// 获取反馈列表
const fetchFeedbackList = async () => {
  loading.value = true;
  try {
    // 构建参数
    const params = new URLSearchParams();
    params.append('page', currentPage.value);
    params.append('page_size', pageSize.value);
    
    if (searchForm.feedbackType) {
      params.append('feedback_type', searchForm.feedbackType);
    }
    
    if (activeStatus.value) {
      params.append('status', activeStatus.value);
    }
    
    if (searchForm.dateRange && searchForm.dateRange.length === 2) {
      params.append('start_date', searchForm.dateRange[0]);
      params.append('end_date', searchForm.dateRange[1]);
    }
    
    if (searchForm.keyword) {
      params.append('keyword', searchForm.keyword);
    }
    
    // 发送请求
    const response = await fetch(`${API_BASE_URL}/admin/feedback/system?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    
    if (!response.ok) {
      throw new Error(`服务器响应错误: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.code === 200) {
      feedbackList.value = data.data.list;
      total.value = data.data.total;
    } else {
      ElMessage.error(data.message || '获取反馈列表失败');
    }
  } catch (error) {
    console.error('获取反馈列表失败:', error);
    ElMessage.error('网络错误，请稍后再试');
  } finally {
    loading.value = false;
  }
};

// 获取反馈状态统计
const fetchStatusCounts = async () => {
  try {
    // 构建参数
    const params = new URLSearchParams();
    
    if (searchForm.feedbackType) {
      params.append('feedback_type', searchForm.feedbackType);
    }
    
    if (searchForm.dateRange && searchForm.dateRange.length === 2) {
      params.append('start_date', searchForm.dateRange[0]);
      params.append('end_date', searchForm.dateRange[1]);
    }
    
    if (searchForm.keyword) {
      params.append('keyword', searchForm.keyword);
    }
    
    // 发送请求
    const response = await fetch(`${API_BASE_URL}/admin/feedback/system/stats?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    
    if (!response.ok) {
      throw new Error(`服务器响应错误: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.code === 200) {
      statusCounts.value = data.data.status_counts;
    }
  } catch (error) {
    console.error('获取状态统计失败:', error);
  }
};

// 搜索反馈
const searchFeedbacks = () => {
  currentPage.value = 1;
  fetchFeedbackList();
  fetchStatusCounts();
};

// 重置搜索条件
const resetSearch = () => {
  searchForm.feedbackType = '';
  searchForm.dateRange = [];
  searchForm.keyword = '';
  activeStatus.value = '';
  currentPage.value = 1;
  fetchFeedbackList();
  fetchStatusCounts();
};

// 处理状态变化
const handleStatusChange = () => {
  currentPage.value = 1;
  fetchFeedbackList();
};

// 分页大小改变
const handleSizeChange = (size) => {
  pageSize.value = size;
  fetchFeedbackList();
};

// 当前页码改变
const handleCurrentChange = (page) => {
  currentPage.value = page;
  fetchFeedbackList();
};

// 查看反馈详情
const handleViewDetail = (feedback) => {
  currentFeedback.value = feedback;
  detailDialogVisible.value = true;
};

// 处理反馈
const handleProcess = (feedback) => {
  currentFeedback.value = feedback;
  processForm.status = feedback.status === 'pending' ? 'processing' : feedback.status;
  processForm.adminReply = feedback.admin_reply || '';
  processDialogVisible.value = true;
};

// 查看用户详情
const handleViewUser = (userId) => {
  ElMessage.info(`查看用户 ${userId} 的详细信息 (待实现)`);
  // 可以跳转到用户管理页面或打开用户详情对话框
};

// 提交处理结果
const submitProcess = async () => {
  if (!currentFeedback.value) return;
  
  try {
    // 构建请求体
    const requestBody = {
      status: processForm.status,
      admin_reply: processForm.adminReply
    };
    
    // 发送请求
    const response = await fetch(`${API_BASE_URL}/admin/feedback/system/${currentFeedback.value.feedback_id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      throw new Error(`服务器响应错误: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.code === 200) {
      ElMessage.success('处理成功');
      processDialogVisible.value = false;
      
      // 更新本地数据
      const index = feedbackList.value.findIndex(item => item.feedback_id === currentFeedback.value.feedback_id);
      if (index !== -1) {
        feedbackList.value[index].status = processForm.status;
        feedbackList.value[index].admin_reply = processForm.adminReply;
        feedbackList.value[index].replied_at = new Date().toISOString();
      }
      
      // 重新获取状态统计
      fetchStatusCounts();
    } else {
      ElMessage.error(data.message || '处理失败');
    }
  } catch (error) {
    console.error('处理反馈失败:', error);
    ElMessage.error('网络错误，请稍后再试');
  }
};

// 获取反馈类型标签类型
const getFeedbackTypeTag = (type) => {
  const typeMap = {
    'suggestion': 'success',
    'bug': 'danger',
    'content': 'warning',
    'other': 'info'
  };
  return typeMap[type] || 'info';
};

// 获取反馈类型显示名称
const getFeedbackTypeName = (type) => {
  const findType = feedbackTypes.find(item => item.value === type);
  return findType ? findType.label : type;
};

// 获取状态标签类型
const getStatusType = (status) => {
  const statusMap = {
    'pending': 'danger',
    'processing': 'warning',
    'resolved': 'success',
    'rejected': 'info'
  };
  return statusMap[status] || 'info';
};

// 获取状态显示名称
const getStatusName = (status) => {
  const statusMap = {
    'pending': '待处理',
    'processing': '处理中',
    'resolved': '已解决',
    'rejected': '已拒绝'
  };
  return statusMap[status] || status;
};

// 格式化日期时间
const formatDateTime = (dateTime) => {
  if (!dateTime) return '';
  
  const date = new Date(dateTime);
  if (isNaN(date.getTime())) return dateTime;
  
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
};

// 组件挂载时获取数据
onMounted(() => {
  fetchFeedbackList();
  fetchStatusCounts();
});
</script>

<style scoped>
.system-feedback-container {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-header h1 {
  margin: 0;
  font-size: 24px;
  color: #303133;
}

.status-overview {
  margin-left: auto;
}

.feedback-overview {
  margin-bottom: 20px;
}

.overview-card {
  height: 100px;
  display: flex;
  align-items: center;
  padding: 20px;
  margin-bottom: 20px;
  transition: all 0.3s;
}

.overview-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
}

.card-icon {
  font-size: 32px;
  margin-right: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  width: 60px;
  height: 60px;
  border-radius: 50%;
}

.card-total .card-icon {
  background-color: #409EFF;
}

.card-pending .card-icon {
  background-color: #F56C6C;
}

.card-processing .card-icon {
  background-color: #E6A23C;
}

.card-resolved .card-icon {
  background-color: #67C23A;
}

.card-info {
  flex: 1;
}

.card-title {
  font-size: 16px;
  color: #606266;
  margin-bottom: 5px;
}

.card-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.search-container {
  background-color: #fff;
  padding: 20px;
  border-radius: 4px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.feedback-list-container {
  background-color: #fff;
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.feedback-content-preview {
  max-height: 60px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: normal;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.pagination-container {
  margin-top: 20px;
  text-align: right;
}

.feedback-content, .admin-reply {
  white-space: pre-wrap;
  background-color: #f8f9fa;
  padding: 10px;
  border-radius: 4px;
  max-height: 200px;
  overflow-y: auto;
}

.feedback-type-option {
  display: flex;
  align-items: center;
}

.feedback-info {
  margin-bottom: 20px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  overflow: hidden;
}

.feedback-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background-color: #f5f7fa;
}

.feedback-header .id {
  font-weight: bold;
}

.feedback-content-box {
  padding: 15px;
  white-space: pre-wrap;
  background-color: white;
  min-height: 80px;
}

.feedback-meta {
  display: flex;
  justify-content: space-between;
  padding: 10px 15px;
  background-color: #f5f7fa;
  color: #909399;
  font-size: 14px;
}
</style>