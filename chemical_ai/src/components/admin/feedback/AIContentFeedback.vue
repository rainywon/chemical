<template>
  <div class="ai-content-feedback-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <div></div>
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
        <el-col :xs="24" :sm="12" :md="8">
          <div class="overview-card total-card">
            <div class="card-icon">
              <el-icon><ChatLineRound /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-title">总反馈数</div>
              <div class="card-value">{{ overviewData.total || 0 }}</div>
            </div>
          </div>
        </el-col>
        <el-col :xs="24" :sm="12" :md="8">
          <div class="overview-card score-card">
            <div class="card-icon">
              <el-icon><Star /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-title">平均分数</div>
              <div class="card-value">{{ (overviewData.avg_score || 0).toFixed(1) }}</div>
              <div class="card-rating">
                <el-rate :model-value="overviewData.avg_score || 0" disabled :colors="ratingColors" allow-half />
              </div>
            </div>
          </div>
        </el-col>
        <el-col :xs="24" :sm="12" :md="8">
          <div class="overview-card pending-card">
            <div class="card-icon">
              <el-icon><Clock /></el-icon>
            </div>
            <div class="card-info">
              <div class="card-title">待处理</div>
              <div class="card-value">{{ statusCounts.pending || 0 }}</div>
            </div>
          </div>
        </el-col>
      </el-row>
    </div>

    <!-- 搜索和筛选区域 -->
    <div class="search-container">
      <el-form :inline="true" :model="searchForm" class="search-form">
        <div class="search-form-left">
          <el-form-item label="评分">
            <el-select v-model="searchForm.rating" placeholder="请选择评分" clearable style="width: 120px;">
              <el-option v-for="rating in [1, 2, 3, 4, 5]" :key="rating" :label="`${rating}星`" :value="rating">
                <div class="rating-option">
                  <el-rate :model-value="rating" disabled :colors="ratingColors" />
                </div>
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="反馈选项">
            <el-select v-model="searchForm.feedbackOption" placeholder="请选择反馈选项" clearable style="width: 150px;">
              <el-option v-for="option in feedbackOptions" :key="option.value" :label="option.label" :value="option.value" />
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
              style="width: 260px;"
            ></el-date-picker>
          </el-form-item>
          <el-form-item>
            <el-input
              v-model="searchForm.keyword"
              placeholder="搜索反馈内容"
              clearable
              style="width: 180px;"
            >
              <template #prefix>
                <el-icon><Search /></el-icon>
              </template>
            </el-input>
          </el-form-item>
        </div>
        <div class="search-form-right">
          <el-form-item>
            <el-button type="primary" @click="searchFeedbacks">搜索</el-button>
            <el-button @click="resetSearch">重置</el-button>
          </el-form-item>
        </div>
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
        <el-table-column label="评分" width="160" align="center">
          <template #default="scope">
            <el-rate v-model="scope.row.rating" disabled :colors="ratingColors" />
          </template>
        </el-table-column>
        <el-table-column label="反馈选项" width="120" align="center">
          <template #default="scope">
            <el-tag :type="getFeedbackOptionType(scope.row.feedback_option)">
              {{ getFeedbackOptionLabel(scope.row.feedback_option) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="feedback" label="反馈内容" min-width="200" show-overflow-tooltip>
          <template #default="scope">
            <div class="feedback-content-preview">{{ scope.row.feedback || '无反馈内容' }}</div>
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
              处理
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
      width="700px"
      :close-on-click-modal="false"
    >
      <div class="feedback-detail" v-if="currentFeedback">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="反馈ID">{{ currentFeedback.feedback_id }}</el-descriptions-item>
          <el-descriptions-item label="评分">
            <el-rate :model-value="currentFeedback.rating" disabled :colors="ratingColors" />
          </el-descriptions-item>
          <el-descriptions-item label="反馈选项">
            <el-tag :type="getFeedbackOptionType(currentFeedback.feedback_option)">
              {{ getFeedbackOptionLabel(currentFeedback.feedback_option) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="反馈内容">
            <div class="feedback-content">
              {{ currentFeedback.feedback || '无反馈内容' }}
            </div>
          </el-descriptions-item>
          <el-descriptions-item label="用户问题">
            <div class="question-content">
              {{ currentFeedback.question }}
            </div>
          </el-descriptions-item>
          <el-descriptions-item label="AI回答">
            <div class="answer-content">
              {{ currentFeedback.message }}
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
          <el-button type="primary" @click="handleProcess(currentFeedback)" 
            v-if="currentFeedback && currentFeedback.status !== 'resolved' && currentFeedback.status !== 'rejected'">
            处理
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 处理反馈对话框 -->
    <el-dialog
      title="处理反馈"
      v-model="processDialogVisible"
      width="600px"
      :close-on-click-modal="false"
    >
      <div class="process-form" v-if="currentFeedback">
        <el-form :model="processForm" label-width="100px">
          <el-form-item label="反馈ID">
            <span>{{ currentFeedback.feedback_id }}</span>
          </el-form-item>
          <el-form-item label="评分">
            <el-rate :model-value="currentFeedback.rating" disabled :colors="ratingColors" />
          </el-form-item>
          <el-form-item label="反馈选项">
            <el-tag :type="getFeedbackOptionType(currentFeedback.feedback_option)">
              {{ getFeedbackOptionLabel(currentFeedback.feedback_option) }}
            </el-tag>
          </el-form-item>
          <el-form-item label="反馈内容">
            <div class="feedback-content">
              {{ currentFeedback.feedback || '无反馈内容' }}
            </div>
          </el-form-item>
          <el-form-item label="处理状态">
            <el-select v-model="processForm.status">
              <el-option label="处理中" value="processing"></el-option>
              <el-option label="已解决" value="resolved"></el-option>
              <el-option label="已拒绝" value="rejected"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="管理员回复">
            <el-input
              v-model="processForm.adminReply"
              type="textarea"
              :rows="4"
              placeholder="请输入回复内容"
            ></el-input>
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="processDialogVisible = false">取消</el-button>
          <el-button type="primary" @click="submitProcess" :loading="submitting">提交</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { ref, reactive, onMounted, computed } from 'vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Star, Warning, Clock, ChatLineRound } from '@element-plus/icons-vue'
import { API_BASE_URL } from '../../../config'

export default {
  name: 'AIContentFeedback',
  components: {
    Search,
    Star,
    Warning,
    Clock,
    ChatLineRound
  },
  setup() {
    // 数据状态
    const loading = ref(false)
    const submitting = ref(false)
    const activeStatus = ref('')
    const currentPage = ref(1)
    const pageSize = ref(10)
    const total = ref(0)
    const feedbackList = ref([])
    const statusCounts = ref({
      pending: 0,
      processing: 0,
      resolved: 0,
      rejected: 0
    })
    const overviewData = ref({
      total: 0,
      avg_score: 0
    })
    
    // 对话框控制
    const detailDialogVisible = ref(false)
    const processDialogVisible = ref(false)
    const currentFeedback = ref(null)
    
    // 表单数据
    const searchForm = reactive({
      rating: '',
      feedbackOption: '',
      dateRange: [],
      keyword: ''
    })
    
    const processForm = reactive({
      status: 'processing',
      adminReply: ''
    })
    
    // 常量定义
    const ratingColors = ['#F56C6C', '#F56C6C', '#E6A23C', '#E6A23C', '#67C23A']
    
    const feedbackOptions = [
      { value: "inaccurate", label: "内容不准确" },
      { value: "incomplete", label: "回答不完整" },
      { value: "irrelevant", label: "与问题不相关" },
      { value: "other", label: "其他问题" }
    ]
    
    // 方法
    const formatDateTime = (dateTime) => {
      if (!dateTime) return ''
      return new Date(dateTime).toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    }
    
    const getStatusType = (status) => {
      const types = {
        pending: 'danger',
        processing: 'warning',
        resolved: 'success',
        rejected: 'info'
      }
      return types[status] || 'info'
    }
    
    const getStatusName = (status) => {
      const names = {
        pending: '待处理',
        processing: '处理中',
        resolved: '已解决',
        rejected: '已拒绝'
      }
      return names[status] || '未知'
    }
    
    const getFeedbackOptionType = (option) => {
      const types = {
        inaccurate: 'danger',
        incomplete: 'warning',
        irrelevant: 'info',
        other: 'info'
      }
      return types[option] || 'info'
    }
    
    const getFeedbackOptionLabel = (option) => {
      const option_obj = feedbackOptions.find(item => item.value === option)
      return option_obj ? option_obj.label : option
    }
    
    // 加载反馈列表
    const loadFeedbackList = async () => {
      loading.value = true
      try {
        // 构建查询参数
        const params = {
          page: currentPage.value,
          page_size: pageSize.value
        }
        
        // 添加筛选条件
        if (activeStatus.value) {
          params.status = activeStatus.value
        }
        
        if (searchForm.rating) {
          params.rating = searchForm.rating
        }
        
        if (searchForm.feedbackOption) {
          params.feedback_option = searchForm.feedbackOption
        }
        
        if (searchForm.dateRange && searchForm.dateRange.length === 2) {
          params.start_date = searchForm.dateRange[0]
          params.end_date = searchForm.dateRange[1]
        }
        
        if (searchForm.keyword) {
          params.keyword = searchForm.keyword
        }
        
        const response = await axios.get(`${API_BASE_URL}/admin/feedback/content`, { 
          params,
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
        
        if (response.data.code === 200) {
          feedbackList.value = response.data.data.list
          total.value = response.data.data.total
        } else {
          console.error('获取反馈列表失败:', response.data.message)
          ElMessage.error(response.data.message || '获取反馈列表失败')
        }
      } catch (error) {
        console.error('获取反馈列表失败:', error)
        ElMessage.error('获取反馈列表失败: ' + error.message)
      } finally {
        loading.value = false
      }
    }
    
    // 加载反馈统计数据
    const loadFeedbackStats = async () => {
      try {
        // 构建查询参数
        const params = {}
        
        if (searchForm.rating) {
          params.rating = searchForm.rating
        }
        
        if (searchForm.feedbackOption) {
          params.feedback_option = searchForm.feedbackOption
        }
        
        if (searchForm.dateRange && searchForm.dateRange.length === 2) {
          params.start_date = searchForm.dateRange[0]
          params.end_date = searchForm.dateRange[1]
        }
        
        if (searchForm.keyword) {
          params.keyword = searchForm.keyword
        }
        
        const response = await axios.get(`${API_BASE_URL}/admin/feedback/content/stats`, { 
          params,
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
        
        if (response.data.code === 200) {
          statusCounts.value = response.data.data.status_counts
          overviewData.value = response.data.data.overview
        } else {
          console.error('获取反馈统计数据失败:', response.data.message)
        }
      } catch (error) {
        console.error('获取反馈统计数据失败:', error)
      }
    }
    
    // 搜索反馈
    const searchFeedbacks = () => {
      currentPage.value = 1
      loadFeedbackList()
      loadFeedbackStats()
    }
    
    // 重置搜索条件
    const resetSearch = () => {
      // 重置搜索表单
      searchForm.rating = ''
      searchForm.feedbackOption = ''
      searchForm.dateRange = []
      searchForm.keyword = ''
      
      // 重置状态过滤
      activeStatus.value = ''
      
      // 重新加载数据
      currentPage.value = 1
      loadFeedbackList()
      loadFeedbackStats()
    }
    
    // 状态过滤变化
    const handleStatusChange = () => {
      currentPage.value = 1
      loadFeedbackList()
    }
    
    // 分页大小改变
    const handleSizeChange = (size) => {
      pageSize.value = size
      currentPage.value = 1
      loadFeedbackList()
    }
    
    // 当前页改变
    const handleCurrentChange = (page) => {
      currentPage.value = page
      loadFeedbackList()
    }
    
    // 查看反馈详情
    const handleViewDetail = async (feedback) => {
      try {
        // 获取详细信息
        const response = await axios.get(`${API_BASE_URL}/admin/feedback/content/${feedback.feedback_id}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
        
        if (response.data.code === 200) {
          currentFeedback.value = response.data.data
          detailDialogVisible.value = true
        } else {
          ElMessage.error(response.data.message || '获取反馈详情失败')
        }
      } catch (error) {
        console.error('获取反馈详情失败:', error)
        ElMessage.error('获取反馈详情失败: ' + error.message)
      }
    }
    
    // 处理反馈
    const handleProcess = (feedback) => {
      currentFeedback.value = { ...feedback }
      processForm.status = feedback.status === 'pending' ? 'processing' : feedback.status
      processForm.adminReply = feedback.admin_reply || ''
      processDialogVisible.value = true
    }
    
    // 提交处理结果
    const submitProcess = async () => {
      if (!processForm.adminReply) {
        ElMessage.warning('请输入管理员回复内容')
        return
      }
      
      submitting.value = true
      try {
        const response = await axios.put(
          `${API_BASE_URL}/admin/feedback/content/${currentFeedback.value.feedback_id}`, 
          {
            status: processForm.status,
            admin_reply: processForm.adminReply
          },
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          }
        )
        
        if (response.data.code === 200) {
          ElMessage.success('处理反馈成功')
          processDialogVisible.value = false
          
          // 更新列表数据
          loadFeedbackList()
          loadFeedbackStats()
        } else {
          ElMessage.error(response.data.message || '处理反馈失败')
        }
      } catch (error) {
        console.error('处理反馈失败:', error)
        ElMessage.error('处理反馈失败: ' + error.message)
      } finally {
        submitting.value = false
      }
    }
    
    // 组件挂载时加载数据
    onMounted(() => {
      loadFeedbackList()
      loadFeedbackStats()
    })
    
    return {
      // 状态数据
      loading,
      submitting,
      activeStatus,
      currentPage,
      pageSize,
      total,
      feedbackList,
      statusCounts,
      overviewData,
      detailDialogVisible,
      processDialogVisible,
      currentFeedback,
      searchForm,
      processForm,
      ratingColors,
      feedbackOptions,
      
      // 方法
      formatDateTime,
      getStatusType,
      getStatusName,
      getFeedbackOptionType,
      getFeedbackOptionLabel,
      searchFeedbacks,
      resetSearch,
      handleStatusChange,
      handleSizeChange,
      handleCurrentChange,
      handleViewDetail,
      handleProcess,
      submitProcess
    }
  }
}
</script>

<style scoped>
.ai-content-feedback-container {
  padding: 24px;
  background-color: #f9fafc;
  min-height: calc(100vh - 64px);
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #ebeef5;
}

.feedback-overview {
  margin-bottom: 24px;
}

.overview-card {
  display: flex;
  align-items: center;
  padding: 20px;
  margin-bottom: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 10px 0 rgba(0, 0, 0, 0.08);
  background-color: #fff;
  transition: all 0.3s;
  height: 90px; /* 进一步减小卡片高度 */
  overflow: hidden;
  cursor: default;
  box-sizing: border-box;
}

.overview-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.12);
}

.overview-card .card-icon {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 50px;
  height: 50px;
  border-radius: 10px;
  font-size: 22px;
  margin-right: 15px;
  flex-shrink: 0; /* 避免图标被挤压 */
}

.total-card .card-icon {
  background-color: rgba(64, 158, 255, 0.15);
  color: #409EFF;
}

.score-card .card-icon {
  background-color: rgba(230, 162, 60, 0.15);
  color: #E6A23C;
}

.pending-card .card-icon {
  background-color: rgba(245, 108, 108, 0.15);
  color: #F56C6C;
}

.card-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.card-title {
  font-size: 13px;
  color: #909399;
  margin-bottom: 4px;
  font-weight: 500;
}

.card-value {
  font-size: 22px;
  font-weight: bold;
  color: #303133;
  line-height: 1.1;
}

.card-rating {
  margin-top: 4px;
}

.search-container {
  margin-bottom: 20px;
  padding: 16px 20px;
  background-color: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 10px 0 rgba(0, 0, 0, 0.08);
}

.search-form {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.search-form-left {
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  gap: 12px;
  flex: 1;
}

.search-form-right {
  margin-left: 16px;
  white-space: nowrap;
}

.search-form .el-form-item {
  margin-bottom: 0;
  margin-right: 0 !important;
}

.search-form .el-form-item__label {
  font-size: 13px; /* 小标签字体 */
  color: #606266;
  padding-right: 6px; /* 减小标签和字段之间的间距 */
}

/* 美化表单控件 */
.search-form .el-input__inner,
.search-form .el-select .el-input__inner,
.search-form .el-date-editor .el-input__inner {
  border-radius: 6px !important;
  height: 36px !important;
  line-height: 36px !important;
}

.search-form .el-button {
  height: 36px !important;
  padding: 0 15px !important;
  border-radius: 6px !important;
  font-weight: 500 !important;
}

/* 响应式调整 */
@media (max-width: 1200px) {
  .search-form-left {
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .search-form-left .el-form-item {
    margin-bottom: 6px;
  }
}

@media (max-width: 768px) {
  .ai-content-feedback-container {
    padding: 16px;
  }
  
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
  
  .search-form {
    flex-direction: column;
    align-items: stretch;
  }
  
  .search-form-left {
    flex-direction: column;
    align-items: stretch;
  }
  
  .search-form-right {
    margin-left: 0;
    margin-top: 12px;
    display: flex;
    justify-content: flex-end;
  }
  
  .search-form .el-form-item {
    width: 100%;
  }
  
  .search-form .el-select,
  .search-form .el-date-editor,
  .search-form .el-input {
    width: 100% !important;
  }
  
  .overview-card {
    padding: 16px;
    height: auto;
    min-height: 100px;
  }
  
  .feedback-list-container, 
  .search-container {
    padding: 16px;
  }
}

.feedback-list-container {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
  padding: 24px;
}

.pagination-container {
  margin-top: 24px;
  display: flex;
  justify-content: flex-end;
}

.feedback-content-preview {
  max-height: 60px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.feedback-detail {
  margin-bottom: 24px;
}

.feedback-content,
.question-content,
.answer-content,
.admin-reply {
  white-space: pre-wrap;
  word-break: break-all;
  padding: 16px;
  max-height: 200px;
  overflow-y: auto;
  background-color: #f8f9fa;
  border-radius: 4px;
  font-size: 14px;
  line-height: 1.6;
}

.rating-option {
  display: flex;
  align-items: center;
}

.question-content {
  border-left: 3px solid #409EFF;
}

.answer-content {
  border-left: 3px solid #67C23A;
}

.admin-reply {
  border-left: 3px solid #E6A23C;
}

/* 自定义表格样式 */
:deep(.el-table) {
  border-radius: 8px;
  overflow: hidden;
}

:deep(.el-table th) {
  background-color: #f5f7fa !important;
  font-weight: 600 !important;
  color: #303133 !important;
  padding: 12px 0;
}

:deep(.el-table td) {
  padding: 12px 0;
}

:deep(.el-table--border, .el-table--group) {
  border: 1px solid #ebeef5;
}

:deep(.el-table__row:hover > td) {
  background-color: #f5f7fa !important;
}

/* 自定义对话框样式 */
:deep(.el-dialog__header) {
  padding: 20px 24px;
  border-bottom: 1px solid #ebeef5;
  margin-right: 0;
}

:deep(.el-dialog__body) {
  padding: 24px;
}

:deep(.el-dialog__footer) {
  padding: 16px 24px;
  border-top: 1px solid #ebeef5;
}

:deep(.el-descriptions__label) {
  font-weight: 500;
}

/* 按钮样式优化 */
:deep(.el-button) {
  font-weight: 500;
  border-radius: 4px;
}

:deep(.el-button--primary:not(.is-plain)) {
  background-color: #409EFF;
  border-color: #409EFF;
}

:deep(.el-button--success:not(.is-plain)) {
  background-color: #67C23A;
  border-color: #67C23A;
}
</style>