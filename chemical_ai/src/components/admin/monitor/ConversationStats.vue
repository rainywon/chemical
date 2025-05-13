<template>
  <div class="conversation-stats">
    
    <!-- 数据概览卡片 -->
    <div class="stats-cards">
      <div class="stat-card">
        <div class="stat-icon">
          <i class="el-icon-chat-line-round"></i>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ totalSessions }}</div>
          <div class="stat-label">总对话数</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">
          <i class="el-icon-message"></i>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ totalMessages }}</div>
          <div class="stat-label">总消息数</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">
          <i class="el-icon-s-custom"></i>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ activeUsers }}</div>
          <div class="stat-label">活跃用户数</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">
          <i class="el-icon-timer"></i>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ avgMessagesPerSession.toFixed(1) }}</div>
          <div class="stat-label">平均消息/对话</div>
        </div>
      </div>
    </div>
    
    <!-- 时间筛选 -->
    <div class="filter-container">
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        range-separator="至"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        @change="handleDateChange"
      ></el-date-picker>
    </div>
    
    <!-- 对话趋势图 -->
    <div class="chart-container">
      <div class="chart-header">
        <h2>对话趋势</h2>
        <div class="chart-actions">
          <el-radio-group v-model="timeUnit" @change="handleTimeUnitChange">
            <el-radio-button value="day">日</el-radio-button>
            <el-radio-button value="week">周</el-radio-button>
            <el-radio-button value="month">月</el-radio-button>
          </el-radio-group>
        </div>
      </div>
      <div class="chart" ref="conversationTrendChart"></div>
    </div>
    
    <!-- 最近对话列表 -->
    <div class="recent-conversations">
      <h2>最近对话</h2>
      <el-table :data="recentConversations" style="width: 100%">
        <el-table-column prop="id" label="会话ID" width="220"></el-table-column>
        <el-table-column prop="title" label="标题"></el-table-column>
        <el-table-column prop="user_id" label="用户ID" width="100"></el-table-column>
        <el-table-column prop="message_count" label="消息数" width="100"></el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="180"></el-table-column>
        <el-table-column prop="updated_at" label="更新时间" width="180"></el-table-column>
      </el-table>
      <div class="pagination">
        <el-pagination
          layout="prev, pager, next"
          :total="totalSessionsCount"
          :page-size="pageSize"
          @current-change="handlePageChange"
        ></el-pagination>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch, onBeforeUnmount } from 'vue';
import * as echarts from 'echarts';
import { API_BASE_URL } from '../../../config';
import { ElMessage } from 'element-plus';
import { useRouter } from 'vue-router';

// 添加路由器
const router = useRouter();

// 状态变量
const totalSessions = ref(0);
const totalMessages = ref(0);
const activeUsers = ref(0);
const avgMessagesPerSession = ref(0);
const dateRange = ref([]);
const timeUnit = ref('day');
const recentConversations = ref([]);
const totalSessionsCount = ref(0);
const pageSize = ref(10);
const currentPage = ref(1);
const isAuthenticated = ref(false); // 添加认证状态

// 图表引用
const conversationTrendChart = ref(null);
let trendChartInstance = null;

// 计算属性和方法
const handleDateChange = () => {
  fetchTrendData();
};

const handleTimeUnitChange = () => {
  fetchTrendData();
};

const handlePageChange = (page) => {
  currentPage.value = page;
  fetchRecentConversations();
};

// 检查用户是否是管理员
const checkAdminStatus = () => {
  return true;
};

// 获取API所需的请求头
const getAuthHeaders = () => {
  // 不检查token，只返回Content-Type
  return {
    'Content-Type': 'application/json'
  };
};

// 格式化日期
const formatDate = (date) => {
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

// 获取统计概览数据
const fetchStatisticsData = async () => {
  try {
    const headers = getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/admin/conversation/stats`, {
      method: 'GET',
      headers: headers
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
      totalSessions.value = data.data.total_sessions;
      totalMessages.value = data.data.total_messages;
      activeUsers.value = data.data.active_users;
      avgMessagesPerSession.value = data.data.avg_messages_per_session;
    } else {
      ElMessage.error(data.message || '获取统计数据失败');
    }
  } catch (error) {
    ElMessage.error('网络连接异常，请检查网络');
    console.error('获取统计数据错误:', error);
  }
};

// 获取趋势数据
const fetchTrendData = async () => {
  try {
    const headers = getAuthHeaders();
    
    let requestData = {
      time_unit: timeUnit.value
    };
    
    // 如果设置了日期范围，添加到请求中
    if (dateRange.value && dateRange.value.length === 2) {
      requestData.start_date = formatDate(dateRange.value[0]);
      requestData.end_date = formatDate(dateRange.value[1]);
    }
    
    const response = await fetch(`${API_BASE_URL}/admin/conversation/trend`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(requestData)
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
      renderConversationTrendChart(data.data);
    } else {
      ElMessage.error(data.message || '获取趋势数据失败');
    }
  } catch (error) {
    ElMessage.error('网络连接异常，请检查网络');
    console.error('获取趋势数据错误:', error);
  }
};

// 获取最近对话列表
const fetchRecentConversations = async () => {
  try {
    const headers = getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/admin/conversation/recent?page=${currentPage.value}&page_size=${pageSize.value}`, {
      method: 'GET',
      headers: headers
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
      recentConversations.value = data.data.conversations;
      totalSessionsCount.value = data.data.total;
    } else {
      ElMessage.error(data.message || '获取最近对话列表失败');
    }
  } catch (error) {
    ElMessage.error('网络连接异常，请检查网络');
    console.error('获取最近对话列表错误:', error);
  }
};

// 渲染对话趋势图
const renderConversationTrendChart = (data) => {
  // 确保DOM元素已存在
  if (!conversationTrendChart.value) {
    console.warn('图表DOM元素不存在');
    return;
  }
  
  // 销毁旧实例（如果存在）
  if (trendChartInstance) {
    trendChartInstance.dispose();
  }
  
  // 创建新实例
  try {
    trendChartInstance = echarts.init(conversationTrendChart.value);
    
    const option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      legend: {
        data: ['对话数', '消息数']
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: data.dates || []
      },
      yAxis: [
        {
          type: 'value',
          name: '对话数',
          position: 'left'
        },
        {
          type: 'value',
          name: '消息数',
          position: 'right'
        }
      ],
      series: [
        {
          name: '对话数',
          type: 'bar',
          data: data.session_data || []
        },
        {
          name: '消息数',
          type: 'line',
          yAxisIndex: 1,
          data: data.message_data || []
        }
      ]
    };
    
    trendChartInstance.setOption(option);
  } catch (e) {
    console.error('初始化图表失败:', e);
  }
};

// 监听时间单位变化
watch(timeUnit, () => {
  fetchTrendData();
});

// 生命周期钩子
onMounted(() => {
  // 检查管理员身份
  if (!checkAdminStatus()) {
    return; // 如果不是管理员，不加载数据
  }
  
  // 延迟加载以确保DOM已完全渲染
  setTimeout(() => {
    // 加载统计数据
    fetchStatisticsData();
    
    // 加载趋势数据
    fetchTrendData();
    
    // 加载最近对话列表
    fetchRecentConversations();
  }, 100);
  
  // 窗口大小变化时重新调整图表大小
  const handleResize = () => {
    if (trendChartInstance && !trendChartInstance.isDisposed()) {
      trendChartInstance.resize();
    }
  };
  
  window.addEventListener('resize', handleResize);
  
  // 在组件卸载时销毁图表实例和移除事件监听
  onBeforeUnmount(() => {
    window.removeEventListener('resize', handleResize);
    if (trendChartInstance && !trendChartInstance.isDisposed()) {
      trendChartInstance.dispose();
      trendChartInstance = null;
    }
  });
});
</script>

<style scoped>
.conversation-stats {
  padding: 20px;
}

.page-title {
  margin-bottom: 24px;
  font-size: 24px;
  font-weight: 500;
  color: #333;
}

.stats-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 30px;
}

.stat-card {
  flex: 1;
  min-width: 200px;
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
}

.stat-icon {
  font-size: 36px;
  margin-right: 16px;
  color: #409EFF;
}

.stat-content {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #909399;
}

.filter-container {
  margin-bottom: 24px;
  display: flex;
  justify-content: flex-end;
}

.chart-container {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 500;
}

.chart {
  height: 350px;
}

.charts-row {
  display: flex;
  gap: 20px;
  margin-bottom: 24px;
}

.half-width {
  flex: 1;
}

.recent-conversations {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.recent-conversations h2 {
  margin-top: 0;
  margin-bottom: 16px;
  font-size: 18px;
  font-weight: 500;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

@media (max-width: 768px) {
  .stats-cards {
    flex-direction: column;
  }
  
  .charts-row {
    flex-direction: column;
  }
  
  .chart {
    height: 300px;
  }
}
</style> 