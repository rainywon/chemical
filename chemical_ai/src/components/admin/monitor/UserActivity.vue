<template>
  <div class="user-activity">
    
    <!-- 数据概览卡片 -->
    <div class="stats-cards">
      <div class="stat-card">
        <div class="stat-icon">
          <i class="el-icon-user"></i>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ activeUsers }}</div>
          <div class="stat-label">活跃用户</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">
          <i class="el-icon-connection"></i>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ onlineUsers }}</div>
          <div class="stat-label">当前在线用户</div>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">
          <i class="el-icon-plus"></i>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ newUsers }}</div>
          <div class="stat-label">新增用户</div>
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
    
    <!-- 活跃趋势图 -->
    <div class="chart-container">
      <div class="chart-header">
        <h2>用户活跃趋势</h2>
        <div class="chart-actions">
          <el-radio-group v-model="timeUnit" @change="handleTimeUnitChange">
            <el-radio-button value="day">日</el-radio-button>
            <el-radio-button value="week">周</el-radio-button>
            <el-radio-button value="month">月</el-radio-button>
          </el-radio-group>
        </div>
      </div>
      <div class="chart" ref="userActivityChart"></div>
    </div>
    
    <!-- 用户登录记录表 -->
    <div class="recent-activity">
      <h2>最近登录记录</h2>
      <el-table 
        :data="recentLogins" 
        style="width: 100%" 
        border 
        stripe
        highlight-current-row
        :header-cell-style="{background:'#f5f7fa', color:'#606266', fontWeight: 'bold'}"
        :cell-style="{padding: '8px 0'}"
        table-layout="fixed">
        <el-table-column prop="user_id" label="用户ID" min-width="33%" align="center"></el-table-column>
        <el-table-column prop="mobile" label="手机号" min-width="33%" align="center"></el-table-column>
        <el-table-column prop="login_time" label="登录时间" min-width="33%" align="center"></el-table-column>
      </el-table>
      <div class="pagination">
        <el-pagination
          background
          layout="total, prev, pager, next, jumper"
          :total="totalLoginCount"
          :page-size="pageSize"
          @current-change="handlePageChange"
        ></el-pagination>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue';
import * as echarts from 'echarts';
import { API_BASE_URL } from '../../../config';
import { ElMessage } from 'element-plus';
import { useRouter } from 'vue-router';

// 添加路由器
const router = useRouter();

// 状态变量
const activeUsers = ref(0);
const onlineUsers = ref(0);
const newUsers = ref(0);
const dateRange = ref([]);
const timeUnit = ref('day');
const recentLogins = ref([]);
const totalLoginCount = ref(0);
const pageSize = ref(10);
const currentPage = ref(1);
const isAuthenticated = ref(false);

// 图表引用
const userActivityChart = ref(null);
let chartInstance = null;

// 计算属性和方法
const handleDateChange = () => {
  fetchActivityTrendData();
};

const handleTimeUnitChange = () => {
  fetchActivityTrendData();
};

const handlePageChange = (page) => {
  currentPage.value = page;
  fetchRecentLogins();
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

// 获取用户活跃度统计数据
const fetchActivityStats = async () => {
  try {
    const headers = getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/admin/user/activity/stats`, {
      method: 'GET',
      headers: headers
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
      activeUsers.value = data.data.active_users;
      onlineUsers.value = data.data.online_users;
      newUsers.value = data.data.new_users;
    } else {
      ElMessage.error(data.message || '获取用户活跃度统计失败');
    }
  } catch (error) {
    ElMessage.error('网络连接异常，请检查网络');
    console.error('获取用户活跃度统计错误:', error);
  }
};

// 获取用户活跃趋势数据
const fetchActivityTrendData = async () => {
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
    
    const response = await fetch(`${API_BASE_URL}/admin/user/activity/trend`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(requestData)
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
      renderActivityTrendChart(data.data);
    } else {
      ElMessage.error(data.message || '获取活跃趋势数据失败');
    }
  } catch (error) {
    ElMessage.error('网络连接异常，请检查网络');
    console.error('获取用户活跃趋势数据错误:', error);
  }
};

// 获取最近登录记录
const fetchRecentLogins = async () => {
  try {
    const headers = getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/admin/user/logins?page=${currentPage.value}&page_size=${pageSize.value}`, {
      method: 'GET',
      headers: headers
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
      recentLogins.value = data.data.logins;
      totalLoginCount.value = data.data.total;
    } else {
      ElMessage.error(data.message || '获取最近登录记录失败');
    }
  } catch (error) {
    ElMessage.error('网络连接异常，请检查网络');
    console.error('获取最近登录记录错误:', error);
  }
};

// 渲染用户活跃趋势图
const renderActivityTrendChart = (data) => {
  if (!chartInstance) {
    chartInstance = echarts.init(userActivityChart.value);
  }
  
  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: ['活跃用户', '新增用户']
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: data.dates
    },
    yAxis: [
      {
        type: 'value',
        name: '活跃用户数',
        position: 'left'
      },
      {
        type: 'value',
        name: '新增用户数',
        position: 'right',
        axisLine: {
          show: true,
          lineStyle: {
            color: '#91cc75'
          }
        },
        axisLabel: {
          formatter: '{value}'
        }
      }
    ],
    series: [
      {
        name: '活跃用户',
        type: 'line',
        data: data.active_user_data,
        smooth: true,
        lineStyle: {
          width: 3
        }
      },
      {
        name: '新增用户',
        type: 'bar',
        yAxisIndex: 1,
        data: data.new_user_data,
        barWidth: '40%'
      }
    ]
  };
  
  chartInstance.setOption(option);
};

// 监听时间单位变化
watch(timeUnit, () => {
  fetchActivityTrendData();
});

// 生命周期钩子
onMounted(() => {
  // 检查管理员身份
  if (!checkAdminStatus()) {
    return; // 如果不是管理员，不加载数据
  }
  
  // 加载用户活跃度统计数据
  fetchActivityStats();
  
  // 加载活跃趋势数据
  fetchActivityTrendData();
  
  // 加载最近登录记录
  fetchRecentLogins();
  
  // 窗口大小变化时重新调整图表大小
  window.addEventListener('resize', () => {
    chartInstance?.resize();
  });
});
</script>

<style scoped>
.user-activity {
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

.recent-activity {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.recent-activity h2 {
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
  
  .chart {
    height: 300px;
  }
}
</style> 