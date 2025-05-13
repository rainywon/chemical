<template>
  <div class="admin-dashboard">
    
    <!-- 1. 核心数据概览区域 -->
    <el-row :gutter="20" class="data-overview">
      <el-col :span="24" style="margin-bottom: 15px;">
        <div class="section-header">
          <h2 class="section-title">核心数据概览</h2>
          <el-button type="primary" class="refresh-btn" plain size="small" @click="refreshStats">
            <i class="el-icon-refresh"></i>
            <span>刷新数据</span>
          </el-button>
        </div>
      </el-col>
      
      <!-- 用户统计 -->
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="overview-card user-card" shadow="hover">
          <div class="card-header">
            <div class="card-icon"><i class="el-icon-user"></i></div>
            <div class="card-title">用户统计</div>
          </div>
          <div class="card-body">
            <div class="data-item">
              <span class="data-label">总用户数量</span>
              <span class="data-value">{{ stats.totalUsers }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">今日新增</span>
              <span class="data-value">{{ stats.newUsers }}</span>
              <span class="data-trend" :class="stats.newUsersTrend > 0 ? 'up' : 'down'" v-if="stats.newUsersTrend">
                {{ Math.abs(stats.newUsersTrend) }}%
              </span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 内容统计 -->
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="overview-card content-card" shadow="hover">
          <div class="card-header">
            <div class="card-icon"><i class="el-icon-document"></i></div>
            <div class="card-title">内容统计</div>
          </div>
          <div class="card-body">
            <div class="data-item">
              <span class="data-label">知识库文件</span>
              <span class="data-value">{{ stats.knowledgeCount }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">安全资料库</span>
              <span class="data-value">{{ stats.safetyCount }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">应急资料库</span>
              <span class="data-value">{{ stats.emergencyCount }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 系统活跃度 -->
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="overview-card activity-card" shadow="hover">
          <div class="card-header">
            <div class="card-icon"><i class="el-icon-chat-line-round"></i></div>
            <div class="card-title">系统活跃度</div>
          </div>
          <div class="card-body">
            <div class="data-item">
              <span class="data-label">总对话数量</span>
              <span class="data-value">{{ stats.totalSessions }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">活跃用户数</span>
              <span class="data-value">{{ stats.activeUsers }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">平均消息/对话</span>
              <span class="data-value">{{ stats.avgMessagesPerSession.toFixed(1) }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 反馈统计 -->
      <el-col :xs="24" :sm="12" :md="6">
        <el-card class="overview-card feedback-card" shadow="hover">
          <div class="card-header">
            <div class="card-icon"><i class="el-icon-star-on"></i></div>
            <div class="card-title">反馈统计</div>
          </div>
          <div class="card-body">
            <div class="data-item">
              <span class="data-label">总反馈数量</span>
              <span class="data-value">{{ stats.totalFeedbacks }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">系统功能反馈</span>
              <span class="data-value">{{ stats.systemFeedbacks }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">AI内容反馈</span>
              <span class="data-value">{{ stats.contentFeedbacks }}</span>
            </div>
            <div class="data-item">
              <span class="data-label">平均评分</span>
              <span class="data-value">{{ stats.avgRating.toFixed(1) }}</span>

            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- 2. 趋势图表区域 -->
    <el-row :gutter="20" class="chart-section">
      <!-- 用户活跃度趋势图 -->
      <el-col :xs="24" :lg="8">
        <el-card class="chart-card" shadow="hover">
          <div class="chart-header">
            <h3 class="chart-title">用户活跃度趋势</h3>
            <div class="chart-actions">
              <el-radio-group v-model="userChartTimeRange" size="small" @change="loadUserActivityChart">
                <el-radio-button :value="'7'">7天</el-radio-button>
                <el-radio-button :value="'30'">30天</el-radio-button>
              </el-radio-group>
            </div>
          </div>
          <div class="chart-container" ref="userActivityChart"></div>
        </el-card>
      </el-col>
      
      <!-- 对话数量趋势图 -->
      <el-col :xs="24" :lg="8">
        <el-card class="chart-card" shadow="hover">
          <div class="chart-header">
            <h3 class="chart-title">对话数量趋势</h3>
            <div class="chart-actions">
              <el-radio-group v-model="conversationChartTimeRange" size="small" @change="loadConversationChart">
                <el-radio-button :value="'7'">7天</el-radio-button>
                <el-radio-button :value="'30'">30天</el-radio-button>
              </el-radio-group>
            </div>
          </div>
          <div class="chart-container" ref="conversationChart"></div>
        </el-card>
      </el-col>
      
      <!-- 反馈评分趋势图 -->
      <el-col :xs="24" :lg="8">
        <el-card class="chart-card" shadow="hover">
          <div class="chart-header">
            <h3 class="chart-title">反馈评分趋势</h3>
            <div class="chart-actions">
              <el-radio-group v-model="feedbackChartTimeRange" size="small" @change="loadFeedbackChart">
                <el-radio-button :value="'7'">7天</el-radio-button>
                <el-radio-button :value="'30'">30天</el-radio-button>
              </el-radio-group>
            </div>
          </div>
          <div class="chart-container" ref="feedbackChart"></div>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- 3. 最近活动区域 -->
    <el-row :gutter="20" class="activity-section">
      <!-- 最近对话列表 -->
      <el-col :xs="24" :md="8">
        <el-card class="activity-card" shadow="hover">
          <template #header>
            <div class="card-header-with-action">
              <h3 class="section-title">最近对话</h3>
              <router-link to="/admin/monitor/conversation" class="view-more">查看更多</router-link>
            </div>
          </template>
          <div class="activity-list">
            <el-table :data="recentConversations" style="width: 100%" size="small" :show-header="false">
              <el-table-column>
                <template #default="scope">
                  <div class="activity-item">
                    <div class="activity-icon conversation-icon">
                      <i class="el-icon-chat-dot-round"></i>
                    </div>
                    <div class="activity-content">
                      <div class="activity-title">{{ scope.row.title || '无标题对话' }}</div>
                      <div class="activity-meta">
                        <span>用户: {{ scope.row.mobile }}</span>
                        <span>消息数: {{ scope.row.message_count }}</span>
                        <span class="activity-time">{{ scope.row.created_at }}</span>
                      </div>
                    </div>
                  </div>
                </template>
              </el-table-column>
            </el-table>
            <div class="empty-data" v-if="recentConversations.length === 0">
              <i class="el-icon-chat-dot-round"></i>
              <span>暂无对话数据</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 最近用户登录 -->
      <el-col :xs="24" :md="8">
        <el-card class="activity-card" shadow="hover">
          <template #header>
            <div class="card-header-with-action">
              <h3 class="section-title">最近登录</h3>
              <router-link to="/admin/users/login-history" class="view-more">查看更多</router-link>
            </div>
          </template>
          <div class="activity-list">
            <el-table :data="recentLogins" style="width: 100%" size="small" :show-header="false">
              <el-table-column>
                <template #default="scope">
                  <div class="activity-item">
                    <div class="activity-icon login-icon">
                      <i class="el-icon-user"></i>
                    </div>
                    <div class="activity-content">
                      <div class="activity-title">用户登录</div>
                      <div class="activity-meta">
                        <span>用户: {{ scope.row.mobile }}</span>
                        <span class="activity-time">{{ scope.row.login_time }}</span>
                      </div>
                    </div>
                  </div>
                </template>
              </el-table-column>
            </el-table>
            <div class="empty-data" v-if="recentLogins.length === 0">
              <i class="el-icon-user"></i>
              <span>暂无登录数据</span>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <!-- 最新反馈 -->
      <el-col :xs="24" :md="8">
        <el-card class="activity-card" shadow="hover">
          <template #header>
            <div class="card-header-with-action">
              <h3 class="section-title">最新反馈</h3>
              <router-link to="/admin/feedback/ratings" class="view-more">查看更多</router-link>
            </div>
          </template>
          <div class="activity-list">
            <el-table :data="recentFeedbacks" style="width: 100%" size="small" :show-header="false">
              <el-table-column>
                <template #default="scope">
                  <div class="activity-item">
                    <div class="activity-icon feedback-icon" :class="getFeedbackIconClass(scope.row.type)">
                      <i :class="getFeedbackIcon(scope.row.type)"></i>
                    </div>
                    <div class="activity-content">
                      <div class="activity-title">
                        <el-tag size="small" :type="getFeedbackType(scope.row.type)">{{ scope.row.typeLabel }}</el-tag>
                        <span v-if="scope.row.rating" class="feedback-rating">
                          <el-rate v-model="scope.row.rating" disabled :colors="ratingColors" :max="5"></el-rate>
                        </span>
                      </div>
                      <div class="activity-desc">{{ scope.row.content }}</div>
                      <div class="activity-meta">
                        <span class="activity-time">{{ scope.row.created_at }}</span>
                      </div>
                    </div>
                  </div>
                </template>
              </el-table-column>
            </el-table>
            <div class="empty-data" v-if="recentFeedbacks.length === 0">
              <i class="el-icon-star-on"></i>
              <span>暂无反馈数据</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- 4. 系统状态区域 & 5. 待处理事项区域 -->
    <el-row :gutter="20" class="status-section">
      <!-- 系统参数和系统版本 -->
      <el-col :xs="24" :lg="24">
        <el-card class="status-card" shadow="hover">
          <template #header>
            <div class="card-header-with-action">
              <h3 class="section-title">系统状态</h3>
              <router-link to="/admin/settings/system-params" class="view-more">系统设置</router-link>
            </div>
          </template>
          <el-row :gutter="20">
            <!-- 系统参数 -->
            <el-col :xs="24" :md="12">
              <div class="status-block">
                <h4 class="block-title">系统参数</h4>
                <ul class="param-list">
                  <li v-for="(param, index) in systemParams" :key="index" class="param-item">
                    <span class="param-label">{{ param.name }}</span>
                    <span class="param-value">{{ param.value }}</span>
                  </li>
                </ul>
              </div>
            </el-col>
            
            <!-- 系统版本 -->
            <el-col :xs="24" :md="12">
              <div class="status-block">
                <h4 class="block-title">系统版本</h4>
                <div class="version-info">
                  <div class="current-version">
                    <span class="version-label">当前版本</span>
                    <span class="version-number">{{ systemVersion.version_number }}</span>
                  </div>
                  <div class="version-date">
                    <span class="version-label">发布日期</span>
                    <span class="version-value">{{ systemVersion.release_date }}</span>
                  </div>
                  <div class="version-notes">
                    <div class="notes-title">更新内容：</div>
                    <div class="notes-content">{{ systemVersion.update_notes }}</div>
                  </div>
                </div>
              </div>
            </el-col>
          </el-row>
        </el-card>
      </el-col>
      
      <!-- 最近操作日志和待处理事项 -->
      <el-col :xs="24" :lg="24" style="margin-top: 20px;">
        <el-row :gutter="20">
          <!-- 操作日志 -->
          <el-col :xs="24" :lg="16">
            <el-card class="status-card" shadow="hover">
              <template #header>
                <div class="card-header-with-action">
                  <h3 class="section-title">最近操作日志</h3>
                  <router-link to="/admin/admins/logs" class="view-more">查看所有日志</router-link>
                </div>
              </template>
              <div class="log-container">
                <ul class="log-list">
                  <li v-for="(log, index) in operationLogs" :key="index" class="log-item">
                    <div class="log-content">
                      <span class="log-admin">{{ log.admin_name }}</span>
                      <span class="log-action">{{ log.action }}</span>
                    </div>
                    <span class="log-time">{{ log.operation_time }}</span>
                  </li>
                </ul>
                <div class="empty-data" v-if="operationLogs.length === 0">
                  <i class="el-icon-document"></i>
                  <span>暂无操作日志</span>
                </div>
              </div>
            </el-card>
          </el-col>
          
          <!-- 待处理事项区域 -->
          <el-col :xs="24" :lg="8">
            <el-card class="todo-card" shadow="hover">
              <template #header>
                <div class="card-header-with-action">
                  <h3 class="section-title">待处理事项</h3>
                  <el-button type="primary" size="small" round @click="refreshTodo">刷新</el-button>
                </div>
              </template>
              
              <div class="todo-list">
                <!-- 待处理反馈 -->
                <router-link to="/admin/feedback/list" class="todo-item" v-if="pendingItems.feedbacks > 0">
                  <div class="todo-icon feedback-icon">
                    <i class="el-icon-message"></i>
                    <el-badge :value="pendingItems.feedbacks" class="todo-badge"></el-badge>
                  </div>
                  <div class="todo-content">
                    <div class="todo-title">待处理反馈</div>
                    <div class="todo-desc">有 {{ pendingItems.feedbacks }} 条用户反馈等待处理</div>
                  </div>
                  <div class="todo-action">
                    <i class="el-icon-arrow-right"></i>
                  </div>
                </router-link>
                
                <!-- 待审核内容 -->
                <router-link to="/admin/content/document-manager" class="todo-item" v-if="pendingItems.documents > 0">
                  <div class="todo-icon document-icon">
                    <i class="el-icon-document"></i>
                    <el-badge :value="pendingItems.documents" class="todo-badge"></el-badge>
                  </div>
                  <div class="todo-content">
                    <div class="todo-title">待审核内容</div>
                    <div class="todo-desc">有 {{ pendingItems.documents }} 份文档等待审核</div>
                  </div>
                  <div class="todo-action">
                    <i class="el-icon-arrow-right"></i>
                  </div>
                </router-link>
                
                <!-- 系统告警 -->
                <div class="todo-item warning-item" v-if="pendingItems.warnings > 0">
                  <div class="todo-icon warning-icon">
                    <i class="el-icon-warning"></i>
                    <el-badge :value="pendingItems.warnings" class="todo-badge"></el-badge>
                  </div>
                  <div class="todo-content">
                    <div class="todo-title">系统告警</div>
                    <div class="todo-desc">系统检测到 {{ pendingItems.warnings }} 个潜在问题</div>
                  </div>
                  <div class="todo-action">
                    <el-button type="danger" size="small" round @click="handleWarnings">查看</el-button>
                  </div>
                </div>
                
                <!-- 无待处理事项 -->
                <div class="empty-todo" v-if="pendingItems.feedbacks === 0 && pendingItems.documents === 0 && pendingItems.warnings === 0">
                  <i class="el-icon-check"></i>
                  <span>没有待处理事项</span>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, reactive, nextTick } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import { API_BASE_URL } from '../../config';
import * as echarts from 'echarts';
import { ElMessage } from 'element-plus';

const router = useRouter();

// 统计数据
const stats = reactive({
  // 用户统计
  totalUsers: 0,
  newUsers: 0,
  newUsersTrend: 0,
  
  // 内容统计
  knowledgeCount: 0,
  safetyCount: 0,
  emergencyCount: 0,
  
  // 系统活跃度
  totalSessions: 0,
  activeUsers: 0,
  avgMessagesPerSession: 0,
  
  // 反馈统计
  totalFeedbacks: 0,
  systemFeedbacks: 0,
  contentFeedbacks: 0,
  avgRating: 0
});

// 最近活动数据
const recentConversations = ref([]);
const recentLogins = ref([]);
const recentFeedbacks = ref([]);

// 图表时间范围
const userChartTimeRange = ref('7');
const conversationChartTimeRange = ref('7');
const feedbackChartTimeRange = ref('7');

// 图表引用
const userActivityChart = ref(null);
const conversationChart = ref(null);
const feedbackChart = ref(null);

// 评分颜色
const ratingColors = reactive(['#F56C6C', '#E6A23C', '#E6A23C', '#67C23A', '#67C23A']);

// 系统参数
const systemParams = ref([]);

// 操作日志
const operationLogs = ref([]);

// 系统版本
const systemVersion = ref({
  version_number: '未知版本',
  release_date: '未知日期',
  update_notes: '暂无更新信息'
});

// 待处理事项
const pendingItems = reactive({
  feedbacks: 0,
  documents: 0,
  warnings: 0
});

// 获取反馈类型
const getFeedbackType = (type) => {
  if (!type) return 'info';
  switch (type) {
    case 'positive': return 'success';
    case 'negative': return 'danger';
    case 'suggestion': return 'warning';
    default: return 'info';
  }
};

// 获取反馈图标
const getFeedbackIcon = (type) => {
  if (!type) return 'el-icon-chat-line-round';
  switch (type) {
    case 'positive': return 'el-icon-circle-check';
    case 'negative': return 'el-icon-circle-close';
    case 'suggestion': return 'el-icon-warning-outline';
    default: return 'el-icon-chat-line-round';
  }
};

// 获取反馈图标类
const getFeedbackIconClass = (type) => {
  if (!type) return '';
  switch (type) {
    case 'positive': return 'positive-icon';
    case 'negative': return 'negative-icon';
    case 'suggestion': return 'suggestion-icon';
    default: return '';
  }
};

// 调试函数 - 在控制台输出数据加载状态
const logDebugInfo = (context, data) => {
  console.log(`[Dashboard] ${context}:`, data);
};

// 加载统计数据
const loadStats = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/stats`);
    if (response && response.data && response.data.code === 200 && response.data.data) {
      const data = response.data.data;
      
      // 用户统计
      if (data.user_stats) {
        stats.totalUsers = data.user_stats.total_users || 0;
        stats.newUsers = data.user_stats.new_users || 0;
        stats.newUsersTrend = data.user_stats.new_users_trend || 0;
      }
      
      // 内容统计 - 确保数据存在且正确映射
      if (data.content_stats) {
        stats.knowledgeCount = data.content_stats.knowledge_count || 0;
        stats.safetyCount = data.content_stats.safety_count || 0;
        stats.emergencyCount = data.content_stats.emergency_count || 0;
      }
      
      // 系统活跃度
      if (data.system_activity) {
        stats.totalSessions = data.system_activity.total_sessions || 0;
        stats.activeUsers = data.system_activity.active_users || 0;
        // 计算每个会话的平均消息数
        if (data.system_activity.avg_messages_per_session !== undefined) {
          stats.avgMessagesPerSession = data.system_activity.avg_messages_per_session;
        } else if (data.system_activity.total_sessions && data.system_activity.total_questions) {
          // 如果后端没有直接提供，尝试自己计算
          stats.avgMessagesPerSession = data.system_activity.total_questions / data.system_activity.total_sessions;
        } else {
          stats.avgMessagesPerSession = 0;
        }
      }
      
      // 反馈统计
      if (data.feedback_stats) {
        stats.totalFeedbacks = data.feedback_stats.total_feedbacks || 0;
        stats.systemFeedbacks = data.feedback_stats.system_feedbacks || 0;
        stats.contentFeedbacks = data.feedback_stats.content_feedbacks || 0;
        stats.avgRating = data.feedback_stats.avg_rating || 0;
      }
      
      console.log("Dashboard stats loaded successfully:", stats);
    }
  } catch (error) {
    console.error('加载统计数据失败:', error);
  }
};

// 加载最近对话
const loadRecentConversations = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/recent-data`);
    if (response && response.data && response.data.code === 200 && response.data.data) {
      recentConversations.value = response.data.data.recent_conversations || [];
    }
  } catch (error) {
    console.error('加载最近对话失败:', error);
    recentConversations.value = []; // 确保在错误情况下使用空数组
  }
};

// 加载最近登录
const loadRecentLogins = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/recent-data`);
    if (response && response.data && response.data.code === 200 && response.data.data) {
      logDebugInfo('Recent logins response', response.data.data);
      recentLogins.value = response.data.data.recent_logins || [];
      
      // 数据格式化和错误处理
      recentLogins.value = recentLogins.value.map(login => ({
        ...login,
        mobile: login.mobile || `用户${login.user_id}`,
        login_time: login.login_time || new Date().toISOString()
      }));
      
      logDebugInfo('Processed recent logins', recentLogins.value);
    }
  } catch (error) {
    console.error('加载最近登录失败:', error);
    recentLogins.value = []; // 确保在错误情况下使用空数组
  }
};

// 加载最新反馈
const loadRecentFeedbacks = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/recent-data`);
    if (response && response.data && response.data.code === 200 && response.data.data) {
      logDebugInfo('Recent feedbacks response', response.data.data);
      recentFeedbacks.value = response.data.data.recent_feedbacks || [];
      
      // 确保每个反馈项都有正确的属性
      recentFeedbacks.value = recentFeedbacks.value.map(feedback => ({
        ...feedback,
        // 确保rating是数值
        rating: typeof feedback.rating === 'number' ? feedback.rating : 0,
        // 确保type存在
        type: feedback.type || 'info',
        // 确保typeLabel存在
        typeLabel: feedback.typeLabel || '反馈',
        // 确保content存在
        content: feedback.content || '无内容'
      }));
      
      logDebugInfo('Processed recent feedbacks', recentFeedbacks.value);
    }
  } catch (error) {
    console.error('加载最新反馈失败:', error);
    recentFeedbacks.value = []; // 确保在错误情况下使用空数组
  }
};

// 加载系统参数和操作日志和系统版本
const loadSystemInfo = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/system-info`);
    if (response && response.data && response.data.code === 200 && response.data.data) {
      systemParams.value = response.data.data.system_params || [];
      operationLogs.value = response.data.data.operation_logs || [];
      
      if (response.data.data.system_version) {
        systemVersion.value = {
          version_number: response.data.data.system_version.version_number || '未知版本',
          release_date: response.data.data.system_version.release_date || '未知日期',
          update_notes: response.data.data.system_version.update_notes || '暂无更新信息'
        };
      }
      
      if (response.data.data.pending_items) {
        pendingItems.feedbacks = response.data.data.pending_items.feedbacks || 0;
        pendingItems.documents = response.data.data.pending_items.documents || 0;
        pendingItems.warnings = response.data.data.pending_items.warnings || 0;
      }
    }
  } catch (error) {
    console.error('加载系统信息失败:', error);
    // 设置默认值
    systemParams.value = [];
    operationLogs.value = [];
    systemVersion.value = {
      version_number: '未知版本',
      release_date: '未知日期',
      update_notes: '暂无更新信息'
    };
    pendingItems.feedbacks = 0;
    pendingItems.documents = 0;
    pendingItems.warnings = 0;
  }
};

// 初始化图表的公共函数
const initChart = (chartRef, option) => {
  if (!chartRef.value) return null;
  
  try {
    // 检查元素是否存在于DOM中
    if (!document.body.contains(chartRef.value)) {
      console.warn('图表容器不在DOM中');
      return null;
    }
    
    // 初始化图表
    const chart = echarts.init(chartRef.value);
    
    // 设置选项
    chart.setOption(option);
    
    // 窗口大小变化时重置图表大小
    const resizeHandler = () => {
      if (chart && !chart.isDisposed()) {
        chart.resize();
      }
    };
    
    window.addEventListener('resize', resizeHandler);
    
    // 返回图表实例便于后续清理
    return chart;
  } catch (error) {
    console.error('初始化图表失败:', error);
    return null;
  }
};

// 清理图表实例
const disposeChart = (chart) => {
  if (chart && !chart.isDisposed()) {
    chart.dispose();
  }
};

// 加载用户活跃度图表
const loadUserActivityChart = async () => {
  if (!userActivityChart.value) return;
  
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/user-activity`, {
      params: { days: parseInt(userChartTimeRange.value) }
    });
    
    if (response && response.data && response.data.code === 200 && response.data.data) {
      const { dates = [], counts = [] } = response.data.data;
      
      if (dates.length === 0 || counts.length === 0) {
        console.warn('返回的图表数据为空');
        return;
      }
      
      const option = {
        tooltip: {
          trigger: 'axis'
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: dates
        },
        yAxis: {
          type: 'value'
        },
        series: [{
          name: '活跃用户',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: {
            width: 3,
            color: '#409EFF'
          },
          itemStyle: {
            color: '#409EFF'
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
              offset: 0,
              color: 'rgba(64, 158, 255, 0.3)'
            }, {
              offset: 1,
              color: 'rgba(64, 158, 255, 0.1)'
            }])
          },
          data: counts
        }]
      };
      
      initChart(userActivityChart, option);
    }
  } catch (error) {
    console.error('加载用户活跃度图表数据失败:', error);
  }
};

// 加载对话数量图表
const loadConversationChart = async () => {
  if (!conversationChart.value) return;
  
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/conversation-trend`, {
      params: { days: parseInt(conversationChartTimeRange.value) }
    });
    
    if (response && response.data && response.data.code === 200 && response.data.data) {
      const { dates = [], counts = [] } = response.data.data;
      
      if (dates.length === 0 || counts.length === 0) {
        console.warn('返回的图表数据为空');
        return;
      }
      
      const option = {
        tooltip: {
          trigger: 'axis'
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: dates
        },
        yAxis: {
          type: 'value'
        },
        series: [{
          name: '对话数量',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: {
            width: 3,
            color: '#67C23A'
          },
          itemStyle: {
            color: '#67C23A'
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
              offset: 0,
              color: 'rgba(103, 194, 58, 0.3)'
            }, {
              offset: 1,
              color: 'rgba(103, 194, 58, 0.1)'
            }])
          },
          data: counts
        }]
      };
      
      initChart(conversationChart, option);
    }
  } catch (error) {
    console.error('加载对话数量图表数据失败:', error);
  }
};

// 加载反馈评分图表
const loadFeedbackChart = async () => {
  if (!feedbackChart.value) return;
  
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/feedback-rating`, {
      params: { days: parseInt(feedbackChartTimeRange.value) }
    });
    
    if (response && response.data && response.data.code === 200 && response.data.data) {
      const { dates = [], ratings = [] } = response.data.data;
      
      if (dates.length === 0 || ratings.length === 0) {
        console.warn('返回的图表数据为空');
        return;
      }
      
      const option = {
        tooltip: {
          trigger: 'axis'
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: dates
        },
        yAxis: {
          type: 'value',
          min: 1,
          max: 5
        },
        series: [{
          name: '平均评分',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: {
            width: 3,
            color: '#E6A23C'
          },
          itemStyle: {
            color: '#E6A23C'
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
              offset: 0,
              color: 'rgba(230, 162, 60, 0.3)'
            }, {
              offset: 1,
              color: 'rgba(230, 162, 60, 0.1)'
            }])
          },
          data: ratings
        }]
      };
      
      initChart(feedbackChart, option);
    }
  } catch (error) {
    console.error('加载反馈评分图表数据失败:', error);
  }
};

// 刷新待处理事项
const refreshTodo = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/dashboard/system-info`);
    if (response && response.data && response.data.code === 200 && response.data.data && response.data.data.pending_items) {
      pendingItems.feedbacks = response.data.data.pending_items.feedbacks || 0;
      pendingItems.documents = response.data.data.pending_items.documents || 0;
      pendingItems.warnings = response.data.data.pending_items.warnings || 0;
    }
    ElMessage.success('刷新成功');
  } catch (error) {
    console.error('刷新待处理事项失败:', error);
    ElMessage.error('刷新失败');
  }
};

// 初始化图表
const initCharts = () => {
  loadUserActivityChart();
  loadConversationChart();
  loadFeedbackChart();
};

// 处理系统告警
const handleWarnings = () => {
  // 实际项目中应该打开告警详情或跳转到相应页面
  ElMessage.info('查看系统告警详情');
};

// 直接获取内容统计数据
const loadContentStats = async () => {
  try {
    // 这里可以调用三个不同的API来获取各类内容数量
    // 知识库文件
    const knowledgeResponse = await axios.get(`${API_BASE_URL}/admin/content/knowledge-files`, {
      params: { page: 1, page_size: 1 }
    });
    if (knowledgeResponse && knowledgeResponse.data && knowledgeResponse.data.success && knowledgeResponse.data.data) {
      stats.knowledgeCount = knowledgeResponse.data.data.total || 0;
    }
    
    // 安全资料库文件
    const safetyResponse = await axios.get(`${API_BASE_URL}/admin/content/safety-documents`, {
      params: { page: 1, page_size: 1 }
    });
    if (safetyResponse && safetyResponse.data && safetyResponse.data.success && safetyResponse.data.data) {
      stats.safetyCount = safetyResponse.data.data.total || 0;
    }
    
    // 应急预案文件
    const emergencyResponse = await axios.get(`${API_BASE_URL}/admin/content/emergency-plans`, {
      params: { page: 1, page_size: 1 }
    });
    if (emergencyResponse && emergencyResponse.data && emergencyResponse.data.success && emergencyResponse.data.data) {
      stats.emergencyCount = emergencyResponse.data.data.total || 0;
    }
  } catch (error) {
    console.error('加载内容统计数据失败:', error);
  }
};

// 定义刷新数据的函数
const refreshStats = async () => {
  try {
    // 显示加载中提示
    ElMessage({
      message: '正在刷新数据...',
      type: 'info',
      duration: 1000
    });
    
    // 并行加载所有数据
    await Promise.all([
      loadStats(),
      loadContentStats(),
      loadRecentConversations(),
      loadRecentLogins(),
      loadRecentFeedbacks(),
      loadSystemInfo()
    ]);
    
    // 刷新成功提示
    ElMessage({
      message: '数据刷新成功',
      type: 'success',
      duration: 1500
    });
  } catch (error) {
    console.error('刷新数据失败:', error);
    ElMessage.error('刷新数据失败，请检查网络连接');
  }
};

// 确保安全地访问属性
const getSafeValue = (obj, path, defaultValue = '') => {
  try {
    return path.split('.').reduce((o, p) => (o && o[p] !== undefined) ? o[p] : defaultValue, obj);
  } catch (e) {
    return defaultValue;
  }
};

// 生命周期钩子
onMounted(async () => {
  // 使用try-catch包裹，防止任何初始化错误影响整个页面
  try {
    logDebugInfo('Dashboard initializing', 'Started loading data');
    
    // 首先加载主统计数据
    await loadStats();
    
    // 加载内容统计数据
    await loadContentStats();
    
    // 延迟加载其他数据，避免并发请求过多
    setTimeout(async () => {
      try {
        // 并行加载最近活动数据
        await Promise.all([
          loadRecentConversations(),
          loadRecentLogins(),
          loadRecentFeedbacks()
        ]);
        
        // 加载系统信息
        await loadSystemInfo();
        
        logDebugInfo('Dashboard initialization', 'All data loaded successfully');
      } catch (error) {
        console.error('加载最近活动数据和系统信息失败:', error);
      }
    }, 300);
    
    // 图表初始化需要等DOM渲染完成后执行
    nextTick(() => {
      // 确保DOM已准备好
      setTimeout(() => {
        initCharts();
        logDebugInfo('Dashboard charts', 'Initialized');
      }, 400);
    });
  } catch (error) {
    console.error('Dashboard 初始化失败:', error);
  }
});
</script>

<style scoped>
.admin-dashboard {
  padding: 20px;
  background-color: #f0f2f5;
  min-height: calc(100vh - 60px);
}

.dashboard-title {
  margin-bottom: 24px;
  font-size: 24px;
  font-weight: 500;
  color: #303133;
}

/* 核心数据概览区域样式 */
.data-overview {
  margin-bottom: 30px;
  position: relative;
  background: linear-gradient(120deg, #ffffff 0%, #f0f7ff 50%, #eef7fe 100%);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.06);
  border: 1px solid rgba(100, 181, 246, 0.1);
  position: relative;
  overflow: hidden;
}

.data-overview::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle at top right, rgba(64, 158, 255, 0.08) 0%, rgba(100, 181, 246, 0) 70%);
  z-index: 0;
}

.data-overview::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle at bottom left, rgba(103, 194, 58, 0.08) 0%, rgba(149, 212, 117, 0) 70%);
  z-index: 0;
}

.data-overview > * {
  position: relative;
  z-index: 1;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 16px;
  margin-bottom: 20px;
  border-bottom: 1px solid rgba(100, 181, 246, 0.1);
  position: relative;
}

.section-header::after {
  content: '';
  position: absolute;
  left: 0;
  bottom: -1px;
  width: 80px;
  height: 3px;
  background: linear-gradient(90deg, #409eff, #67c23a);
  border-radius: 3px;
}

.section-title {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #303133;
  position: relative;
  padding-left: 12px;
  letter-spacing: 0.5px;
}

.section-title::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 18px;
  background: linear-gradient(to bottom, #409eff, #67c23a);
  border-radius: 2px;
}

.overview-card {
  height: 100%;
  border-radius: 12px;
  overflow: hidden;
  transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
  margin-bottom: 20px;
  border: none;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  position: relative;
  z-index: 1;
}

.overview-card::after {
  content: '';
  position: absolute;
  z-index: -1;
  width: 100%;
  height: 100%;
  opacity: 0;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
  top: 0;
  left: 0;
}

.overview-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
  background: rgba(255, 255, 255, 0.95);
}

.overview-card:hover::after {
  opacity: 1;
}

.card-header {
  display: flex;
  align-items: center;
  padding-bottom: 14px;
  margin-bottom: 14px;
  border-bottom: 1px dashed rgba(220, 223, 230, 0.7);
  position: relative;
}

.card-header::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 30%;
  height: 2px;
  transition: all 0.3s ease;
}

.user-card .card-header::after {
  background: linear-gradient(90deg, #409eff, transparent);
}

.content-card .card-header::after {
  background: linear-gradient(90deg, #67c23a, transparent);
}

.activity-card .card-header::after {
  background: linear-gradient(90deg, #e6a23c, transparent);
}

.feedback-card .card-header::after {
  background: linear-gradient(90deg, #f56c6c, transparent);
}

.overview-card:hover .card-header::after {
  width: 100%;
}

.card-icon {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 16px;
  font-size: 24px;
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.card-icon::before {
  content: '';
  position: absolute;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.15);
  top: -50%;
  left: -50%;
  transform: scale(0);
  transition: transform 0.5s ease;
  border-radius: 50%;
}

.overview-card:hover .card-icon::before {
  transform: scale(3);
}

.overview-card:hover .card-icon {
  transform: scale(1.15) rotate(8deg);
}

.user-card .card-icon {
  background: linear-gradient(135deg, #409eff 0%, #64b5f6 100%);
  color: white;
}

.content-card .card-icon {
  background: linear-gradient(135deg, #67c23a 0%, #95d475 100%);
  color: white;
}

.activity-card .card-icon {
  background: linear-gradient(135deg, #e6a23c 0%, #f5cd79 100%);
  color: white;
}

.feedback-card .card-icon {
  background: linear-gradient(135deg, #f56c6c 0%, #f78989 100%);
  color: white;
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  letter-spacing: 0.5px;
  transition: all 0.3s ease;
}

.overview-card:hover .card-title {
  transform: translateX(5px);
}

.card-body {
  padding: 0 4px;
}

.data-item {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 16px;
  position: relative;
  padding: 8px 10px;
  border-radius: 8px;
  transition: all 0.3s ease;
  justify-content: space-between;
  background-color: rgba(255, 255, 255, 0.6);
  border: 1px solid rgba(230, 236, 241, 0.8);
}

.data-item:hover {
  background-color: rgba(240, 247, 255, 0.9);
  transform: translateX(6px);
  border-color: rgba(64, 158, 255, 0.15);
  box-shadow: 0 3px 8px rgba(0, 0, 0, 0.03);
}

.data-item:last-child {
  margin-bottom: 0;
}

.data-item::after {
  content: '';
  position: absolute;
  bottom: -7px;
  left: 0;
  width: 100%;
  height: 1px;
  background: linear-gradient(to right, transparent, rgba(64, 158, 255, 0.1), transparent);
}

.data-item:last-child::after {
  display: none;
}

.data-label {
  color: #606266;
  font-size: 14px;
  margin-right: 10px;
  flex: 1 1 auto;
  font-weight: 500;
  transition: all 0.2s ease;
  letter-spacing: 0.3px;
}

.data-item:hover .data-label {
  color: #303133;
  transform: translateX(2px);
}

.data-value {
  font-size: 18px;
  font-weight: 700;
  color: #303133;
  margin-right: 10px;
  transition: all 0.3s ease;
  position: relative;
  text-align: right;
  margin-left: auto;
  min-width: 40px;
  padding-left: 10px;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5));
  padding: 2px 8px;
  border-radius: 20px;
}

.data-value::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  width: 0;
  height: 2px;
  transition: all 0.3s ease;
}

.user-card .data-value::after {
  background-color: #409eff;
}

.content-card .data-value::after {
  background-color: #67c23a;
}

.activity-card .data-value::after {
  background-color: #e6a23c;
}

.feedback-card .data-value::after {
  background-color: #f56c6c;
}

.data-trend {
  font-size: 12px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  margin-right: 0;
  min-width: 40px;
}

.data-trend.up {
  color: #67c23a;
  background-color: rgba(103, 194, 58, 0.1);
}

.data-trend.down {
  color: #f56c6c;
  background-color: rgba(245, 108, 108, 0.1);
}

.rating-stars {
  margin-top: 6px;
}

/* 趋势图表区域样式 */
.chart-section {
  margin-bottom: 24px;
}

.chart-card {
  height: 100%;
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.3s;
  margin-bottom: 20px;
}

.chart-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-title {
  font-size: 16px;
  font-weight: 500;
  color: #303133;
  margin: 0;
}

.chart-actions {
  margin-left: auto;
}

.chart-container {
  height: 300px;
}

/* 最近活动区域样式 */
.activity-section {
  margin-bottom: 24px;
}

.activity-card {
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.3s;
  margin-bottom: 20px;
  height: 100%;
}

.card-header-with-action {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.view-more {
  font-size: 13px;
  color: #409eff;
  text-decoration: none;
}

.view-more:hover {
  color: #66b1ff;
}

.activity-list {
  max-height: 400px;
  overflow-y: auto;
}

.activity-item {
  display: flex;
  align-items: flex-start;
  padding: 12px 0;
}

.activity-icon {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;
  flex-shrink: 0;
}

.conversation-icon {
  background-color: #ecf5ff;
  color: #409eff;
}

.login-icon {
  background-color: #f0f9eb;
  color: #67c23a;
}

.feedback-icon {
  background-color: #fdf6ec;
  color: #e6a23c;
}

.positive-icon {
  background-color: #f0f9eb;
  color: #67c23a;
}

.negative-icon {
  background-color: #fef0f0;
  color: #f56c6c;
}

.suggestion-icon {
  background-color: #fdf6ec;
  color: #e6a23c;
}

.activity-content {
  flex: 1;
}

.activity-title {
  font-weight: 500;
  color: #303133;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
}

.feedback-rating {
  margin-left: 8px;
}

.activity-desc {
  font-size: 13px;
  color: #606266;
  margin-bottom: 4px;
  word-break: break-all;
}

.activity-meta {
  font-size: 12px;
  color: #909399;
}

.activity-time {
  margin-left: 8px;
}

.empty-data {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 0;
  color: #909399;
}

.empty-data i {
  font-size: 48px;
  margin-bottom: 16px;
  color: #dcdfe6;
}

/* 系统状态区域样式 */
.status-section {
  margin-bottom: 24px;
}

.status-card, .todo-card {
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.3s;
  margin-bottom: 20px;
  height: 100%;
  border: none;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  position: relative;
}

.status-card:hover, .todo-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
}

.status-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, #409eff, #67c23a);
  opacity: 0;
  transition: all 0.3s;
}

.status-card:hover::before {
  opacity: 1;
}

.status-block {
  margin-bottom: 20px;
  position: relative;
}

.block-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #303133;
  padding-bottom: 10px;
  border-bottom: 1px solid #ebeef5;
  position: relative;
}

.block-title::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 40px;
  height: 3px;
  background: linear-gradient(90deg, #409eff, #67c23a);
  border-radius: 3px;
  transition: all 0.3s;
}

.status-block:hover .block-title::after {
  width: 80px;
}

/* 系统参数样式 */
.param-list {
  list-style: none;
  padding: 0;
  margin: 0;
  background: linear-gradient(135deg, #ffffff 0%, #f1f5ff 100%);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(64, 158, 255, 0.1);
  padding: 8px;
  transition: all 0.3s;
}

.param-list:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.param-item {
  display: flex;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px dashed rgba(64, 158, 255, 0.15);
  transition: all 0.3s;
  border-radius: 6px;
  margin-bottom: 4px;
  position: relative;
  overflow: hidden;
  background-color: rgba(255, 255, 255, 0.8);
}

.param-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba(64, 158, 255, 0.05) 0%, rgba(103, 194, 58, 0.05) 100%);
  opacity: 0;
  transition: all 0.3s;
}

.param-item:hover {
  background-color: rgba(255, 255, 255, 1);
  transform: translateX(4px);
}

.param-item:hover::before {
  opacity: 1;
}

.param-item:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.param-label {
  color: #606266;
  flex: 0 0 60%;
  font-weight: 500;
  position: relative;
}

.param-value {
  color: #303133;
  font-weight: 600;
  background: linear-gradient(135deg, rgba(103, 194, 58, 0.1) 0%, rgba(64, 158, 255, 0.1) 100%);
  padding: 4px 12px;
  border-radius: 20px;
  transition: all 0.3s;
  border: 1px solid rgba(103, 194, 58, 0.1);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  min-width: 80px;
  text-align: center;
  position: relative;
}

.param-item:hover .param-value {
  background: linear-gradient(135deg, rgba(103, 194, 58, 0.2) 0%, rgba(64, 158, 255, 0.2) 100%);
  transform: scale(1.05);
  box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
}

/* 系统版本样式 */
.version-info {
  padding: 16px;
  background: linear-gradient(135deg, #ffffff 0%, #f1f5ff 100%);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(64, 158, 255, 0.1);
  transition: all 0.3s;
}

.version-info:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.current-version {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}

.version-label {
  color: #606266;
  margin-right: 10px;
  font-weight: 500;
}

.version-number {
  font-size: 20px;
  font-weight: 700;
  color: #409eff;
  background: linear-gradient(90deg, #409eff, #67c23a);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  padding: 2px 0;
}

.version-date {
  margin-bottom: 16px;
  display: flex;
  align-items: center;
}

.version-value {
  color: #303133;
  font-weight: 500;
}

.version-notes {
  background-color: white;
  padding: 12px;
  border-radius: 6px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
  border-left: 4px solid #67c23a;
  transition: all 0.3s;
  position: relative;
  overflow: hidden;
}

.version-notes::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba(103, 194, 58, 0.05) 0%, rgba(64, 158, 255, 0.05) 100%);
  opacity: 0;
  transition: all 0.3s;
}

.version-notes:hover {
  box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.version-notes:hover::before {
  opacity: 1;
}

.notes-title {
  font-weight: 600;
  margin-bottom: 8px;
  color: #303133;
  position: relative;
}

.notes-content {
  font-size: 14px;
  color: #606266;
  white-space: pre-line;
  line-height: 1.6;
  position: relative;
}

/* 待处理事项样式优化 */
.todo-list {
  display: flex;
  flex-direction: column;
}

.todo-item {
  display: flex;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #ebeef5;
  text-decoration: none;
  color: inherit;
  transition: all 0.3s;
  border-radius: 6px;
}

.todo-item:hover {
  background-color: #f5f7fa;
  transform: translateX(4px);
}

.todo-item:last-child {
  border-bottom: none;
}

.todo-icon {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 16px;
  position: relative;
  font-size: 20px;
  transition: all 0.3s;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
}

.todo-item:hover .todo-icon {
  transform: scale(1.1) rotate(5deg);
}

.feedback-icon {
  background: linear-gradient(135deg, #409eff 0%, #64b5f6 100%);
  color: white;
}

.document-icon {
  background: linear-gradient(135deg, #67c23a 0%, #95d475 100%);
  color: white;
}

.warning-icon {
  background: linear-gradient(135deg, #f56c6c 0%, #f78989 100%);
  color: white;
}

.todo-badge {
  position: absolute;
  top: -6px;
  right: -6px;
}

.todo-content {
  flex: 1;
}

.todo-title {
  font-weight: 600;
  color: #303133;
  margin-bottom: 6px;
  font-size: 15px;
}

.todo-desc {
  font-size: 13px;
  color: #606266;
}

.todo-action {
  margin-left: 12px;
  color: #909399;
  font-size: 18px;
  transition: all 0.3s;
}

.todo-item:hover .todo-action {
  color: #409eff;
  transform: translateX(4px);
}

.warning-item {
  background-color: #fff8f8;
}

.warning-item:hover {
  background-color: #fef0f0;
}

.empty-todo {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 0;
  color: #67c23a;
}

.empty-todo i {
  font-size: 48px;
  margin-bottom: 16px;
}

/* 刷新按钮样式 */
.refresh-btn {
  border-radius: 20px;
  transition: all 0.3s ease;
  padding: 8px 16px;
  border: 1px solid rgba(64, 158, 255, 0.2);
  background: rgba(64, 158, 255, 0.1);
  color: #409eff;
  font-weight: 500;
}

.refresh-btn:hover {
  background: rgba(64, 158, 255, 0.2);
  transform: translateY(-2px);
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.3);
}

.refresh-btn i {
  margin-right: 4px;
  transition: all 0.3s ease;
}

.refresh-btn:hover i {
  transform: rotate(180deg);
}

/* 操作日志样式 */
.log-container {
  height: 100%;
  max-height: 300px;
  overflow-y: auto;
}

.log-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.log-item {
  padding: 10px;
  border-bottom: 1px dashed #ebeef5;
  transition: all 0.3s;
  border-radius: 4px;
}

.log-item:hover {
  background-color: #f5f7fa;
  transform: translateX(4px);
}

.log-item:last-child {
  border-bottom: none;
}

.log-content {
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.log-admin {
  font-weight: 600;
  color: #409eff;
  margin-right: 6px;
  position: relative;
  padding-right: 12px;
}

.log-admin::after {
  content: '';
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  height: 12px;
  width: 1px;
  background-color: #dcdfe6;
}

.log-action {
  color: #606266;
  flex: 1;
}

.log-time {
  font-size: 12px;
  color: #909399;
  display: block;
  width: 100%;
  text-align: right;
}
</style>