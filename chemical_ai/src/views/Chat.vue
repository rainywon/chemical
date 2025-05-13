<template>
  <div id="common">
    <el-container class="container">
      <!-- 侧边栏区域 -->
      <el-aside class="app-sidebar" :width="sidebarWidth">
        <!-- 侧边栏头部 -->
        <header class="sidebar-header">
          <template v-if="!isCollapsed">
            <img class="brand-icon" src="../assets/product.png" alt="Logo" />
            <span class="brand-text">天工AI智能助手</span>
          </template>
          <el-tooltip :content="isCollapsed ? '展开侧栏' : '收起侧栏'" placement="bottom" :enterable="false" :hide-after="30" effect="dark" popper-class="action-tooltip" :manual="false">
            <div class="action-icon-button collapse-icon" @click="toggleSidebar">
              <el-icon v-if="!isCollapsed"><Fold /></el-icon>
              <el-icon v-else><Expand /></el-icon>
            </div>
          </el-tooltip>
        </header>

        <!-- 新对话 -->
        <div class="new-chat" v-if="!isCollapsed">
          <el-button type="primary" class="new-chat-btn" @click="createNewChatSession">+ 新对话</el-button>
        </div>
        <div class="new-chat collapsed-new-chat" v-if="isCollapsed">
          <el-tooltip content="新建对话" placement="bottom" :enterable="false" :hide-after="30" effect="dark" popper-class="action-tooltip" :manual="false">
            <div class="action-icon-button new-chat-icon" @click="createNewChatSession">
              <el-icon><Plus /></el-icon>
            </div>
          </el-tooltip>
        </div>
        
        <!-- 生成模式选择 -->
        <div class="model-select-container" v-if="!isCollapsed">
          <el-select v-model="selectedOption" placeholder="选择生成模式" size="default" class="modern-select" @change="handleChange" popper-class="mode-select-dropdown">
            <template #prefix>
              <div class="select-prefix-container">
                <span class="select-prefix">生成模式</span>
              </div>
            </template>
            <el-option 
              v-for="item in options" 
              :key="item.value" 
              :label="item.label" 
              :value="item.value"
              class="mode-option"
            >
              <el-tooltip 
                :content="item.description" 
                placement="top" 
                effect="dark" 
                popper-class="option-tooltip"
                :enterable="false"
                :show-after="300"
              >
                <div class="option-content">
                  <span class="option-label">{{ item.label }}</span>
                  <span class="option-icon">
                    <i class="el-icon-check" v-if="selectedOption === item.value"></i>
                  </span>
                </div>
              </el-tooltip>
            </el-option>
          </el-select>
          <div class="mode-tooltip">
            <el-tooltip placement="right" effect="light" popper-class="custom-mode-tooltip">
              <template #content>
                <div class="tooltip-content">
                  <p class="tooltip-title">{{ selectedOptionLabel }}</p>
                  <p class="tooltip-description">
                    {{ getModeTip(selectedOption) }}
                  </p>
                </div>
              </template>
              <i class="mode-info-icon el-icon-question"></i>
            </el-tooltip>
          </div>
        </div>

        <!-- 分割线 - 添加淡入淡出效果 -->
        <div class="sidebar-divider" v-if="!isCollapsed"></div>

        <!-- 历史对话 -->
        <div class="history" v-if="!isCollapsed">
          <div class="history-header">
            <span>历史对话</span>
            <el-tooltip content="清除所有历史记录" placement="top">
              <img class="delete-icon" src="../assets/delete_history.png" width="20px"
                height="20px" @click="clearAllChatHistory" />
            </el-tooltip>
          </div>

          <!-- 历史记录列表部分 -->
          <div class="history-list">
            <template v-if="totalChatHistory.length > 0">
              <div v-for="chat in totalChatHistory" :key="chat.id" class="history-item"
                :class="{ 'active-chat': chat.id === currentChatId }" @click="selectChatSession(chat.id)">
                <span class="history-text" :title="chat.title">
                  {{ chat.title || '新对话' }}
                </span>
                <el-tooltip content="删除记录" placement="top">
                  <!-- 添加.stop修饰符阻止事件冒泡 -->
                  <img class="delete-item-icon" src="../assets/delete_history.png" 
                    @click.stop="deleteChatSession(chat.id)" width="20px" height="20px" />
                </el-tooltip>
              </div>
            </template>
            <div v-else class="empty-history">
              <img src="../assets/no_history.png" class="empty-icon" alt="空状态">
              <p class="empty-text">暂无历史对话</p>
              <p class="empty-tip">开始新的对话吧～</p>
            </div>
          </div>
        </div>

        <!-- 个人信息 -->
        <div class="user-info" v-if="!isCollapsed">
          <img src="../assets/user_img.png" alt="用户头像" class="user-avatar">
          <span class="username">{{UserName}}</span>
          <el-tooltip effect="dark" content="退出登录" placement="top" popper-class="custom-tooltip">
            <img class="setting-icon" src="../assets/leave.png" alt="设置" @click="tologin">
          </el-tooltip>
        </div>
      </el-aside>

      <!-- 主内容区 -->
      <el-container class="main-container">
        <el-header class="chat-header">
          <div class="conversation-title" @click="startEditingTitle" v-if="!isEditingTitle">
            {{ currentChatTitle || '新对话' }}
            <el-tooltip content="点击修改标题" effect="dark" placement="top" v-if="currentChatId">
              <i class="el-icon-edit-outline title-edit-icon"></i>
            </el-tooltip>
          </div>
          <div class="title-edit-container" v-else>
            <el-input 
              v-model="editingTitle" 
              placeholder="输入对话标题" 
              size="small" 
              ref="titleInputRef"
              @blur="saveEditedTitle"
              @keyup.enter="saveEditedTitle"
              @keyup.esc="cancelEditingTitle"
            ></el-input>
          </div>
          
          <!-- 添加跳转到welcome页面的按钮 -->
          <div class="welcome-btn" @click="goToWelcome">
            <img src="../assets/leave.png" alt="首页" class="home-icon" />
            <span class="home-text">首页</span>
          </div>
        </el-header>
        <el-main class="chat-main">
          <!-- 滚动容器,消息展示区 -->
          <div class="chat-scroll-container" ref="scrollContainer">
            <div class="chat-content-container">
              <ChatContentTop />
              <!-- 使用虚拟滚动优化大量消息渲染 -->
              <template v-if="currentChatHistory.length > 0">
                <div 
                  v-for="(message, index) in visibleMessages" 
                  :key="message.id" 
                  :id="`message-${message.id}`"
                  class="message-item"
                >
                  <ChatMessage 
                    :message="message.text"
                    :messageType="message.type" 
                    :isLoading="message.isLoading" 
                    @regenerate="regenerateResponse"
                    :messageId="message.id" 
                    :references="message.references" 
                    :question="message.question"
                    :isLast="isLastAiMessage(message)" 
                  />
                </div>
              </template>
              <div v-else-if="isChatHistoryLoading" class="loading-messages">
                <div class="loading-spinner">
                  <i class="el-icon-loading"></i>
                  <span>加载消息中...</span>
                </div>
              </div>
            </div>
          </div>

          <!-- 滚动到底部按钮 - 放在滚动容器外部 -->
          <div class="scroll-to-bottom-btn" v-show="showScrollButton" @click="handleScrollToBottom">
            <i class="scroll-bottom-icon">⬇</i>
          </div>

          <!-- 用户输入容器 -->
          <div class="chat-input-container">
            <div class="custom-chat-input">
              <textarea 
                ref="inputRef" 
                v-model="textarea2" 
                @keyup.enter="handleEnter" 
                @keydown.enter="handleShiftEnter"
                @focus="handleInputFocus"
                @blur="handleInputBlur"
                @input="autoResizeTextarea"
                placeholder="给天工AI发送消息"
                class="chat-textarea"
              ></textarea>
              <!-- 发送按钮 -->
              <div class="send-button-container" @click="handleSendMessage">
                <img src="../assets/send_message.png" alt="发送" class="send-button" />
              </div>
              <!-- 停止按钮 -->
              <div v-if="isSending" class="stop-button-container" @click="handleStopMessage">
                <img src="../assets/stop_message.png" alt="停止" class="stop-button" />
              </div>
              <!-- 新增输入提示 -->
              <div class="input-tip" v-if="!isFocused && !textarea2.trim()">
                <span><kbd>Enter</kbd> 发送 &nbsp;•&nbsp; <kbd>Shift+Enter</kbd> 换行</span>
              </div>
            </div>
            <div class="footer-tip">
              <span>内容由AI生成,请仔细甄别</span>
            </div>
          </div>

          <!-- 登录提示弹窗 -->
          <el-dialog v-model="showLoginPrompt" title="登录提示" width="30%" :show-close="true">
            <template #header>
              <div class="dialog-header">
                <span>登录提示</span>
                <i class="el-icon-close" @click="showLoginPrompt = false" style="cursor: pointer;"></i>
              </div>
            </template>

            <div style="text-align: center">
              <p style="margin-bottom: 20px; font-size: 16px;">您需要登录后才能继续操作</p>
              <el-button type="primary" @click="router.push('/login')" style="width: 120px;">
                立即登录
              </el-button>
            </div>
          </el-dialog>

          <!-- 自定义确认弹窗 -->
          <div class="custom-confirm-overlay" v-if="showCustomConfirm">
            <div class="custom-confirm-modal">
              <div class="custom-confirm-header">
                <div :class="['custom-confirm-icon', confirmType]">
                  <i v-if="confirmType === 'warning'" class="el-icon-warning" style="font-size: 24px;"></i>
                  <i v-else-if="confirmType === 'danger'" class="el-icon-delete" style="font-size: 24px;"></i>
                </div>
                <div class="custom-confirm-title">{{ confirmTitle }}</div>
              </div>
              <div class="custom-confirm-content">
                {{ confirmMessage }}
              </div>
              <div class="custom-confirm-actions">
                <button class="custom-confirm-btn cancel" @click="cancelConfirm">{{ cancelText }}</button>
                <button :class="['custom-confirm-btn', confirmButtonClass]" @click="confirmAction">{{ confirmText }}</button>
              </div>
            </div>
          </div>
        </el-main>
      </el-container>
    </el-container>
    <router-view></router-view>
  </div>
</template>

<script setup name="Chat">
import { ref, onMounted, nextTick, computed, onBeforeUnmount, watch } from "vue";
import { useRouter } from "vue-router";
import { ElMessageBox, ElMessage, ElSelect, ElOption } from "element-plus";
import { API_BASE_URL } from "../config";
import ChatMessage from "../components/common/ChatMessage.vue";
import ChatContentTop from "../components/common/ChatContentTop.vue";
import axios from "axios";

/* ------------------ 状态管理 ------------------ */
// 响应式状态：用于管理组件的各种状态
const router = useRouter(); // 路由实例，用于页面导航
const scrollContainer = ref(null); // 滚动容器引用，用于控制消息区域的滚动
const inputRef = ref(null); // 输入框引用，用于聚焦输入框
const titleInputRef = ref(null); // 标题输入框引用
const isCollapsed = ref(false); // 侧边栏折叠状态，控制侧边栏的展开/收起
const showLoginPrompt = ref(false); // 登录提示显示状态，控制登录提示弹窗的显示/隐藏
const textarea2 = ref(""); // 输入框内容，存储用户输入的消息
const currentChatHistory = ref([]); // 当前对话历史，存储当前会话的所有消息
const visibleMessages = ref([]); // 当前可见的消息，用于虚拟滚动
const currentChatId = ref(null); // 当前对话ID，标识当前正在进行的会话
const currentChatTitle = ref(""); // 当前对话标题
const totalChatHistory = ref([]); // 所有对话历史，存储所有历史会话
const isLoading = ref(false); // 加载状态，标识是否正在加载数据
const isSending = ref(false); // 发送状态，标识是否正在发送消息
const isChatHistoryLoading = ref(false); // 消息历史加载状态
const isEditingTitle = ref(false); // 是否正在编辑标题
const editingTitle = ref(""); // 编辑中的标题
let UserName = ref("AI用户"); // 用户名，显示当前登录用户
let abortController = null; // 中止控制器，用于取消正在进行的请求
const selectedOption = ref("知识库生成"); // 选中的生成模式，控制使用哪种方式生成回答
const options = [ // 生成模式选项列表
  { 
    value: "llm", 
    label: "大模型生成",
    description: "直接调用大模型生成回答，适用于通用问题咨询和创意性任务。"
  },
  { 
    value: "kb", 
    label: "知识库生成",
    description: "基于知识库数据生成更准确的专业领域回答，适用于查询专业知识和获取权威信息。" 
  },
];
const shouldAutoScroll = ref(true);
const showScrollButton = ref(false);
const debounceTimer = ref(null); // 用于防抖
const isFocused = ref(false); // 新增输入框焦点状态

// 确认框状态
const showCustomConfirm = ref(false);
const confirmTitle = ref('');
const confirmMessage = ref('');
const confirmType = ref('warning');
const confirmText = ref('确定');
const cancelText = ref('取消');
const confirmAction = ref(() => {});
const confirmButtonClass = computed(() => confirmType.value === 'danger' ? 'danger' : 'confirm');

// 计算属性
const sidebarWidth = computed(() => (isCollapsed.value ? "64px" : "300px"));
const selectedOptionLabel = computed(() => {
  const option = options.find(
    (option) => option.value === selectedOption.value
  );
  return option ? option.label : "";
});
const lastAiMessageId = computed(() => {
  const aiMessages = currentChatHistory.value.filter((m) => m.type === "ai");
  return aiMessages.length ? aiMessages[aiMessages.length - 1].id : null;
});

// 修改虚拟滚动实现
// 之前的方法没有真正实现虚拟滚动，而是简单地设置了全部消息，这会导致大量消息时的性能问题
watch(currentChatHistory, () => {
  updateVisibleMessages();
}, { deep: true });

// 改进的防抖函数，使用闭包保存状态，避免使用ref
const debounce = (fn, delay = 300) => {
  let timer = null;
  return (...args) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      fn(...args);
      timer = null;
    }, delay);
  };
};

// 滚动相关 - 优化滚动处理逻辑
const handleScroll = () => {
  const container = scrollContainer.value;
  if (!container) return;
  
  const threshold = 100; // 增大阈值，更容易检测到底部
  const { scrollTop, scrollHeight, clientHeight } = container;
  const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
  const isNearBottom = distanceFromBottom < threshold;
  
  shouldAutoScroll.value = isNearBottom;
  // 只有当滚动容器有足够内容并且不在底部时才显示按钮
  showScrollButton.value = !isNearBottom && scrollHeight > clientHeight + 200;
};

// 滚动到底部 - 通用函数，供其他地方调用
const scrollToBottom = (behavior = "smooth") => {
  if (!scrollContainer.value) return;
  
  const container = scrollContainer.value;
  
  // 根据behavior参数决定滚动方式
  if (behavior === "auto") {
    // 直接设置scrollTop - 更高效但没有动画
    container.scrollTop = container.scrollHeight;
  } else if (behavior === "smooth") {
    // 使用自定义平滑滚动效果
    smoothScrollToBottom();
  } else {
    // 使用原生scrollTo - 有平滑动画效果
    container.scrollTo({
      top: container.scrollHeight,
      behavior: behavior
    });
  }
};

// 滚动按钮点击处理函数 - 专用于按钮点击
const handleScrollToBottom = () => {
  if (!scrollContainer.value) return;
  
  // 强制设置自动滚动标志为true
  shouldAutoScroll.value = true;
  
  // 使用自定义平滑滚动
  smoothScrollToBottom();
  
  // 设置一个短暂的延时，然后隐藏按钮，提供更好的视觉反馈
  setTimeout(() => {
    showScrollButton.value = false;
  }, 100); // 降低延迟，提高响应速度
};

// 防抖处理滚动事件
const debouncedHandleScroll = debounce(handleScroll, 150); // 增加延迟减少触发频率

// 设置滚动监听
const setupScrollListener = () => {
  const container = scrollContainer.value;
  if (container) {
    // 使用passive: true提高滚动性能
    container.addEventListener("scroll", debouncedHandleScroll, { passive: true });
  }
};

// 真正的虚拟滚动实现
const updateVisibleMessages = () => {
  if (!scrollContainer.value) return;
  
  // 使用requestAnimationFrame减少重绘和计算
  requestAnimationFrame(() => {
    const container = scrollContainer.value;
    const { scrollTop, clientHeight } = container;
    
    // 如果消息数量较少(小于30条)，直接显示全部消息
    if (currentChatHistory.value.length <= 30) {
      visibleMessages.value = currentChatHistory.value;
      return;
    }
    
    // 实现基于可见区域的消息渲染
    // 假设每条消息平均高度为150px，这可以根据实际情况调整
    const estimatedMsgHeight = 150;
    // 获取大约在可视区域的消息索引
    const startIndex = Math.max(0, Math.floor(scrollTop / estimatedMsgHeight) - 5); // 缓冲5条
    const endIndex = Math.min(
      currentChatHistory.value.length,
      Math.ceil((scrollTop + clientHeight) / estimatedMsgHeight) + 5 // 缓冲5条
    );
    
    // 只显示可见区域附近的消息
    visibleMessages.value = currentChatHistory.value.slice(startIndex, endIndex);
  });
};

// 优化的流式响应文本更新
const optimizedUpdateMessageText = debounce((messageId, text) => {
  // 使用map查找而不是每次都遍历全部消息
  const index = currentChatHistory.value.findIndex(msg => msg.id === messageId);
  if (index !== -1) {
    // 使用不可变数据模式更新，避免修改原对象触发不必要的重渲染
    const updatedMsg = { ...currentChatHistory.value[index], text };
    currentChatHistory.value = [
      ...currentChatHistory.value.slice(0, index),
      updatedMsg,
      ...currentChatHistory.value.slice(index + 1)
    ];
  }
}, 50); // 降低到50ms提升流式响应的平滑度

// 分批加载历史消息的实现 (已被1277行的优化版本替代)
const loadMessagesBatchLegacy = (messages, batchSize = 10, delay = 50) => {
  return new Promise((resolve) => {
    const totalMessages = messages.length;
    let loadedCount = 0;
    
    const loadNextBatch = () => {
      const end = Math.min(loadedCount + batchSize, totalMessages);
      const batch = messages.slice(loadedCount, end);
      
      // 添加这批消息
      currentChatHistory.value = [...currentChatHistory.value, ...batch];
      loadedCount = end;
      
      // 更新后滚动到最新消息，但只在最后一批使用动画
      if (loadedCount >= totalMessages) {
        // 最后一批消息 - 使用优化的滚动方法
        nextTick(() => {
          requestAnimationFrame(() => {
            if (scrollContainer.value) {
              smoothScrollToBottom();
            }
          });
        });
      } else {
        // 中间批次 - 使用立即滚动避免多次动画
        nextTick(() => {
          if (scrollContainer.value) {
            scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight;
          }
        });
      }
      
      // 是否加载完所有消息
      if (loadedCount < totalMessages) {
        setTimeout(loadNextBatch, delay);
      } else {
        resolve();
      }
    };
    
    loadNextBatch();
  });
};

// 判断是否为最后一条AI消息
const isLastAiMessage = (msg) => {
  return msg.type === "ai" && msg.id === lastAiMessageId.value;
};

// 确认框相关
const showConfirm = (options) => {
  confirmTitle.value = options.title || '确认操作';
  confirmMessage.value = options.message || '确定要执行此操作吗？';
  confirmType.value = options.type || 'warning'; 
  confirmText.value = options.confirmButtonText || '确定';
  cancelText.value = options.cancelButtonText || '取消';
  confirmAction.value = async () => {
    showCustomConfirm.value = false;
    try {
      if (options.onConfirm) await options.onConfirm();
    } catch (error) {
      ElMessage.error(error.message || "操作失败");
    }
  };
  showCustomConfirm.value = true;
  
  return new Promise((resolve, reject) => {
    confirmAction.value = async () => {
      showCustomConfirm.value = false;
      try {
        if (options.onConfirm) await options.onConfirm();
        resolve(true);
      } catch (error) {
        ElMessage.error(error.message || "操作失败");
        reject(error);
      }
    };
  });
};

const cancelConfirm = () => {
  showCustomConfirm.value = false;
};

/* ------------------ 用户操作 ------------------ */
// 登录相关操作
const tologin = async () => {
  try {
    await showConfirm({
      title: '退出登录',
      message: '确定要退出登录吗？',
      type: 'warning',
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      onConfirm: async () => {
        try {
          const token = localStorage.getItem('token');
          if (!token) {
            console.error('未找到认证令牌');
            return;
          }
          
          const response = await fetch(`${API_BASE_URL}/logout/`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });

          if (!response.ok) {
            throw new Error('退出登录失败');
          }

          localStorage.removeItem("token");
          localStorage.removeItem("mobile");
          localStorage.removeItem("generateMode");
          
          setTimeout(() => {
            router.push("/login");
          }, 200);
        } catch (error) {
          console.error("退出登录失败:", error);
          ElMessage.error("退出登录失败，请重试");
        }
      }
    });
  } catch (e) {
    // 移除非必要的日志
  }
};

// 页面导航
const goToWelcome = () => {
  router.push('/');
};

// 侧边栏操作
const toggleSidebar = () => {
  const scrollPos = scrollContainer.value ? scrollContainer.value.scrollTop : 0;
  document.body.classList.add('sidebar-transitioning');
  isCollapsed.value = !isCollapsed.value;
  
  // 手动隐藏所有tooltip，防止它们在侧边栏切换后依然显示
  const tooltips = document.querySelectorAll('.el-tooltip__popper');
  tooltips.forEach(tooltip => {
    tooltip.style.display = 'none';
  });
  
  nextTick(() => {
    if (scrollContainer.value) {
      scrollContainer.value.scrollTop = scrollPos;
    }
    setTimeout(() => {
      document.body.classList.remove('sidebar-transitioning');
    }, 350);
  });
};

// 生成模式切换
const handleChange = (value) => {
  if (value === "llm") {
    selectedOption.value = "大模型生成";
  } else if (value === "kb") {
    selectedOption.value = "知识库生成";
  }
  localStorage.setItem("generateMode", selectedOption.value);
};

/* ------------------ 消息处理 ------------------ */
// 消息发送相关
const handleEnter = (e) => !e.shiftKey && handleSendMessage();
const handleShiftEnter = (e) => e.shiftKey && null;

// 发送消息
const handleSendMessage = async () => {

  if (!textarea2.value.trim()) return;
  isSending.value = true;

  try {
    // 记录输入文本并清空输入框，提升用户体验
    const userText = textarea2.value;
    textarea2.value = "";
    
    // 始终启用自动滚动
    shouldAutoScroll.value = true;
    
    // 如果没有会话ID，需要确定是新建会话还是使用现有会话
    if (!currentChatId.value) {
      // 检查是否有现有会话可以使用
      if (totalChatHistory.value.length > 0) {
        // 使用最近的会话
        const mostRecentSession = totalChatHistory.value[0];
        currentChatId.value = mostRecentSession.id;
        currentChatTitle.value = mostRecentSession.title;
        // 加载会话消息
        await loadChatSessionMessages(mostRecentSession.id);
      } else {
        // 没有现有会话，创建新会话
        const token = localStorage.getItem('token');
        if (!token) {
          console.error('未找到认证令牌');
          return;
        }
        
        const title = `新对话`;
        
        const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            title: title
          })
        });

        if (!response.ok) throw new Error("创建会话失败");
        
        const data = await response.json();
        
        currentChatId.value = data.id;
        currentChatTitle.value = data.title;
        
        // 添加到会话列表
        const newSession = {
          id: data.id,
          title: data.title,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        };
        
        totalChatHistory.value = [newSession, ...totalChatHistory.value];
      }
    }

    const userMsgId = Date.now().toString();
    const aiMsgId = (Date.now() + 1).toString();
    
    // 创建用户消息对象
    const userMsg = {
      text: userText,
      type: "user",
      id: userMsgId,
      pairedAiId: aiMsgId,
    };
    
    // 创建AI占位消息对象
    const aiMsg = {
      text: "",
      type: "ai",
      id: aiMsgId,
      isLoading: true,
      parentId: userMsgId,
      references: [],
      question: userText,
    };
    
    // 使用批处理添加消息，减少重绘
    requestAnimationFrame(() => {
      // 添加消息到前端显示
      currentChatHistory.value = [...currentChatHistory.value, userMsg, aiMsg];
      
      // 立即强制滚动到底部，确保用户能看到自己发送的消息
      nextTick(() => {
        // 增加小延时确保DOM完全更新
        setTimeout(() => {
          if (scrollContainer.value) {
            // 使用平滑滚动到底部
            smoothScrollToBottom();
          }
        }, 50);
      });
    });
    
    // 保存用户消息到数据库
    await saveMessage(userMsg);
    
    // 保存空的AI占位消息到数据库
    await saveMessage(aiMsg);
    
    // 如果是会话的第一条消息，则使用这条消息的内容作为会话标题
    if (currentChatHistory.value.length === 2) {
      await updateSessionTitle(currentChatId.value, userText);
    }
    
    // 处理AI响应
    await processAIResponse(userText, aiMsg);
  } catch (error) {
    console.error("发送消息失败:", error);
    ElMessage.error("发送消息失败，请重试");
    isSending.value = false;
  }
};

// 停止消息
const handleStopMessage = () => {
  if (abortController) {
    abortController.abort();
    // 移除非必要的日志
  } else {
    // 移除非必要的日志
  }
  isSending.value = false;
};

// 重新生成消息
const regenerateResponse = async (aiMsgId) => {
  const aiMessage = currentChatHistory.value.find((msg) => msg.id === aiMsgId);
  if (!aiMessage) return;

  const userMessage = currentChatHistory.value.find(
    (msg) => msg.id === aiMessage.parentId
  );
  if (userMessage) {
    currentChatHistory.value = currentChatHistory.value.map((msg) =>
      msg.id === aiMsgId
        ? { ...msg, text: "", isLoading: true, references: [] }
        : msg
    );
    
    // 更新消息状态
    if (currentChatId.value) {
      const updatedAiMsg = currentChatHistory.value.find(msg => msg.id === aiMsgId);
      if (updatedAiMsg) {
        await updateMessage(updatedAiMsg);
      }
    }
    
    await processAIResponse(userMessage.text, aiMsgId);
  }
};




/* ------------------ API 调用 ------------------ */
// 处理AI响应
const processAIResponse = async (question, aiMsgOrId) => {
  abortController = new AbortController();
  isSending.value = true;
  
  try {
    // 确定aiMsg
    let aiMsg = aiMsgOrId;
    if (typeof aiMsgOrId === 'number' || typeof aiMsgOrId === 'string') {
      aiMsg = currentChatHistory.value.find((msg) => msg.id === aiMsgOrId);
      if (!aiMsg) {
        throw new Error("找不到对应的AI消息");
      }
    }
    
    let generate_mode = "";
    if (selectedOption.value === "大模型生成") {
      generate_mode = "query_model";
    } else if (selectedOption.value === "知识库生成") {
      generate_mode = "query_rag";
    }

    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      throw new Error("未找到认证令牌，请重新登录");
    }

    // 添加会话ID支持多轮对话
    let apiEndpoint = `${API_BASE_URL}/${generate_mode}/`;
    let requestBody = { question: question };
    
    // 如果有currentChatId则使用多轮对话接口
    if (currentChatId.value) {
      if (selectedOption.value === "大模型生成") {
        apiEndpoint = `${API_BASE_URL}/query_model_chat/`;  // 使用直接大模型多轮对话接口
      } else if (selectedOption.value === "知识库生成") {
        apiEndpoint = `${API_BASE_URL}/query_rag_chat/`;  // 使用知识库RAG多轮对话接口
      }
      
      requestBody = { 
        question: question,
        session_id: currentChatId.value
      };
    }

    const response = await fetch(apiEndpoint, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify(requestBody),
      signal: abortController.signal,
    });

    if (!response.body) throw new Error("未收到有效的流式响应");

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let partialText = "";
    let lastUpdateTime = Date.now();
    const updateInterval = 100; // 控制UI更新频率 (ms)
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let newlineIndex;
      while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
        const line = buffer.substring(0, newlineIndex);
        buffer = buffer.substring(newlineIndex + 1);

        try {
          const { type, data } = JSON.parse(line);

          if (type === "references") {
            // 使用函数查找并更新，避免不必要的全部渲染
            updateAIMessageReferences(aiMsg.id, data);
          } else if (type === "content") {
            partialText += data;
            
            // 控制UI更新频率，使用累积更新策略
            const now = Date.now();
            if (now - lastUpdateTime > updateInterval) {
              // 使用优化后的更新函数
              optimizedUpdateMessageText(aiMsg.id, partialText);
              lastUpdateTime = now;
              
              // 使用requestIdleCallback在浏览器空闲时执行滚动，避免阻塞渲染
              if (shouldAutoScroll.value && scrollContainer.value) {
                if (window.requestIdleCallback) {
                  window.requestIdleCallback(() => {
                    const container = scrollContainer.value;
                    if (container) container.scrollTop = container.scrollHeight;
                  }, { timeout: 100 });
                } else {
                  // 降级处理
                  requestAnimationFrame(() => {
                    const container = scrollContainer.value;
                    if (container) container.scrollTop = container.scrollHeight;
                  });
                }
              }
              
              // 减少数据库更新频率，提高到500字符一次
              if (partialText.length % 500 === 0 && currentChatId.value) {
                const updatedMsg = { ...aiMsg, text: partialText };
                // 使用防抖避免频繁更新数据库
                debouncedUpdateMessage(updatedMsg);
              }
            }
          } else if (type === "error") {
            partialText += data;
            
            // 错误消息立即更新UI
            updateAIMessageContent(aiMsg.id, partialText);
            
            // 出错时立即更新数据库
            if (currentChatId.value) {
              const updatedMsg = { ...aiMsg, text: partialText, isLoading: false };
              await updateMessage(updatedMsg);
            }
          }
        } catch (e) {
          console.error("解析JSON失败:", e);
        }
      }
    }

    // 处理剩余的buffer
    if (buffer) {
      try {
        const { type, data } = JSON.parse(buffer);
        if (type === "content" || type === "error") {
          partialText += data;
        }
      } catch (e) {
        console.error("解析剩余内容失败:", e);
      }
    }

    // 确保最终文本被更新到UI
    currentChatHistory.value = currentChatHistory.value.map((msg) =>
      msg.id === aiMsg.id ? { ...msg, text: partialText, isLoading: false } : msg
    );
    
    // 保存完整的AI消息
    if (currentChatId.value) {
      const completedAiMsg = { ...aiMsg, text: partialText, isLoading: false };
      await updateMessage(completedAiMsg);
    }
    
    // 最后进行一次滚动确保所有内容可见
    await nextTick(() => {
      if (scrollContainer.value && shouldAutoScroll.value) {
        setTimeout(() => {
          // 使用平滑滚动代替直接设置scrollTop
          smoothScrollToBottom();
        }, 100); // 增加延时，确保内容完全渲染
      }
    });
    
  } catch (error) {
    if (error.name === 'AbortError') {
      // 移除非必要的日志
    } else {
      console.error("处理AI响应时出错:", error);
      
      // 在当前AI消息中显示错误信息
      const errorMsg = "抱歉，生成回复时出错: " + error.message;
      
      // 更新界面中的消息
      currentChatHistory.value = currentChatHistory.value.map((msg) =>
        msg.id === aiMsg.id ? { ...msg, text: errorMsg, isLoading: false } : msg
      );
      
      // 更新数据库
      if (currentChatId.value) {
        const updatedMsg = { 
          ...aiMsg, 
          text: errorMsg, 
          isLoading: false 
        };
        await updateMessage(updatedMsg);
      }
    }
  } finally {
    isSending.value = false;
  }
};




/* ------------------ 聊天历史管理 ------------------ */
// 加载用户的所有聊天会话
const loadUserChatSessions = async () => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      return;
    }
    
    const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (!response.ok) {
      throw new Error("获取会话列表失败");
    }
    
    const data = await response.json();
    totalChatHistory.value = data.sessions || [];
    
    // 按更新时间降序排序
    totalChatHistory.value.sort((a, b) => {
      return new Date(b.updated_at) - new Date(a.updated_at);
    });
  } catch (error) {
    console.error("加载聊天历史失败:", error);
    ElMessage.error("加载聊天历史失败");
  }
};

// 加载特定会话的消息
const loadChatSessionMessages = async (sessionId) => {
  isChatHistoryLoading.value = true;
  currentChatHistory.value = []; // 清空当前消息
  
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      return;
    }
    
    const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (!response.ok) {
      throw new Error("获取会话消息失败");
    }
    
    const data = await response.json();
    
    // 转换消息格式以适应前端展示
    const messages = (data.messages || []).map(msg => {
      // 解析 message_references 字符串为对象
      let references = [];
      try {
        if (msg.message_references) {
          // 检查类型，确保只对字符串使用 trim
          if (typeof msg.message_references === 'string' && msg.message_references.trim() !== '') {
            references = JSON.parse(msg.message_references);
          } else if (typeof msg.message_references === 'object') {
            // 已经是对象，直接使用
            references = msg.message_references;
          }
        }
      } catch (error) {
        console.error("解析 message_references 失败:", error, "原始数据:", msg.message_references);
        references = [];
      }
      
      return {
        id: msg.id,
        text: msg.content || "",
        type: msg.message_type,
        parentId: msg.parent_id,
        pairedAiId: msg.paired_ai_id,
        references: references, // 使用解析后的对象，如果解析失败则使用空数组
        question: msg.question || "",
        isLoading: Boolean(msg.is_loading),
        createdAt: msg.created_at
      };
    });
    
    // 使用批处理加载消息，提升性能
    if (messages.length > 20) {
      // 大量消息使用分批加载 (使用第1277行的优化实现)
      await loadMessagesBatch(messages);
    } else {
      // 少量消息直接设置
      currentChatHistory.value = messages;
      
      // 确保DOM更新后再滚动
      nextTick(() => {
        requestAnimationFrame(() => {
          // 使用自定义平滑滚动
          smoothScrollToBottom();
        });
      });
    }
    
    // 查找该会话的标题
    const session = totalChatHistory.value.find(s => s.id === sessionId);
    if (session) {
      currentChatTitle.value = session.title;
      
      // 检查标题是否为默认标题（包含"新对话"或"对话"字样）
      const isDefaultTitle = session.title === '新对话' || 
                            session.title.includes('对话 ');
      
      // 如果是默认标题且有消息记录，则用第一条用户消息更新标题
      if (isDefaultTitle && messages.length > 0) {
        // 查找第一条用户消息
        const firstUserMsg = messages.find(msg => msg.type === 'user');
        if (firstUserMsg && firstUserMsg.text) {
          await updateSessionTitle(sessionId, firstUserMsg.text);
        }
      }
    }
  } catch (error) {
    console.error("加载聊天消息失败:", error);
    ElMessage.error("加载聊天消息失败");
  } finally {
    isChatHistoryLoading.value = false;
    // 更新可见消息
    updateVisibleMessages();
    
    // 强制启用自动滚动
    shouldAutoScroll.value = true;
  }
};

// 创建新的聊天会话
const createNewChatSession = async () => {
  if (!localStorage.getItem("token")) {
    showLoginPrompt.value = true;
    return;
  }
  
  try {
    // 创建新会话
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      return;
    }
    
    const title = `新对话`;
    
    const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        title: title
      })
    });
    
    if (!response.ok) {
      throw new Error("创建会话失败");
    }
    
    const data = await response.json();
    
    // 添加到会话列表
    const newSession = {
      id: data.id,
      title: data.title,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    
    totalChatHistory.value = [newSession, ...totalChatHistory.value];
    
    // 切换到新会话
    await selectChatSession(data.id);
    
    // 提示用户
    ElMessage.success("已创建新对话");
  } catch (error) {
    console.error("创建新会话失败:", error);
    ElMessage.error("创建新会话失败");
  }
};

// 选择聊天会话
const selectChatSession = async (sessionId) => {
  if (sessionId === currentChatId.value) return;
  
  // 立即更新侧边栏选中状态，提供即时反馈
  currentChatId.value = sessionId;
  
  // 查找该会话的标题并预先更新，减少等待感
  const session = totalChatHistory.value.find(s => s.id === sessionId);
  if (session) {
    currentChatTitle.value = session.title;
  }
  
  // 设置加载状态
  isChatHistoryLoading.value = true;
  
  // 使用过渡效果让当前内容淡出
  const container = scrollContainer.value;
  if (container) {
    container.classList.add('content-switching');
  }
  
  // 使用requestAnimationFrame避免阻塞UI
  requestAnimationFrame(() => {
    // 清空当前消息列表
    currentChatHistory.value = [];
    
    // 始终启用自动滚动
    shouldAutoScroll.value = true;
    
    // 使用虚拟滚动和批量加载优化
    loadChatSessionMessages(sessionId).then(() => {
      // 完成加载后移除过渡类
      if (container) {
        container.classList.remove('content-switching');
      }
      
      // 通知Vue更新DOM，并在更新后执行滚动
      nextTick(() => {
        // 双重requestAnimationFrame确保所有DOM更新完成
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            // 流畅滚动到底部
            smoothScrollToBottom(300); // 降低滚动时间提高响应感
            
            // 聚焦输入框并自动调整大小
            if (inputRef.value) {
              inputRef.value.focus();
              autoResizeTextarea();
            }
          });
        });
      });
    }).catch(err => {
      // 出错时也要移除过渡类
      if (container) {
        container.classList.remove('content-switching');
      }
      console.error("加载会话消息失败:", err);
      ElMessage.error("加载会话消息失败");
    });
  });
};

// 优化平滑滚动函数，允许自定义持续时间
const smoothScrollToBottom = (customDuration) => {
  if (!scrollContainer.value) return;
  
  const container = scrollContainer.value;
  const targetPosition = container.scrollHeight;
  const startPosition = container.scrollTop;
  const distance = targetPosition - startPosition;
  
  // 如果距离很小，直接滚动无需动画
  if (distance < 200) {
    container.scrollTop = targetPosition;
    return;
  }
  
  // 否则使用平滑滚动动画
  let startTime = null;
  const duration = customDuration || 500; // 允许外部设置滚动时间
  
  // 取消之前的滚动动画（如果有）
  if (window.scrollAnimationFrame) {
    cancelAnimationFrame(window.scrollAnimationFrame);
  }
  
  // 使用requestAnimationFrame实现平滑滚动
  const animateScroll = (timestamp) => {
    if (!startTime) startTime = timestamp;
    const elapsed = timestamp - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // 使用easeOutQuart缓动函数使滚动更自然
    // 这个函数比easeInOutQuad更有加速感
    const easeProgress = 1 - Math.pow(1 - progress, 4);
    
    container.scrollTop = startPosition + distance * easeProgress;
    
    if (progress < 1) {
      window.scrollAnimationFrame = requestAnimationFrame(animateScroll);
    }
  };
  
  window.scrollAnimationFrame = requestAnimationFrame(animateScroll);
};

// 批量加载消息的优化版本
const loadMessagesBatch = async (messages) => {
  if (messages.length === 0) return;
  
  // 如果消息数量很少，直接设置
  if (messages.length <= 10) {
    currentChatHistory.value = messages;
    return;
  }
  
  // 优先显示最新的10条消息
  const latestMessages = messages.slice(-10);
  currentChatHistory.value = latestMessages;
  
  // 确保最新消息先渲染
  await nextTick();
  
  // 先滚动到底部显示最新消息
  if (scrollContainer.value) {
    scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight;
  }
  
  // 使用定时器分批加载剩余消息，避免UI阻塞
  const remaining = messages.slice(0, -10);
  const batchSize = 20; // 每批加载20条
  
  // 使用Worker或requestIdleCallback批量加载
  if (window.requestIdleCallback) {
    let loadedIndex = 0;
    
    const loadNextBatch = (deadline) => {
      // 当有空闲时间时加载更多消息
      while (deadline.timeRemaining() > 0 && loadedIndex < remaining.length) {
        const endIndex = Math.min(loadedIndex + batchSize, remaining.length);
        const batch = remaining.slice(loadedIndex, endIndex);
        
        // 将批次插入到正确的位置（前面）
        currentChatHistory.value = [...batch, ...currentChatHistory.value];
        
        loadedIndex = endIndex;
        if (loadedIndex >= remaining.length) break;
      }
      
      if (loadedIndex < remaining.length) {
        // 如果还有消息待加载，继续请求空闲时间
        window.requestIdleCallback(loadNextBatch);
      }
    };
    
    window.requestIdleCallback(loadNextBatch);
  } else {
    // 降级处理：使用setTimeout分批加载
    let i = 0;
    const loadBatch = () => {
      const batch = remaining.slice(i, i + batchSize);
      if (batch.length > 0) {
        currentChatHistory.value = [...batch, ...currentChatHistory.value];
        i += batchSize;
        if (i < remaining.length) {
          setTimeout(loadBatch, 0);
        }
      }
    };
    setTimeout(loadBatch, 0);
  }
};

// 删除聊天会话
const deleteChatSession = async (sessionId) => {
  try {
    await showConfirm({
      title: '删除对话',
      message: '确定要删除这个对话吗？此操作不可恢复。',
      type: 'danger',
      confirmButtonText: '确定删除',
      cancelButtonText: '取消',
      onConfirm: async () => {
        try {
          const token = localStorage.getItem('token');
          if (!token) {
            console.error('未找到认证令牌');
            return;
          }
          
          const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, {
            method: 'DELETE',
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });
          
          if (!response.ok) {
            throw new Error("删除会话失败");
          }
          
          // 从列表中移除
          totalChatHistory.value = totalChatHistory.value.filter(s => s.id !== sessionId);
          
          // 如果删除的是当前会话，则清空当前消息
          if (sessionId === currentChatId.value) {
            currentChatHistory.value = [];
            currentChatId.value = null;
            currentChatTitle.value = "";
          }
          
          ElMessage.success("对话已删除");
        } catch (error) {
          console.error("删除会话失败:", error);
          ElMessage.error("删除会话失败");
        }
      }
    });
  } catch (e) {
    // 移除非必要的日志
  }
};

// 清除所有聊天历史
const clearAllChatHistory = async () => {
  if (!localStorage.getItem("token")) {
    showLoginPrompt.value = true;
    return;
  }
  
  if (totalChatHistory.value.length === 0) {
    ElMessage.info("暂无历史对话可清除");
    return;
  }
  
  try {
    await showConfirm({
      title: '清除所有历史',
      message: '确定要清除所有对话历史吗？此操作不可恢复。',
      type: 'danger',
      confirmButtonText: '确定清除',
      cancelButtonText: '取消',
      onConfirm: async () => {
        try {
          const token = localStorage.getItem('token');
          if (!token) {
            console.error('未找到认证令牌');
            throw new Error("未找到认证令牌，请重新登录");
          }
          
          try {
            // 尝试常规删除
            const response = await axios.delete(`${API_BASE_URL}/chat/sessions`, {
              headers: {
                'Authorization': `Bearer ${token}`
              }
            }).catch(error => {
              console.error("常规清除历史错误详情:", error.response || error);
              // 如果常规请求失败，记录但不抛出错误
              return { status: error.response?.status || 500, data: error.response?.data || {} };
            });
            
            // 检查常规删除是否成功
            if (response.status === 200) {
              // 清空会话列表和当前消息
              totalChatHistory.value = [];
              currentChatHistory.value = [];
              currentChatId.value = null;
              currentChatTitle.value = "";
              
              ElMessage.success("所有历史对话已清除");
              return;
            }
            
            // 如果常规删除返回403（权限错误）或404（未找到），尝试使用紧急清理端点
            if (response.status === 403 || response.status === 404 || response.status === 500) {
              // 尝试使用紧急清理端点
              const emergencyResponse = await axios.delete(`${API_BASE_URL}/chat/sessions/emergency-clean`, {
                headers: {
                  'Authorization': `Bearer ${token}`
                }
              });
              
              // 清空会话列表和当前消息
              totalChatHistory.value = [];
              currentChatHistory.value = [];
              currentChatId.value = null;
              currentChatTitle.value = "";
              
              ElMessage.success("所有历史对话已清除");
            }
          } catch (finalError) {
            console.error("清除历史彻底失败:", finalError);
            throw new Error(`清除历史失败: ${finalError.message || "未知错误"}`);
          }
        } catch (error) {
          console.error("清除历史失败:", error);
          throw error;
        }
      }
    });
  } catch (error) {
    console.error("清除历史操作被取消或出错:", error);
  }
};

// 保存消息到数据库
const saveMessage = async (message) => {
  if (!currentChatId.value) {
    console.warn("无法保存消息：当前没有活动的会话ID");
    return;
  }
  
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      return;
    }
    
    // 处理 references 字段，将对象转换为 JSON 字符串
    let referencesJson = '[]';
    if (message.references) {
      try {
        referencesJson = JSON.stringify(message.references);
      } catch (error) {
        console.error("序列化 references 失败:", error);
        referencesJson = '[]';
      }
    }
    
    const messageData = {
      id: message.id,
      session_id: currentChatId.value,
      message_type: message.type,
      content: message.text,
      parent_id: message.parentId || null,
      paired_ai_id: message.pairedAiId || null,
      message_references: referencesJson,
      question: message.question || null,
      is_loading: message.isLoading || false
    };
    
    // 移除非必要的日志
    
    const response = await fetch(`${API_BASE_URL}/chat/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(messageData)
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error("保存消息失败，状态码:", response.status, "错误详情:", errorData);
      throw new Error(`保存消息失败: ${response.status} ${response.statusText}`);
    } else {
      // 移除非必要的日志
    }
  } catch (error) {
    console.error(`保存消息失败 ID=${message.id}:`, error);
  }
};

// 更新消息
const updateMessage = async (message) => {
  if (!currentChatId.value) {
    console.warn("无法更新消息：当前没有活动的会话ID");
    return;
  }
  
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      return;
    }
    
    // 处理 references 字段，将对象转换为 JSON 字符串
    let referencesJson = '[]';
    if (message.references) {
      try {
        referencesJson = JSON.stringify(message.references);
      } catch (error) {
        console.error("序列化 references 失败:", error);
        referencesJson = '[]';
      }
    }
    
    const messageData = {
      content: message.text,
      message_references: referencesJson,
      is_loading: message.isLoading
    };
    
    // 移除非必要的日志
    
    const response = await fetch(`${API_BASE_URL}/chat/messages/${message.id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(messageData)
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error("更新消息失败，状态码:", response.status, "错误详情:", errorData);
      throw new Error(`更新消息失败: ${response.status} ${response.statusText}`);
    } else {
      // 移除非必要的日志
    }
  } catch (error) {
    console.error(`更新消息失败 ID=${message.id}:`, error);
  }
};

// 编辑对话标题
const startEditingTitle = () => {
  if (!currentChatId.value) return;
  
  editingTitle.value = currentChatTitle.value;
  isEditingTitle.value = true;
  
  nextTick(() => {
    if (titleInputRef.value) {
      titleInputRef.value.focus();
    }
  });
};

const saveEditedTitle = async () => {
  if (!currentChatId.value || !isEditingTitle.value) return;
  
  // 去除前后空格
  const newTitle = editingTitle.value.trim();
  
  // 如果为空，使用默认标题
  if (!newTitle) {
    editingTitle.value = `对话 ${new Date().toLocaleTimeString()}`;
  }
  
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      return;
    }
    
    const response = await fetch(`${API_BASE_URL}/chat/sessions/${currentChatId.value}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        title: editingTitle.value
      })
    });
    
    if (!response.ok) {
      throw new Error("更新标题失败");
    }
    
    // 更新当前标题
    currentChatTitle.value = editingTitle.value;
    
    // 更新总会话列表中的标题
    totalChatHistory.value = totalChatHistory.value.map(session => {
      if (session.id === currentChatId.value) {
        return { ...session, title: editingTitle.value };
      }
      return session;
    });
  } catch (error) {
    console.error("更新标题失败:", error);
    ElMessage.error("更新标题失败");
  } finally {
    isEditingTitle.value = false;
  }
};

const cancelEditingTitle = () => {
  isEditingTitle.value = false;
};

// 新增：更新会话标题为第一条消息内容
const updateSessionTitle = async (sessionId, messageText) => {
  if (!sessionId) return;
  
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      console.error('未找到认证令牌');
      return;
    }
    
    // 截取消息的前20个字符作为标题
    let title = messageText.trim();
    if (title.length > 20) {
      title = title.substring(0, 20) + "...";
    }
    
    // 更新数据库中的会话标题
    const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        title: title
      })
    });
    
    if (!response.ok) {
      throw new Error("更新会话标题失败");
    }
    
    // 更新前端显示
    currentChatTitle.value = title;
    
    // 更新会话列表中的标题
    totalChatHistory.value = totalChatHistory.value.map(session => {
      if (session.id === sessionId) {
        return { ...session, title: title };
      }
      return session;
    });
    
    // 移除非必要的日志
  } catch (error) {
    console.error("更新会话标题失败:", error);
  }
};

// 添加新的辅助函数优化更新操作
// 优化更新AI消息内容的函数
const updateAIMessageContent = (messageId, text) => {
  const index = currentChatHistory.value.findIndex(msg => msg.id === messageId);
  if (index !== -1) {
    const newHistory = [...currentChatHistory.value];
    newHistory[index] = { ...newHistory[index], text };
    currentChatHistory.value = newHistory;
  }
};

// 优化更新AI消息引用的函数
const updateAIMessageReferences = (messageId, references) => {
  const index = currentChatHistory.value.findIndex(msg => msg.id === messageId);
  if (index !== -1) {
    const newHistory = [...currentChatHistory.value];
    newHistory[index] = { ...newHistory[index], references };
    currentChatHistory.value = newHistory;
  }
};

// 防抖更新消息到数据库
const debouncedUpdateMessage = debounce(async (message) => {
  await updateMessage(message);
}, 300);

// 优化预加载资源的方法
const preloadAssets = () => {
  // 使用链接预取，提高后续导航的性能
  const prefetchLinks = [
    '../assets/product.png',
    '../assets/aside.png',
    '../assets/newchat.png',
    '../assets/delete_history.png',
    '../assets/user_img.png',
    '../assets/leave.png',
    '../assets/send_message.png',
    '../assets/stop_message.png'
  ];
  
  // 使用IntersectionObserver延迟加载不可见资源
  if ('IntersectionObserver' in window) {
    // 创建一个本地预加载对象数组
    const preloads = prefetchLinks.map(url => ({ url, loaded: false }));
    
    // 创建一个隐藏的div作为观察目标
    const preloadTarget = document.createElement('div');
    preloadTarget.style.position = 'absolute';
    preloadTarget.style.width = '1px';
    preloadTarget.style.height = '1px';
    preloadTarget.style.opacity = '0';
    document.body.appendChild(preloadTarget);
    
    // 创建观察器
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        // 视口中出现时加载资源
        preloads.forEach(item => {
          if (!item.loaded) {
            const img = new Image();
            img.src = item.url;
            item.loaded = true;
          }
        });
        observer.disconnect();
        preloadTarget.remove();
      }
    });
    
    observer.observe(preloadTarget);
  } else {
    // 降级方案 - 使用requestIdleCallback延迟加载
    if (window.requestIdleCallback) {
      window.requestIdleCallback(() => {
        prefetchLinks.forEach(url => {
          const img = new Image();
          img.src = url;
        });
      }, { timeout: 2000 });
    } else {
      // 最终降级 - 使用setTimeout延迟加载
      setTimeout(() => {
        prefetchLinks.forEach(url => {
          const img = new Image();
          img.src = url;
        });
      }, 1000);
    }
  }
};

// 添加获取模式提示的方法
const getModeTip = (mode) => {
  const option = options.find(o => o.value === mode);
  return option ? option.description : "请选择生成模式";
};

/* ------------------ 初始化相关 ------------------ */
// 组件挂载时执行
onMounted(async () => {
  try {
    const mobile = localStorage.getItem("mobile");
    const generateMode = localStorage.getItem("generateMode");
    if (generateMode) {
      selectedOption.value = generateMode;
    }
    if (mobile) {
      UserName = mobile;
    }
    
    // 先预加载资源
    preloadAssets();
    
    // 再设置滚动监听
    setupScrollListener();
    
    // 加载用户的所有聊天历史
    await loadUserChatSessions();
      
    // 从后端获取最近的会话并加载
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        console.error('未找到认证令牌');
        return;
      }
      
      const response = await fetch(`${API_BASE_URL}/chat/sessions?latest=true`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error("获取最近会话失败");
      }
      
      const data = await response.json();
      if (data.sessions && data.sessions.length > 0) {
        // 获取服务器返回的最新会话
        const latestSession = data.sessions[0];
        // 设置当前会话
        currentChatId.value = latestSession.id;
        currentChatTitle.value = latestSession.title;
        // 加载该会话的消息
        await loadChatSessionMessages(latestSession.id);
      }
    } catch (error) {
      console.error("获取最近会话失败:", error);
      // 如果后端API不支持获取最近会话，则使用前端列表的第一个
      if (totalChatHistory.value.length > 0 && !currentChatId.value) {
        const mostRecentSession = totalChatHistory.value[0];
        await selectChatSession(mostRecentSession.id);
      }
    }
    
    nextTick(() => {
      if (scrollContainer.value) {
        scrollContainer.value.scrollTo({
          top: scrollContainer.value.scrollHeight,
          behavior: 'auto'
        });
      }
    });

    if (inputRef.value) {
      inputRef.value.focus();
    }
  } catch (error) {
    console.error("初始化过程中发生了错误:", error);
  }
});

// 组件卸载前执行 - 优化清理逻辑
onBeforeUnmount(() => {
  const container = scrollContainer.value;
  if (container) {
    container.removeEventListener("scroll", debouncedHandleScroll);
  }
  
  // 中止任何进行中的请求
  if (abortController) {
    abortController.abort();
  }
  
  // 清除所有可能的计时器
  if (debounceTimer.value) {
    clearTimeout(debounceTimer.value);
  }
});

// 输入框自动调整高度
const autoResizeTextarea = () => {
  if (!inputRef.value) return;
  
  const textarea = inputRef.value;
  
  // 重置高度以获取正确的scrollHeight
  textarea.style.height = 'auto';
  
  // 根据内容设置高度，同时考虑最小高度和最大高度限制
  const newHeight = Math.min(Math.max(textarea.scrollHeight, 60), 200);
  textarea.style.height = `${newHeight}px`;
};

// 确保在内容变化时也调整高度
watch(textarea2, () => {
  nextTick(() => {
    autoResizeTextarea();
  });
});

// 确保组件挂载后初始化高度
onMounted(() => {
  nextTick(() => {
    autoResizeTextarea();
  });
});

// 输入框焦点处理
const handleInputFocus = () => {
  isFocused.value = true;
  // 延迟后滚动到底部，确保键盘弹出后能正确滚动
  setTimeout(() => {
    scrollToBottom('auto');
    autoResizeTextarea(); // 确保获得焦点时也调整高度
  }, 300);
};

// 输入框失去焦点处理
const handleInputBlur = () => {
  isFocused.value = false;
};
</script>

<style scoped lang="less">
@import '../styles/chat.less';

/* 添加滚动相关的优化样式 */
.chat-scroll-container {
  scroll-behavior: smooth; /* 启用原生平滑滚动 */
  -webkit-overflow-scrolling: touch; /* 优化移动端滚动 */
  transform: translateZ(0); /* 启用GPU加速 */
  will-change: scroll-position; /* 提示浏览器优化滚动性能 */
  overscroll-behavior: contain; /* 防止滚动链接 */
  transition: opacity 0.2s ease;
  
  &.content-switching {
    opacity: 0.6;
  }
}

/* 添加侧边栏增强样式 */
.app-sidebar {
  /* 增强侧边栏视觉效果 */
  background: linear-gradient(to bottom, rgba(255, 255, 255, 0.95), rgba(250, 252, 255, 0.92));
  backdrop-filter: blur(12px);
  box-shadow: 0 2px 24px rgba(0, 0, 0, 0.08);
  border-right: 1px solid rgba(226, 232, 240, 0.9);
  
  /* 平滑过渡效果 */
  transition: all 0.4s cubic-bezier(0.25, 1, 0.5, 1);
}

/* 侧边栏头部优化 */
.sidebar-header {
  padding: 13px 18px;
  height: 60px;
  line-height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(226, 232, 240, 0.9);
  
  .brand-icon {
    height: 34px;
    margin-right: 14px;
    filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.12));
    transition: transform 0.3s ease;
    
    &:hover {
      transform: scale(1.05);
    }
  }
  
  .brand-text {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(120deg, #3b82f6, #8b5cf6, #6366f1);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 1px 1px rgba(0, 0, 0, 0.07);
  }
  
  .action-icon-button {
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
    border-radius: 8px;
    background: #f0f7ff;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    position: relative;
    overflow: hidden;
    
    &::before {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle, rgba(59, 130, 246, 0.12) 0%, rgba(59, 130, 246, 0) 70%);
      transform: translate(-50%, -50%) scale(0);
      opacity: 0;
      transition: transform 0.3s ease, opacity 0.3s ease;
      pointer-events: none;
    }
    
    :deep(.el-icon) {
      color: #3b82f6;
      transition: all 0.2s ease;
      position: relative;
      z-index: 2;
    }
    
    &:hover {
      background: #e6f0ff;
      transform: translateY(-2px);
      box-shadow: 0 4px 10px rgba(59, 130, 246, 0.18);
      
      &::before {
        transform: translate(-50%, -50%) scale(2.5);
        opacity: 1;
      }
      
      :deep(.el-icon) {
        color: #2563eb;
        transform: scale(1.08);
      }
    }
    
    &:active {
      transform: scale(0.92);
      box-shadow: 0 1px 2px rgba(59, 130, 246, 0.1);
      
      &::before {
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(59, 130, 246, 0) 70%);
        transform: translate(-50%, -50%) scale(1);
        opacity: 1;
      }
      
      :deep(.el-icon) {
        transform: scale(0.95);
      }
    }
  }
}

/* 收起侧栏图标 */
.collapse-icon {
  width: 32px;
  height: 32px;
  margin: 0 auto;
  
  :deep(.el-icon) {
    font-size: 16px;
  }
}

/* 新对话按钮优化 */
.new-chat {
  padding: 14px 16px 10px;

  .new-chat-btn {
    height: 44px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.2px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25);
    border: none;
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
    }
    
    &:active {
      transform: translateY(1px);
    }
  }
}

/* 分割线 */
.sidebar-divider {
  height: 1px;
  margin: 12px 16px;
  background: linear-gradient(to right, rgba(226, 232, 240, 0.1), rgba(226, 232, 240, 0.8), rgba(226, 232, 240, 0.1));
}

/* 历史对话区域优化 */
.history {
  padding: 0 14px;
  
  .history-header {
    padding: 8px 6px;
    font-weight: 600;
    color: #4b5563;
    font-size: 15px;
    letter-spacing: 0.2px;
    
    .delete-icon {
      transition: all 0.3s ease;
      opacity: 0.7;
      padding: 5px;
      border-radius: 6px;
      
      &:hover {
        background: rgba(239, 68, 68, 0.1);
        opacity: 1;
        transform: scale(1.1);
      }
    }
  }
  
  .history-list {
    margin-top: 6px;
    
    .history-item {
      margin-bottom: 4px;
      padding: 6px 12px;  /* 降低行高 - 由10px减至6px */
      border-radius: 8px;
      font-size: 14px;
      transition: all 0.25s ease;
      display: flex;
      align-items: center;
      justify-content: space-between;
      
      &:hover {
        background: rgba(243, 244, 246, 0.8);
        transform: translateX(2px);
      }
      
      .history-text {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: #4b5563;
        font-weight: 500;
        text-align: left;  /* 确保文字左对齐 */
        padding-left: 2px; /* 增加小间距使文字更偏左 */
      }
      
      .delete-item-icon {
        opacity: 0;
        transition: all 0.2s ease;
        padding: 4px;
        border-radius: 4px;
        margin-left: 8px; /* 增加与文字的间距 */
        
        &:hover {
          background: rgba(239, 68, 68, 0.1);
        }
      }
      
      &:hover .delete-item-icon {
        opacity: 0.7;
      }
      
      &.active-chat {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding-left: 9px;
        
        .history-text {
          color: #3b82f6;
          font-weight: 600;
        }
        
        .delete-item-icon {
          opacity: 0.7;
        }
      }
    }
  }
  
  .empty-history {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px 0;
    color: #9ca3af;
    
    .empty-icon {
      width: 60px;
      height: 60px;
      margin-bottom: 12px;
      opacity: 0.7;
    }
    
    .empty-text {
      font-size: 14px;
      font-weight: 500;
      margin-bottom: 6px;
    }
    
    .empty-tip {
      font-size: 12px;
      opacity: 0.7;
    }
  }
}

/* 用户信息区域优化 */
.user-info {
  margin-top: auto;
  padding: 16px;
  border-top: 1px solid rgba(226, 232, 240, 0.8);
  display: flex;
  align-items: center;
  background: rgba(249, 250, 251, 0.8);
  
  .user-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    margin-right: 12px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    border: 2px solid white;
    
    &:hover {
      transform: scale(1.05);
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.12);
    }
  }
  
  .username {
    flex: 1;
    font-weight: 500;
    color: #4b5563;
    font-size: 14px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .setting-icon {
    width: 20px;
    height: 20px;
    padding: 4px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.3s ease;
    opacity: 0.7;
    
    &:hover {
      opacity: 1;
      background: rgba(239, 68, 68, 0.1);
      transform: scale(1.1);
    }
  }
}

/* 添加性能优化相关的CSS */
.message-item {
  will-change: transform, opacity;
  transition: opacity 0.3s ease;
  contain: content;
  content-visibility: auto;
  contain-intrinsic-size: 0 150px; /* 提供预估高度，优化渲染 */
}

.loading-messages {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100px;
  margin-top: 20px;
  
  .loading-spinner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    color: #909399;
    
    i {
      font-size: 24px;
    }
  }
}

/* 优化消息淡入效果 - 使用transform替代opacity以提高性能 */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-item {
  animation: fadeIn 0.3s ease;
  backface-visibility: hidden; /* 防止动画过程中的闪烁 */
}

/* 添加消息项的交错动画延迟，使大量消息加载时更平滑 */
/* 使用LESS循环创建交错延迟 */
.message-item:nth-child(1) { animation-delay: 0.05s; }
.message-item:nth-child(2) { animation-delay: 0.1s; }
.message-item:nth-child(3) { animation-delay: 0.15s; }
.message-item:nth-child(4) { animation-delay: 0.2s; }
.message-item:nth-child(5) { animation-delay: 0.25s; }
.message-item:nth-child(6) { animation-delay: 0.3s; }
.message-item:nth-child(7) { animation-delay: 0.35s; }
.message-item:nth-child(8) { animation-delay: 0.4s; }
.message-item:nth-child(9) { animation-delay: 0.45s; }
.message-item:nth-child(10) { animation-delay: 0.5s; }

/* 滚动到底部按钮优化 - 修复位置固定问题 */
.scroll-to-bottom-btn {
  position: fixed; /* 使用fixed定位 */
  bottom: 120px; /* 距离底部120px，避免与输入框重叠 */
  right: 30px; /* 距离右侧30px */
  width: 40px; /* 设置固定宽度 */
  height: 40px; /* 设置固定高度 */
  border-radius: 50%; /* 圆形按钮 */
  background-color: #ffffff; /* 白色背景 */
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15); /* 添加阴影效果 */
  display: flex; /* 使用flex布局使图标居中 */
  align-items: center; /* 垂直居中 */
  justify-content: center; /* 水平居中 */
  cursor: pointer; /* 鼠标指针样式 */
  z-index: 9999; /* 使用更高的z-index确保按钮显示在最上层 */
  transform: translateZ(0); /* 启用GPU加速 */
  transition: all 0.2s ease; /* 平滑过渡效果 */
  opacity: 0.85;
  
  &:hover {
    opacity: 1;
    transform: translateZ(0) scale(1.1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  }
  
  /* 按钮内的图标样式 */
  .scroll-bottom-icon {
    font-size: 18px;
    color: #409EFF; /* 使用Element Plus主色调 */
    font-weight: bold;
  }
}

/* 增加主内容区样式以支持按钮的定位 */
.chat-main {
  display: flex;
  flex-direction: column;
  padding: 0;
  position: relative;
  height: 100%;
  overflow: hidden;
}

/* 生成模式选择相关样式 */
.model-select-container {
  margin: 15px 0;
  padding: 0 16px;
  position: relative;
  display: flex;
  align-items: center;
  
  .modern-select {
    width: 100%;
    border-radius: 8px;
    
    :deep(.el-input__wrapper) {
      background-color: #f5f7fa;
      border-radius: 8px;
      box-shadow: none;
      border: 1px solid transparent;
      transition: all 0.3s ease;
      
      &:hover, &.is-focus {
        border-color: #dcdfe6;
        box-shadow: 0 0 0 1px rgba(64, 158, 255, 0.1);
      }
    }
    
    :deep(.el-input__inner) {
      font-size: 14px;
      color: #606266;
      height: 36px;
    }
  }
  
  .select-prefix-container {
    margin-right: 8px;
    
    .select-prefix {
      color: #909399;
      font-size: 13px;
      font-weight: 500;
    }
  }
  
  .mode-tooltip {
    margin-left: 8px;
    
    .mode-info-icon {
      color: #909399;
      font-size: 16px;
      cursor: pointer;
      transition: color 0.2s ease;
      
      &:hover {
        color: #409EFF;
      }
    }
  }
}

:deep(.custom-mode-tooltip) {
  max-width: 280px;
  padding: 12px 16px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  
  .tooltip-content {
    .tooltip-title {
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 8px;
      color: #303133;
    }
    
    .tooltip-description {
      font-size: 13px;
      line-height: 1.5;
      color: #606266;
      margin: 0;
    }
  }
}

:deep(.el-select-dropdown__item) {
  padding: 0 15px;
  
  &.selected {
    color: #409EFF;
    font-weight: 500;
    background-color: rgba(64, 158, 255, 0.08);
  }
  
  &:hover {
    background-color: #f5f7fa;
  }
  
  .option-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 34px;
    
    .option-label {
      font-size: 14px;
    }
    
    .option-icon {
      color: #409EFF;
    }
  }
}

/* 添加下拉选项悬浮提示样式 */
:deep(.option-tooltip) {
  max-width: 240px;
  padding: 10px 12px;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  font-size: 13px;
  line-height: 1.5;
  background-color: rgba(0, 0, 0, 0.85) !important;
  color: #ffffff !important;
}

:deep(.mode-select-dropdown) {
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  
  .el-select-dropdown__item {
    height: auto;
    padding: 8px 15px;
  }
}

/* 收起侧栏状态下的新建对话图标样式 */
.collapsed-new-chat {
  display: flex;
  justify-content: center;
  padding: 16px 0;
  
  .new-chat-icon {
    width: 36px;
    height: 36px;
    position: relative;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    
    &::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      border-radius: 8px;
      background: rgba(59, 130, 246, 0.08);
      transform: scale(0);
      opacity: 0;
      transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease;
      z-index: 1;
    }
    
    :deep(.el-icon) {
      font-size: 18px;
      transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    &:hover {
      transform: translateY(-3px) scale(1.05);
      box-shadow: 0 6px 16px rgba(59, 130, 246, 0.22);
      
      &::before {
        transform: translate(-50%, -50%) scale(3);
        opacity: 1;
      }
      
      &::after {
        transform: scale(1.5);
        opacity: 1;
      }
      
      :deep(.el-icon) {
        color: #2563eb;
        transform: scale(1.15) rotate(180deg);
      }
    }
    
    &:active {
      transform: scale(0.88);
      box-shadow: 0 2px 5px rgba(59, 130, 246, 0.12);
      transition: all 0.15s cubic-bezier(0.34, 1.56, 0.64, 1);
      
      &::before {
        background: radial-gradient(circle, rgba(59, 130, 246, 0.35) 0%, rgba(59, 130, 246, 0) 70%);
        transform: translate(-50%, -50%) scale(0.8);
        opacity: 1;
        transition: all 0.15s ease;
      }
      
      :deep(.el-icon) {
        transform: scale(0.85);
        transition: all 0.15s cubic-bezier(0.34, 1.56, 0.64, 1);
      }
    }
  }
}

/* 提示样式 */
:deep(.action-tooltip) {
  font-size: 13px;
  padding: 6px 10px;
  border-radius: 4px;
  background-color: rgba(45, 55, 72, 0.9);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 9999;
  transition: opacity 0.2s ease !important;
}

/* 历史记录项目优化 */
.history-item {
  transition: background-color 0.15s ease, transform 0.15s ease;
  will-change: background-color, transform;
  position: relative;
  
  &:active {
    transform: scale(0.98);
  }
  
  &.active-chat {
    position: relative;
    overflow: hidden;
    
    &::before {
      content: '';
      position: absolute;
      left: 0;
      top: 0;
      height: 100%;
      width: 3px;
      background: linear-gradient(to bottom, #3b82f6, #6366f1);
      transform: translateZ(0);
    }
  }
  
  /* 增强点击反馈 */
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: transparent;
    opacity: 0;
    transition: opacity 0.2s ease;
    pointer-events: none;
  }
  
  &:hover::after {
    opacity: 0.1;
    background-color: #6366f1;
  }
  
  &:active::after {
    opacity: 0.15;
    background-color: #3b82f6;
    transition: opacity 0.1s ease;
  }
}

.sidebar-transitioning * {
  pointer-events: none;
}

.sidebar-transitioning .app-sidebar {
  will-change: width;
}

/* 确保侧边栏过渡期间不显示任何tooltip */
.sidebar-transitioning .el-popper {
  display: none !important;
}

.chat-scroll-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  height: calc(100vh - 190px); /* 固定高度，减去头部和输入框的高度 */
  max-height: calc(100vh - 190px);
  scroll-behavior: smooth;
}

.chat-input-container {
  padding: 16px 20px;
  border-top: 1px solid #f0f0f0;
  background-color: #fff;
  position: relative;
  z-index: 2;
  margin-top: auto;
}

.container {
  height: 100vh;
  overflow: hidden; /* 防止整个页面出现滚动条 */
}
</style>