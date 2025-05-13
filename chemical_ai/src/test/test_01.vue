<template>
  <div id="common">
    <el-container class="container">
      <!-- 侧边栏区域（已优化） -->
      <el-aside class="app-sidebar" :width="sidebarWidth">
        <!-- 侧边栏头部 -->
        <header class="sidebar-header">
          <template v-if="!isCollapsed">
            <img class="brand-icon" src="../assets/product.png" alt="Logo" />
            <span class="brand-text">天工AI智能助手</span>
          </template>
          <el-tooltip :content="isCollapsed ? '展开侧栏' : '收起侧栏'">
            <img class="collapse-icon" src="../assets/aside.png" @click="toggleSidebar" />
          </el-tooltip>
        </header>

        <!-- 新对话 -->
        <div class="new-chat" v-if="!isCollapsed">
          <el-button type="primary" @click="newChat" class="new-chat-btn">+ 新对话</el-button>
        </div>

        <!-- 分割线 -->
        <el-divider></el-divider>

        <!-- 历史对话 -->
        <div class="history" v-if="!isCollapsed">
          <div class="history-header">
            <span>历史对话</span>
            <el-tooltip content="清除所有历史记录">
              <img class="delete-icon" src="../assets/delete_history.png" @click="clearTotalHistory" width="20px"
                height="20px" />
            </el-tooltip>
          </div>

          <!-- 历史记录列表部分 -->
          <div class="history-list">
            <template v-if="totalChatHistory.length > 0">
              <div v-for="(chat, index) in [...totalChatHistory].reverse()" :key="index" class="history-item"
                @click="selectChat(chat)" :class="{ 'active-chat': chat.id === currentChatId }">
                <span class="history-text">
                  {{ chat.messages?.length > 0 ? chat.messages[0].text : '空对话' }}
                </span>
                <el-tooltip content="删除记录">
                  <!-- 添加.stop修饰符阻止事件冒泡 -->
                  <img class="delete-item-icon" src="../assets/delete_history.png" @click.stop="deleteChat(chat.id)"
                    width="20px" height="20px" />
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
          <img src="../assets/user_img.png" alt="用户头像" width="40px" height="40px">
          <span class="username">{{UserName}}</span>
          <img class="setting-icon" src="../assets/setting.png" alt="设置" width="30px" height="30px" @click="tologin">
        </div>
      </el-aside>

      <!-- 主内容区 -->
      <el-container class="main-container">
        <el-header style="height: 40px;"></el-header>
        <el-main class="chat-main">
          <!-- 滚动容器,消息展示区 -->
          <div class="chat-scroll-container" ref="scrollContainer">
            <div class="chat-content-container">

              <ChatContentTop />
              <ChatMessage v-for="(message, index) in currentChatHistory" :key="index" :message="message.text"
                :messageType="message.type" />
            </div>
          </div>

          <!-- 用户输入容器 -->
          <div class="chat-input-container">
            <div class="custom-chat-input">
              <textarea ref="inputRef" v-model="textarea2" @keyup.enter="handleEnter" @keydown.enter="handleShiftEnter"
                placeholder="给天工AI发送消息,Shift + Enter 换行,Enter 发送"
                style="width: 100%; height: 100%; padding: 10px; box-sizing: border-box; background-color: #f9f9f9; color: rgb(71, 71, 71); font-size: 16px; font-weight: normal; border: none; resize: none;   font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;"></textarea>
              <button class="send-button" @click="handleSendMessage">
                发送<el-icon class="el-icon--right">
                  <Upload />
                </el-icon>
              </button>
            </div>
            <div class="footer-tip">
              <span style="font-size: 13px; color: rgb(155, 154, 153);">内容由AI生成,请仔细甄别</span>
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
        </el-main>
      </el-container>
    </el-container>
    <router-view></router-view> <!-- 路由内容会渲染到这里 -->
  </div>
</template>

<script setup name="Chat">
import { ref, onMounted, nextTick, computed } from "vue";
import { useRouter } from "vue-router";
import { Calendar, Search } from "@element-plus/icons-vue";
import { ElMessageBox, ElMessage } from "element-plus";
import ChatMessage from "../components/ChatMessage.vue"; // 引入新的 ChatMessage 组件
import ChatContentTop from "../components/ChatContentTop.vue";
//登录路由
const router = useRouter();
// 添加滚动容器引用
const scrollContainer = ref(null);
// 控制自动聚焦
const inputRef = ref(null);
/* ------------------ 计算属性 ------------------ */
// 侧边栏宽度：展开时300px，收起时64px
const sidebarWidth = computed(() => (isCollapsed.value ? "64px" : "300px"));
// 侧边栏的收起与展开
const isCollapsed = ref(false);
const toggleAside = () => {
  isCollapsed.value = !isCollapsed.value;
};
// 确认用户是否登录
const showLoginPrompt = ref(false);

//用户输入内容
const textarea2 = ref("");

// 当前页面的聊天记录
const currentChatHistory = ref([]);
const currentChatId = ref(null); // 当前对话唯一标识
// 总的历史记录
const totalChatHistory = ref([]); // 结构改为 {id, title, messages, timestamp}

let UserName = ref("AI用户");

// 生成唯一ID
const generateChatId = () =>
  Date.now().toString(36) + Math.random().toString(36).substr(2);

// 检查登录状态
const checkLogin = () => {
  return localStorage.getItem("isAuthenticated") === "true";
};

// 处理回车发送消息
const handleEnter = (e) => {
  if (!e.shiftKey) {
    handleSendMessage();
  }
};
// 处理 Shift + Enter 换行
const handleShiftEnter = (e) => {
  if (e.shiftKey) {
    // 允许换行
    return;
  }
};
// 添加滚动到底部的方法
const scrollToBottom = () => {
  if (scrollContainer.value) {
    scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight;
  }
};

// 切换侧边栏展开/收起
const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value;
};
// 消息发送处理
const handleSendMessage = async () => {
  if (!checkLogin()) {
    showLoginPrompt.value = true;
    return;
  }

  if (textarea2.value.trim()) {
    const newMessage = { text: textarea2.value, type: "user" };
    currentChatHistory.value.push(newMessage); // 将新消息添加到当前聊天记录中
    saveCurrentChatHistoryToLocal(); // 保存当前页面聊天记录到本地
    await nextTick();
    scrollToBottom();
    textarea2.value = ""; // 清空输入框
    // 模拟AI的回应
    setTimeout(async () => {
      currentChatHistory.value.push({
        text: "AI回应: 这里是模拟的AI回复",
        type: "ai",
      });
      saveCurrentChatHistoryToLocal(); // 再次保存更新后的聊天记录
      await nextTick();
      scrollToBottom(); // AI回复后再次滚动
    }, 1000);
  }
};

// 保存当前页面聊天记录到本地
const saveCurrentChatHistoryToLocal = () => {
  localStorage.setItem(
    "currentChatHistory",
    JSON.stringify(currentChatHistory.value)
  );
};

// 从本地加载当前页面聊天记录
const loadCurrentChatHistoryFromLocal = () => {
  const storedChatHistory = localStorage.getItem("currentChatHistory");
  if (storedChatHistory) {
    currentChatHistory.value = JSON.parse(storedChatHistory);
  } else {
    currentChatHistory.value = [];
    localStorage.setItem("currentChatHistory", JSON.stringify([]));
  }
};

// 将当前页面聊天记录加入到总的历史记录中
const addToTotalHistory = () => {
  if (currentChatHistory.value.length > 0) {
    saveCurrentChat();
  }
  // 初始化新对话
  currentChatId.value = generateChatId();
  currentChatHistory.value = [];
  saveCurrentChatHistoryToLocal();
};
// 保存当前对话到总历史记录
const saveCurrentChat = () => {
  const existingIndex = totalChatHistory.value.findIndex(
    (c) => c.id === currentChatId.value
  );
  const newChatData = {
    id: currentChatId.value,
    title: `对话 ${new Date().toLocaleTimeString()}`,
    messages: [...currentChatHistory.value],
    timestamp: Date.now(),
  };

  if (existingIndex > -1) {
    // 更新现有记录
    totalChatHistory.value[existingIndex] = newChatData;
  } else {
    // 添加新记录
    totalChatHistory.value.push(newChatData);
  }

  localStorage.setItem(
    "totalChatHistory",
    JSON.stringify(totalChatHistory.value)
  );
};
// 新建对话
const newChat = () => {
  // 判断是否需要保存（当前有内容且非空对话）
  if (currentChatHistory.value.length > 0) {
    const existingIndex = totalChatHistory.value.findIndex(
      (c) => c.id === currentChatId.value
    );

    // 仅当不存在时才保存
    if (existingIndex === -1) {
      totalChatHistory.value.push({
        id: currentChatId.value,
        title: `对话 ${new Date().toLocaleTimeString()}`,
        messages: [...currentChatHistory.value],
        timestamp: Date.now(),
      });
      localStorage.setItem(
        "totalChatHistory",
        JSON.stringify(totalChatHistory.value)
      );
    }
  }
  // 无论当前是否为空，都强制新建对话（清空当前+新ID）
  currentChatId.value = generateChatId();
  currentChatHistory.value = [];
  localStorage.setItem("currentChatHistory", JSON.stringify([]));
};

// 删除单条历史记录优化
const deleteChat = (chatId) => {
  // 使用深拷贝避免响应式数据干扰
  const newHistory = JSON.parse(JSON.stringify(totalChatHistory.value)).filter(
    (c) => c.id !== chatId
  );

  totalChatHistory.value = newHistory;
  localStorage.setItem("totalChatHistory", JSON.stringify(newHistory));

  // 当删除当前对话时
  if (chatId === currentChatId.value) {
    // 立即清空当前对话显示
    currentChatHistory.value = [];
    // 延迟确保界面更新
    nextTick(() => {
      // 生成新对话并强制刷新状态
      currentChatId.value = generateChatId();
      localStorage.setItem("currentChatHistory", JSON.stringify([]));
      // 清除可能的路由参数
      router.replace({ path: "/chat" });
    });
  }

  ElMessage.success("已删除该对话记录");
};
// 清除全部历史记录优化
const clearTotalHistory = () => {
  ElMessageBox.confirm("确定删除所有历史记录吗？", "警告", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => {
      // 清空历史记录
      totalChatHistory.value = [];
      localStorage.removeItem("totalChatHistory");

      // 重置当前对话
      currentChatId.value = generateChatId();
      currentChatHistory.value = [];
      localStorage.setItem("currentChatHistory", JSON.stringify([]));

      ElMessage.success("已清除所有历史记录");
    })
    .catch(() => {
      // 取消操作
    });
};
// 加载历史对话优化
const selectChat = (chat) => {
  if (currentChatHistory.value.length > 0) {
    saveCurrentChat();
  }

  currentChatId.value = chat.id;
  currentChatHistory.value = [...chat.messages];
  nextTick(scrollToBottom);
};

// 数据加载逻辑优化
onMounted(() => {
  try {
    // 加载总历史记录
    const storedTotal = localStorage.getItem("totalChatHistory");
    totalChatHistory.value = storedTotal ? JSON.parse(storedTotal) : [];

    // 加载当前对话
    const storedCurrent = localStorage.getItem("currentChatHistory");
    if (storedCurrent) {
      const { id, messages } = JSON.parse(storedCurrent);
      currentChatId.value = id || generateChatId();
      currentChatHistory.value = messages || [];
    } else {
      currentChatId.value = generateChatId();
    }
  } catch (e) {
    console.error("数据加载失败:", e);
    initializeFreshChat();
  }
});
const initializeFreshChat = () => {
  currentChatId.value = generateChatId();
  currentChatHistory.value = [];
  totalChatHistory.value = [];
  localStorage.removeItem("totalChatHistory");
  localStorage.removeItem("currentChatHistory");
};
</script>

<style scoped lang="less">
#common {
  height: 100%;
}
.container {
  height: 100%;
  display: flex;
} /* ----- 侧边栏样式（仅此处做了优化） ----- */
.app-sidebar {
  background: #f9fbff;
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
}

.sidebar-header {
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #ebeef5;
  height: 64px;
}

.brand-icon {
  height: 28px;
  margin-right: 8px;
}

.brand-text {
  font-family: "Poppins", sans-serif;
  font-size: 24px;
  font-weight: 600;
  text-align: center;
  display: flex;
  justify-content: space-between;

  /* 渐变文本颜色（蓝色系） */
  background: linear-gradient(to right, #4facfe, #00f2fe); /* 深蓝 → 浅蓝 */
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.collapse-icon {
  cursor: pointer;
  padding: 4px;
  height: 24px;
  transition: transform 0.2s;
}

.collapse-icon:hover {
  transform: scale(1.1);
}

.sidebar-actions {
  padding: 16px;
}

.new-chat {
  display: flex;
  flex: 0.5;
  justify-content: center;
}

.new-chat-btn {
  width: 90%;
  height: 40px;
  border-radius: 5px;
  border-bottom: 2px solid #007bff; /* 蓝色下划线 */
}

/* 历史对话 */
.history {
  padding: 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  flex: 8.5;
}

.history-header {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 10px;
}
.delete-icon {
  padding: 5px;
  border-radius: 4px;
  transition: background-color 0.3s ease-in-out;
}

.delete-icon:hover {
  background-color: #ddd; /* 悬浮时变灰 */
  cursor: pointer;
}
.history-list {
  display: flex;
  flex-direction: column;
}

.history-item {
  display: flex;
  align-items: center;
  padding: 8px;
  cursor: pointer;
  border-radius: 5px;
  transition: background-color 0.3s;
}

.history-item:hover {
  background: #e6e6e6;
}

.history-text {
  margin-left: 8px;
  font-size: 14px;
}
.history-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  transition: background-color 0.3s;
  margin-right: 20px;
  &:hover {
    background: #f5f5f5;

    .delete-item-icon {
      opacity: 1;
      visibility: visible;
    }
  }
}

.delete-item-icon {
  opacity: 0;
  visibility: hidden;
  transition: all 0.3s ease;
  cursor: pointer;
  margin-left: auto; /* 关键属性 */
  padding-left: 10px; /* 保持间距 */
}
.active-chat {
  background: #f0f7ff !important;
  border-left: 3px solid #409eff;
}
/* 个人信息 */
.user-info {
  display: flex;
  flex: 0.5;
  align-items: center;
  padding: 10px;
  border-top: 1px solid #ddd;
}

.username {
  margin-left: 10px;
  font-size: 14px;
}

.setting-icon {
  margin-left: auto;
  cursor: pointer;
  padding: 5px; /* 添加内边距，让悬浮效果更明显 */
  border-radius: 5px; /* 圆角边框，让过渡更柔和 */
  transition: background-color 0.2s ease-in-out; /* 平滑过渡效果 */
}

.setting-icon:hover {
  background-color: #ddd; /* 鼠标悬浮时变为灰色 */
}

.main-container {
  flex: 1;
  height: 100vh;
  overflow: hidden;
  background: #fff;
}

.chat-main {
  height: 100%;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.chat-scroll-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px 0;
  height: calc(100vh - 200px); /* 根据实际情况调整高度 */
}

// 内容显示区域
.chat-content-container {
  max-width: 65%;
  margin: 0 auto;
  padding: 40px 20px;
  min-height: calc(100% - 80px);
}

// 用户输入区域
.chat-input-container {
  max-width: 65%;
  margin: 0 auto;
  padding: 5px 0;
  width: 100%;
  background: #fff;
  border-top: 1px solid #eee;
  position: sticky;
  bottom: 0;
  z-index: 1;
}

.custom-chat-input {
  position: relative;
  border-radius: 24px; /* 增大圆角半径 */
  background-color: #f3f4f6;
  overflow: hidden; /* 确保子元素圆角生效 */
}

.custom-chat-input textarea {
  min-height: 100px;
  max-height: 120px;
  overflow-y: auto;
  outline: none;
  border-radius: 24px; /* 同步设置文本域圆角 */
  padding-right: 80px; /* 为发送按钮留出空间 */
}

.send-button {
  position: absolute;
  bottom: 8px; /* 调整按钮位置适应新圆角 */
  right: 8px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 16px; /* 增大按钮圆角 */
  padding: 8px 16px;
  cursor: pointer;
  transition: background-color 0.3s;
}

/* 新增：输入框聚焦状态样式 */
.custom-chat-input:focus-within {
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.3);
}
.send-button:hover {
  background-color: #0056b3; /* 鼠标悬停时按钮背景色 */
}

/* 滚动条美化 */
.custom-chat-input textarea::-webkit-scrollbar {
  width: 8px;
  background: transparent;
}

.custom-chat-input textarea::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-right: 20px;

  .el-icon-close {
    font-size: 18px;
    color: #999;
    transition: color 0.2s;

    &:hover {
      color: #666;
    }
  }
  .footer-tip {
    margin-top: 20px;
    text-align: center;
  }

  .footer-tip span {
    font-size: 12px;
    color: #474747;
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 4px 8px;
  }
  .chat-input {
    display: flex;
  }
}
.empty-history {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  padding: 20px;
  text-align: center;
  opacity: 0.6;
  animation: fadeIn 0.5s ease;

  .empty-icon {
    width: 80px;
    height: 80px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-text {
    font-size: 16px;
    color: #909399;
    margin-bottom: 8px;
  }

  .empty-tip {
    font-size: 14px;
    color: #c0c4cc;
  }
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 0.6;
    transform: translateY(0);
  }
}
</style>