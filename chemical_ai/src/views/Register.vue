<template>
  <div class="register-page">
    <!-- 背景效果 -->
    <div class="bg-molecules"></div>
    <div class="bg-tech-overlay"></div>
    <div class="bg-data-flow">
      <div v-for="n in 12" :key="`flow-${n}`" class="data-stream"></div>
    </div>
    <div class="bg-nodes">
      <div v-for="n in 8" :key="`node-${n}`" class="node">
        <div class="node-center"></div>
        <div v-for="c in 3" :key="`connection-${n}-${c}`" class="node-connection"></div>
      </div>
      </div>

    <!-- 注册面板 -->
    <div class="register-panel">
      <!-- 左侧品牌区域 -->
      <div class="brand-area">
        <div class="brand-content">
          <img src="../assets/product.png" alt="Logo" class="logo" />
          <h1 class="title">天工AI智能助手</h1>
          <p class="subtitle">智能问答系统 · 专业高效</p>
          </div>
        </div>

      <!-- 右侧注册区域 -->
      <div class="register-area">
        <h2 class="register-title">账号注册</h2>
        


        <!-- 表单区域 -->
        <div class="form-area">
          <!-- 手机号输入 -->
          <div class="input-group">
            <div class="input-icon">
              <el-icon><PhoneFilled /></el-icon>
            </div>
          <el-input 
            v-model="formData.phone" 
              class="form-input" 
            placeholder="请输入11位手机号" 
            @input="handlePhoneInput"
            type="tel"
            name="phone"
            autocomplete="tel"
            maxlength="11"
          />
        </div>
        
          <!-- 验证码输入 -->
          <div class="input-group auth-input-container">
            <div class="input-icon">
              <el-icon><Key /></el-icon>
            </div>
            <div class="code-input-group">
              <el-input 
                v-model="formData.verificationCode" 
                class="form-input code-input" 
                placeholder="请输入6位数验证码"
                @input="handleVerificationCodeInput"
              />
              <button 
                class="code-btn" 
                :disabled="isCodeSent || !isPhoneValid"
                @click="sendVerificationCode"
              >
            {{ isCodeSent ? countdownText : '发送验证码' }}
              </button>
            </div>
        </div>
        
          <!-- 密码输入 -->
          <div class="input-group">
            <div class="input-icon">
              <el-icon><Lock /></el-icon>
            </div>
          <el-input 
            v-model="formData.password" 
              class="form-input" 
            placeholder="请输入密码（至少6位）" 
            type="password"
            show-password
            name="password"
            autocomplete="new-password"
          />
        </div>
        
          <!-- 确认密码输入 -->
          <div class="input-group">
            <div class="input-icon">
              <el-icon><Lock /></el-icon>
            </div>
          <el-input 
            v-model="formData.confirmPassword" 
              class="form-input" 
            placeholder="请再次输入密码" 
            type="password"
            show-password
            name="confirmPassword"
            autocomplete="new-password"
          />
      </div>
      
      <!-- 结果信息 -->
      <div v-if="resultMessage" :class="['result-message', messageType]">
            <span class="message-icon"></span>
            <span class="message-text">{{ resultMessage }}</span>
      </div>

      <!-- 隐私协议 -->
          <div class="agreement-area">
            <el-checkbox v-model="formData.agreed" class="agreement-checkbox">
              我已阅读并同意<a href="javascript:void(0)" class="agreement-link">《用户服务协议》</a>和<a href="javascript:void(0)" class="agreement-link">《隐私政策》</a>
        </el-checkbox>
      </div>

          <!-- 按钮区域 -->
          <div class="button-area">
            <button 
              class="submit-btn" 
              :class="{ disabled: !isFormValid }"
              :disabled="!isFormValid" 
              @click="register"
            >
              <span v-if="!loading">立即注册</span>
              <span v-else class="loading-spinner"></span>
            </button>
            
            <!-- 返回登录按钮 -->
            <button 
              class="register-btn" 
              @click="goToLogin"
            >
              <span>返回登录</span>
              <el-icon class="register-icon"><ArrowLeft /></el-icon>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onUnmounted } from "vue";
import { PhoneFilled, Key, Lock, ArrowLeft } from "@element-plus/icons-vue";
import { useRouter } from "vue-router";
import { API_BASE_URL } from "../config";
import { ElMessage } from 'element-plus';

const router = useRouter();

// 表单数据
const formData = reactive({
  phone: "",
  verificationCode: "",
  password: "",
  confirmPassword: "",
  agreed: false,
});

// 表单验证
const isPhoneValid = computed(() => {
  return /^1[3-9]\d{9}$/.test(formData.phone);
});

const isVerificationCodeValid = computed(() => {
  return /^\d{6}$/.test(formData.verificationCode);
});

const isPasswordValid = computed(() => {
  return formData.password.length >= 6;
});

const isConfirmPasswordValid = computed(() => {
  return formData.password === formData.confirmPassword && formData.confirmPassword.length > 0;
});

// 表单整体验证
const isFormValid = computed(() => {
  return isPhoneValid.value && 
         isVerificationCodeValid.value && 
         isPasswordValid.value && 
         isConfirmPasswordValid.value && 
         formData.agreed;
});

// 输入处理
const handlePhoneInput = (value) => {
  if (typeof value === 'string') {
    // 移除所有非数字字符
    const digitsOnly = value.replace(/\D/g, "");
    
    // 如果以86开头且长度超过11位，则去掉前面的86
    if (digitsOnly.startsWith('86') && digitsOnly.length > 11) {
      formData.phone = digitsOnly.substring(2).slice(0, 11);
    } else {
      // 否则直接取最后11位数字
      formData.phone = digitsOnly.slice(-11);
    }
  }
};

const handleVerificationCodeInput = (value) => {
  formData.verificationCode = value.replace(/\D/g, "").slice(0, 6);
};

// 验证码相关状态
const resultMessage = ref("");
const messageType = ref("info");
const countdown = ref(60);
const isCodeSent = ref(false);
let timer = null;

const countdownText = computed(() => {
  return `${countdown.value.toString().padStart(2, "0")}`;
});

// 清理定时器
onUnmounted(() => {
  if (timer) clearInterval(timer);
});

// 发送验证码
const sendVerificationCode = async () => {
  // 清空之前的提示信息
  resultMessage.value = "";

  // 检查手机号格式
  if (!isPhoneValid.value) {
    resultMessage.value = "请输入有效的11位手机号";
    messageType.value = "warning";
    return;
  }

  // 发送验证码
  try {
    // 提示用户正在发送验证码
    resultMessage.value = "正在发送验证码...";
    messageType.value = "info";

    const response = await fetch(`${API_BASE_URL}/send_sms/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        mobile: formData.phone,
        purpose: 'register'
      })
    });

    const data = await response.json();

    if (response.ok) {
      resultMessage.value = "验证码已发送，请查收";
      messageType.value = "success";
      // 启动倒计时
      isCodeSent.value = true;
      countdown.value = 60;
      if (timer) clearInterval(timer);
      timer = setInterval(() => {
        countdown.value--;
        if (countdown.value <= 0) {
          clearInterval(timer);
          isCodeSent.value = false;
          resultMessage.value = ""; // 清除发送成功提示
        }
      }, 1000);
    } else {
      resultMessage.value = data.message || "验证码发送失败，请稍后重试";
      messageType.value = "error";
    }
  } catch (error) {
    // 网络请求失败的处理
    resultMessage.value = "网络连接异常，请检查网络";
    messageType.value = "error";
  }
};

// 注册处理
const loading = ref(false);

const register = async () => {
  // 验证表单
  if (!isPhoneValid.value) {
    resultMessage.value = "请输入有效的11位手机号";
    messageType.value = "warning";
    return;
  }

  if (!isVerificationCodeValid.value) {
    resultMessage.value = "验证码必须是6位数字";
    messageType.value = "warning";
    return;
  }

  if (!isPasswordValid.value) {
    resultMessage.value = "密码长度不能少于6位";
    messageType.value = "warning";
    return;
  }

  if (!isConfirmPasswordValid.value) {
    resultMessage.value = "两次输入的密码不一致";
    messageType.value = "warning";
    return;
  }

  if (!formData.agreed) {
    resultMessage.value = "请阅读并同意用户协议和隐私政策";
    messageType.value = "warning";
    return;
  }

  // 发送注册请求
  try {
    loading.value = true;
    resultMessage.value = "正在注册...";
    messageType.value = "info";

    const response = await fetch(`${API_BASE_URL}/register/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        mobile: formData.phone,
        code: formData.verificationCode,
        password: formData.password,
        confirm_password: formData.confirmPassword
      })
    });

    const data = await response.json();

    if (response.ok && data.code === 200) {
      // 注册成功，自动登录
      localStorage.setItem("isAuthenticated", "true");
      localStorage.setItem("mobile", formData.phone);
      if (data.token) {
        localStorage.setItem("token", data.token);
        localStorage.setItem("user_id", data.user_id);
      }
      
      resultMessage.value = "注册成功，正在跳转...";
      messageType.value = "success";
      
      // 显示注册成功提示
      ElMessage.success('注册成功');
      
      // 跳转到首页
      setTimeout(() => {
        router.push("/");
      }, 1000);
    } else {
      resultMessage.value = data.message || "注册失败，请稍后重试";
      messageType.value = "error";
    }
  } catch (error) {
    console.error("注册错误:", error);
    resultMessage.value = "网络连接异常，请检查网络";
    messageType.value = "error";
  } finally {
    loading.value = false;
  }
};

// 返回登录页
const goToLogin = () => {
  router.push('/login');
};
</script>

<style scoped>
/* 全局变量 */
:root {
  --primary: #4299e1;
  --primary-light: #63b3ed;
  --primary-dark: #3182ce;
  --primary-bg: #ebf8ff;
  --neutral-50: #f8fafc;
  --neutral-100: #f1f5f9;
  --neutral-200: #e2e8f0;
  --neutral-300: #cbd5e1;
  --neutral-500: #64748b;
  --neutral-600: #475569;
  --neutral-700: #334155;
  --neutral-800: #1e293b;
  --success: #10b981;
  --warning: #f59e0b;
  --error: #ef4444;
  --info: #3b82f6;
}

/* 基础布局 */
.register-page {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  width: 100%;
  overflow: hidden;
  position: relative;
  background-color: #f0f9ff;
}

/* 背景效果 - 分子背景 */
.bg-molecules {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: 
    radial-gradient(circle at 15% 20%, rgba(59, 130, 246, 0.08) 0, transparent 120px),
    radial-gradient(circle at 85% 30%, rgba(49, 130, 206, 0.09) 0, transparent 160px),
    radial-gradient(circle at 35% 70%, rgba(30, 64, 175, 0.07) 0, transparent 140px),
    radial-gradient(circle at 65% 85%, rgba(66, 153, 225, 0.08) 0, transparent 130px);
  background-size: 100% 100%;
  z-index: 0;
}

.bg-tech-overlay {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: 
    radial-gradient(circle at 25% 25%, rgba(191, 219, 254, 0.04) 0, transparent 80px),
    radial-gradient(circle at 75% 40%, rgba(147, 197, 253, 0.04) 0, transparent 100px),
    radial-gradient(circle at 45% 60%, rgba(96, 165, 250, 0.04) 0, transparent 90px),
    radial-gradient(circle at 80% 75%, rgba(59, 130, 246, 0.04) 0, transparent 110px);
  z-index: 1;
  opacity: 0.9;
}

.bg-data-flow {
  position: absolute;
  width: 100%;
  height: 100%;
  z-index: 2;
  pointer-events: none;
}

.data-stream {
  position: absolute;
  height: 2px;
  background: linear-gradient(90deg, 
    transparent 0%, 
    rgba(66, 153, 225, 0.15) 20%, 
    rgba(30, 64, 175, 0.25) 50%,
    rgba(66, 153, 225, 0.15) 80%,
    transparent 100%);
  opacity: 0.6;
  animation: dataStream 8s infinite linear;
  transform-origin: left;
  border-radius: 2px;
}

.data-stream:nth-child(1) {
  width: 20%;
  top: 15%;
  left: 0;
  animation-delay: 0s;
  animation-duration: 7s;
}

.data-stream:nth-child(2) {
  width: 30%;
  top: 25%;
  left: 20%;
  animation-delay: 1s;
  animation-duration: 8s;
}

.data-stream:nth-child(3) {
  width: 25%;
  top: 35%;
  left: 10%;
  animation-delay: 2s;
  animation-duration: 6s;
}

.data-stream:nth-child(4) {
  width: 35%;
  top: 48%;
  left: 30%;
  animation-delay: 3s;
  animation-duration: 9s;
}

.data-stream:nth-child(5) {
  width: 28%;
  top: 62%;
  left: 15%;
  animation-delay: 2.5s;
  animation-duration: 7.5s;
}

.data-stream:nth-child(6) {
  width: 32%;
  top: 78%;
  left: 5%;
  animation-delay: 1.5s;
  animation-duration: 8.5s;
}

.data-stream:nth-child(7) {
  width: 22%;
  top: 85%;
  left: 40%;
  animation-delay: 3.5s;
  animation-duration: 7s;
}

.data-stream:nth-child(8) {
  width: 18%;
  top: 8%;
  left: 50%;
  animation-delay: 4s;
  animation-duration: 6.5s;
}

.data-stream:nth-child(9) {
  width: 28%;
  top: 28%;
  left: 60%;
  animation-delay: 2s;
  animation-duration: 7.5s;
}

.data-stream:nth-child(10) {
  width: 35%;
  top: 40%;
  left: 65%;
  animation-delay: 1s;
  animation-duration: 8s;
}

.data-stream:nth-child(11) {
  width: 30%;
  top: 65%;
  left: 55%;
  animation-delay: 3s;
  animation-duration: 7s;
}

.data-stream:nth-child(12) {
  width: 25%;
  top: 80%;
  left: 70%;
  animation-delay: 2.5s;
  animation-duration: 6.5s;
}

@keyframes dataStream {
  0% {
    transform: scaleX(0);
    opacity: 0.7;
  }
  50% {
    transform: scaleX(1);
    opacity: 0.3;
  }
  100% {
    transform: scaleX(0);
    opacity: 0.7;
  }
}

.bg-nodes {
  position: absolute;
  width: 100%;
  height: 100%;
  z-index: 3;
  pointer-events: none;
}

.node {
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: rgba(66, 153, 225, 0.08);
  box-shadow: 0 0 20px rgba(66, 153, 225, 0.2);
}

.node-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background-color: rgba(30, 64, 175, 0.4);
  animation: pulse 4s infinite ease-in-out;
}

.node-connection {
  position: absolute;
  top: 50%;
  left: 50%;
  height: 2px;
  width: 50px;
  background: linear-gradient(90deg, 
    rgba(59, 130, 246, 0.4) 0%,
    transparent 100%);
  transform-origin: left;
}

.node:nth-child(1) {
  top: 20%;
  left: 25%;
  animation-delay: 0s;
}

.node:nth-child(1) .node-connection:nth-child(2) {
  transform: rotate(45deg);
}

.node:nth-child(1) .node-connection:nth-child(3) {
  transform: rotate(180deg);
}

.node:nth-child(1) .node-connection:nth-child(4) {
  transform: rotate(270deg);
}

/* 其余节点样式与Login.vue相同 */

@keyframes pulse {
  0%, 100% {
    opacity: 0.5;
    transform: translate(-50%, -50%) scale(1);
  }
  50% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1.5);
  }
}

/* 注册面板 */
.register-panel {
  display: flex;
  width: 900px;
  max-width: 90%;
  height: 600px;
  max-height: 90vh;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
  position: relative;
  z-index: 10;
  background-color: white;
}

/* 品牌区域 */
.brand-area {
  width: 40%;
  background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
  color: white;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.brand-area::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: 
    linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, transparent 50%),
    radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.5) 0%, transparent 50%);
  z-index: 1;
}

.brand-area::after {
  content: '';
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M54.627,0.353 L59.383,5.109 C59.533,5.260 59.533,5.503 59.383,5.654 L54.627,10.410 C54.476,10.561 54.233,10.561 54.083,10.410 L49.327,5.654 C49.176,5.503 49.176,5.260 49.327,5.109 L54.083,0.353 C54.233,0.202 54.476,0.202 54.627,0.353 Z' fill='%23FFFFFF' fill-opacity='0.02'/%3E%3C/svg%3E");
  background-size: 60px 60px;
  opacity: 0.4;
  z-index: 0;
}

.brand-content {
  z-index: 2;
  text-align: center;
  padding: 40px 30px;
}

.logo {
  width: 80px;
  height: 80px;
  margin-bottom: 30px;
  filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.2));
  transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.brand-content:hover .logo {
  transform: translateY(-5px);
  filter: drop-shadow(0 8px 12px rgba(0, 0, 0, 0.3));
}

.title {
  font-size: 30px;
  font-weight: 800;
  margin-bottom: 16px;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  letter-spacing: 1px;
}

.subtitle {
  position: relative;
  display: inline-block;
  font-size: 16px;
  opacity: 0.9;
  letter-spacing: 2px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
  padding: 0 30px;
  line-height: 1.6;
  white-space: nowrap;
  min-width: 200px;
}

.subtitle::before,
.subtitle::after {
  content: '';
  position: absolute;
  top: 50%;
  width: 20px;
  height: 1px;
  background-color: rgba(255, 255, 255, 0.5);
}

.subtitle::before {
  left: 0;
}

.subtitle::after {
  right: 0;
}

/* 注册区域 */
.register-area {
  width: 60%;
  padding: 30px;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.register-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--neutral-800);
  margin-bottom: 20px;
  text-align: center;
}

/* 步骤指示器 */
.step-indicator {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-bottom: 20px;
}

.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  z-index: 1;
}

.step-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  background-color: var(--neutral-200);
  color: var(--neutral-600);
  margin-bottom: 8px;
  transition: all 0.3s ease;
}

.step.active .step-number {
  background-color: var(--primary);
  color: white;
  box-shadow: 0 0 0 4px rgba(66, 153, 225, 0.2);
}

.step.completed .step-number {
  background-color: var(--success);
  color: white;
}

.step-label {
  font-size: 14px;
  color: var(--neutral-600);
  font-weight: 500;
}

.step.active .step-label {
  color: var(--primary);
  font-weight: 600;
}

.step.completed .step-label {
  color: var(--success);
}

.step-line {
  flex: 1;
  height: 2px;
  background-color: var(--neutral-200);
  margin: 0 10px;
  margin-bottom: 20px;
}

/* 表单区域 */
.form-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
}

.input-group {
  position: relative;
  margin-bottom: 16px;
  overflow: hidden;
  transition: all 0.3s ease;
}

.input-icon {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--neutral-500);
  font-size: 16px;
  z-index: 2;
}

.form-input {
  width: 100%;
  height: 44px;
  border-radius: 8px !important;
  border: 1px solid var(--neutral-200) !important;
  background-color: var(--neutral-50) !important;
  transition: all 0.2s ease !important;
  padding-left: 38px !important;
  font-size: 14px !important;
}

.form-input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.15) !important;
  background-color: white !important;
}

.code-input-group {
  display: flex;
  align-items: center;
  width: 100%;
  transition: all 0.3s ease;
}

.code-input {
  border-top-right-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
  transition: all 0.3s ease;
}

.code-btn {
  min-width: 120px;
  height: 48px;
  border: none;
  background-color: #2b6cb0; /* 更深的蓝色，增强可见性 */
  color: #ffffff; /* 确保文字为纯白色 */
  font-weight: 700; /* 加粗字体 */
  border-radius: 0 8px 8px 0;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 16px; /* 增大字体 */
  white-space: nowrap;
  padding: 0 15px;
  box-shadow: 0 4px 8px rgba(49, 130, 206, 0.4); /* 增加阴影 */
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2); /* 添加文字阴影 */
}

.code-btn:hover:not(:disabled) {
  background-color: #1a4e82; /* 悬停时颜色更深 */
  box-shadow: 0 6px 12px rgba(49, 130, 206, 0.5);
  transform: translateY(-2px);
}

.code-btn:disabled {
  background-color: #4299e1;
  opacity: 0.8;
  cursor: not-allowed;
}

/* 结果消息 */
.result-message {
  padding: 12px 16px;
  border-radius: 8px;
  margin: 10px 0 15px;
  display: flex;
  align-items: center;
}

.result-message.success {
  background-color: rgba(16, 185, 129, 0.1);
  border-left: 4px solid var(--success);
}

.result-message.warning {
  background-color: rgba(245, 158, 11, 0.1);
  border-left: 4px solid var(--warning);
}

.result-message.error {
  background-color: rgba(239, 68, 68, 0.1);
  border-left: 4px solid var(--error);
}

.result-message.info {
  background-color: rgba(59, 130, 246, 0.1);
  border-left: 4px solid var(--info);
}

.message-icon {
  display: inline-block;
  width: 20px;
  height: 20px;
  margin-right: 10px;
  position: relative;
}

.message-icon::before {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  text-align: center;
  font-weight: bold;
  font-size: 16px;
}

.success .message-icon::before { content: '✓'; color: var(--success); }
.warning .message-icon::before { content: '!'; color: var(--warning); }
.error .message-icon::before { content: '✕'; color: var(--error); }
.info .message-icon::before { content: 'i'; color: var(--info); font-style: italic; }

.message-text {
  flex: 1;
  font-size: 14px;
  color: var(--neutral-700);
}

/* 协议区域 */
.agreement-area {
  margin-bottom: 20px;
}

.agreement-checkbox {
  font-size: 13px;
  color: var(--neutral-600);
  line-height: 1.5;
  display: flex;
  align-items: flex-start;
}

.agreement-checkbox :deep(.el-checkbox__input) {
  margin-top: 3px;
  align-self: flex-start;
}

.agreement-checkbox :deep(.el-checkbox__label) {
  white-space: normal;
  line-height: 1.5;
  word-break: break-word;
  text-align: left;
  padding-top: 0;
}

.agreement-link {
  color: #3182ce;
  text-decoration: none;
  transition: color 0.2s ease;
  font-weight: 500;
  white-space: nowrap;
}

.agreement-link:hover {
  color: #2c5282;
  text-decoration: underline;
}

/* 按钮区域 */
.button-area {
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
  gap: 15px;
  transition: all 0.3s ease;
}

.submit-btn {
  width: 100%;
  height: 48px;
  border-radius: 8px;
  background-color: #3182ce; /* 更深的蓝色 */
  color: white;
  font-weight: 800; /* 更粗的字体 */
  font-size: 17px; /* 更大的字体 */
  border: none;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 5px 10px rgba(49, 130, 206, 0.5); /* 增强阴影 */
  letter-spacing: 2px;
  position: relative;
  overflow: hidden;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2); /* 添加文字阴影 */
}

.submit-btn:hover:not(:disabled) {
  background-color: #2c5282; /* 悬停时颜色更深 */
  box-shadow: 0 7px 14px rgba(49, 130, 206, 0.6);
  transform: translateY(-3px);
}

.submit-btn.disabled {
  background-color: #63b3ed;
  opacity: 0.8;
  cursor: not-allowed;
  box-shadow: 0 3px 6px rgba(66, 153, 225, 0.3);
}

.loading-spinner {
  display: inline-block;
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 注册按钮 */
.register-btn {
  width: 100%;
  height: 48px;
  border-radius: 8px;
  background-color: rgba(235, 248, 255, 0.7);
  color: #2b6cb0;
  font-weight: 600;
  font-size: 16px;
  border: 2px solid rgba(49, 130, 206, 0.6);
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  letter-spacing: 1px;
  box-shadow: 0 4px 8px rgba(49, 130, 206, 0.15);
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.register-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg, 
    rgba(190, 227, 248, 0) 0%, 
    rgba(190, 227, 248, 0.3) 50%, 
    rgba(190, 227, 248, 0) 100%
  );
  transition: all 0.8s ease;
  z-index: 0;
}

.register-btn span, .register-btn .register-icon {
  position: relative;
  z-index: 1;
}

.register-btn:hover {
  background-color: rgba(235, 248, 255, 0.9);
  border-color: #2b6cb0;
  box-shadow: 0 6px 12px rgba(49, 130, 206, 0.25);
  transform: translateY(-2px);
  color: #1a4e82;
}

.register-btn:hover::before {
  left: 100%;
}

.register-btn:active {
  transform: translateY(0);
  box-shadow: 0 2px 6px rgba(49, 130, 206, 0.2);
}

.register-icon {
  font-size: 16px;
  opacity: 0.8;
  transition: all 0.3s ease;
  transform: translateX(-5px);
}

.register-btn:hover .register-icon {
  opacity: 1;
  transform: translateX(-3px);
}

/* 固定高度容器 */
.auth-input-container {
  min-height: 48px;
  margin-bottom: 20px;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .register-panel {
    flex-direction: column;
    height: auto;
    max-height: 90vh;
  }
  
  .brand-area, .register-area {
    width: 100%;
  }
  
  .brand-area {
    padding: 30px 0;
  }
  
  .brand-content {
    padding: 30px 20px;
    max-width: 300px;
    width: 100%;
  }
  
  .logo {
    width: 65px;
    height: 65px;
    margin-bottom: 20px;
  }
  
  .title {
    font-size: 26px;
    margin-bottom: 12px;
  }
  
  .subtitle {
    font-size: 14px;
    padding: 0 20px;
    letter-spacing: 1.5px;
    min-width: 240px;
    white-space: nowrap;
  }
  
  .subtitle::before,
  .subtitle::after {
    width: 15px;
  }
  
  .register-area {
    padding: 25px 20px;
  }
}

@media (max-width: 480px) {
  .global-message {
    max-width: calc(100% - 40px);
    top: 10px;
    right: 10px;
    padding: 10px 15px 10px 10px;
  }
  
  .code-input-group {
    flex-direction: column;
    align-items: stretch;
  }
  
  .code-input {
    border-radius: 8px !important;
  }
  
  .code-btn {
    height: 50px;
    font-size: 16px;
    margin-top: 12px;
    width: 100%;
    border-radius: 8px;
    box-shadow: 0 5px 10px rgba(49, 130, 206, 0.4); /* 增强阴影 */
    background-color: #3182ce; /* 确保在移动端也有正确的颜色 */
  }
  
  .auth-input-container {
    min-height: 120px; /* 移动端更大的容器高度以适应验证码按钮的位置变化 */
  }
  
  .step-label {
    font-size: 12px;
  }
  
  .agreement-checkbox {
    font-size: 12px;
    line-height: 1.4;
  }
  
  .agreement-checkbox :deep(.el-checkbox__label) {
    padding-left: 4px;
    font-size: 12px;
  }
}
</style>
