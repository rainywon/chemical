<template>
  <div class="login-page">
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
    
    <!-- 全局消息提示 -->
    <transition name="message-fade">
      <div v-if="resultMessage" class="global-message" :class="messageType">
        <span class="message-icon"></span>
        <span class="message-text">{{ resultMessage }}</span>
        <span class="message-close" @click="resultMessage = ''">×</span>
    </div>
    </transition>
    
    <!-- 登录面板 -->
    <div class="login-panel">
      <!-- 左侧品牌区域 -->
      <div class="brand-area">
        <div class="brand-content">
          <img src="../assets/product.png" alt="Logo" class="logo" />
          <h1 class="title">天工AI智能助手</h1>
          <p class="subtitle">智能问答系统 · 专业高效</p>
    </div>
      </div>
    
      <!-- 右侧登录区域 -->
      <div class="login-area">
        <h2 class="login-title">账号登录</h2>

      <!-- 登录方式切换 -->
        <div class="login-tabs">
          <div 
            v-for="(tab, index) in [
              {value: 0, label: '验证码登录', icon: 'Message'},
              {value: 1, label: '密码登录', icon: 'Lock'}, 
              {value: 2, label: '管理员登录', icon: 'User'}
            ]" 
            :key="`tab-${index}`"
            :class="['tab-item', { active: loginMode === tab.value }]"
            @click="loginMode = tab.value"
          >
            <el-icon class="tab-icon"><component :is="tab.icon" /></el-icon>
            <span>{{ tab.label }}</span>
          </div>
      </div>

        <!-- 表单区域 -->
        <form class="form-area" @submit.prevent="login" autocomplete="on" :id="loginMode === 2 ? 'admin-login-form' : 'user-login-form'">
          <!-- 隐藏字段辅助浏览器自动填充 -->
          <input v-if="loginMode === 2" type="text" name="username" autocomplete="username" style="display:none">
          <input v-else type="tel" name="phone" autocomplete="tel" style="display:none">
          
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
            :name="loginMode === 2 ? 'username' : 'phone'"
            :autocomplete="loginMode === 2 ? 'username' : 'tel'"
            :id="loginMode === 2 ? 'admin-username' : 'user-phone'"
            maxlength="11"
            ref="phoneInputRef"
          />
        </div>
        
          <!-- 密码/验证码输入 -->
          <div class="input-group auth-input-container">
            <div class="input-icon">
              <el-icon v-if="loginMode === 0"><Search /></el-icon>
              <el-icon v-else><Lock /></el-icon>
            </div>
            
            <!-- 密码输入框 -->
            <transition name="fade-transform" mode="out-in">
            <el-input 
                v-if="loginMode === 1 || loginMode === 2"
              v-model="formData.password" 
                class="form-input" 
              :placeholder="loginMode === 2 ? '请输入管理员密码' : '请输入密码'" 
              type="password"
              show-password
              name="password"
              :autocomplete="loginMode === 2 ? 'current-password' : 'current-password'"
              :id="loginMode === 2 ? 'admin-password' : 'user-password'"
                key="password-input"
                ref="passwordInputRef"
              />
              
              <!-- 验证码输入框 -->
              <div class="code-input-group" v-else key="code-input-group">
                <el-input 
                  v-model="formData.verificationCode" 
                  class="form-input code-input" 
                  placeholder="请输入6位数验证码"
                  @input="handleVerificationCodeInput"
                  name="one-time-code"
                  autocomplete="one-time-code"
                  id="verification-code"
                  inputmode="numeric"
                  pattern="[0-9]*"
                  ref="verificationCodeInputRef"
                />
                <button 
                  class="code-btn" 
                  :disabled="isCodeSent || !isPhoneValid"
                  @click="sendVerificationCode"
                  type="button"
                >
              {{ isCodeSent ? countdownText : '发送验证码' }}
                </button>
          </div>
            </transition>
      </div>
      

      <!-- 隐私协议或管理员说明 -->
      <div class="agreement-area">
        <template v-if="loginMode !== 2">
          <el-checkbox v-model="formData.agreed" class="agreement-checkbox">
            我已阅读并同意<a href="javascript:void(0)" class="agreement-link">《用户服务协议》</a>和<a href="javascript:void(0)" class="agreement-link">《隐私政策》</a>，未注册手机号将自动完成注册
          </el-checkbox>
        </template>
        <div v-else class="admin-notice">
          <div class="admin-notice-icon">
            <el-icon><Lock /></el-icon>
          </div>
          <div class="admin-notice-content">
            <h4 class="admin-notice-title">管理员专属入口</h4>
            <p class="admin-notice-desc">此入口仅限系统管理员使用，普通用户请使用验证码或密码登录</p>
        </div>
      </div>
      </div>

          
          <!-- 按钮区域 -->
          <div class="button-area">
            <button 
              class="submit-btn" 
              :class="{ disabled: !isFormValid }"
              :disabled="!isFormValid" 
              type="submit"
            >
              <span v-if="!loading">{{ loginMode === 2 ? '管理员登录' : '登录' }}</span>
              <span v-else class="loading-spinner"></span>
            </button>
            
            <!-- 底部按钮/提示 -->
            <button 
              v-if="loginMode !== 2"
              class="register-btn" 
              @click="goToRegister"
              type="button"
            >
              <span>注册账号</span>
              <el-icon class="register-icon"><ArrowRight /></el-icon>
            </button>

          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onUnmounted, watch, onMounted, nextTick } from "vue";
import { PhoneFilled, Search, Lock, Message, User, ArrowRight } from "@element-plus/icons-vue";
import { useRouter } from "vue-router";
import { API_BASE_URL } from "../config";
import axios from "axios";
import { ElMessage } from 'element-plus';

const router = useRouter();

// 登录方式 - 0为验证码登录，1为密码登录，2为管理员登录
const loginMode = ref(0);

// 表单数据
const formData = reactive({
  phone: "",
  verificationCode: "",
  password: "",
  agreed: false,
});

// 登录相关状态
const loading = ref(false);

// 切换登录方式
const toggleLoginType = () => {
  // 清空错误消息
  resultMessage.value = "";
};

// 输入处理
const handlePhoneInput = (value) => {
  // 处理各种可能的输入格式，包括带有+86的号码
  if (typeof value === 'string') {
    // 移除所有非数字字符（包括+号）
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

// 验证计算属性
const isPhoneValid = computed(() => {
  return /^1[3-9]\d{9}$/.test(formData.phone);
});

const isVerificationCodeValid = computed(() => {
  return /^\d{6}$/.test(formData.verificationCode);
});

const isPasswordValid = computed(() => {
  return formData.password.length >= 6;
});

// 表单整体验证
const isFormValid = computed(() => {
  // 基础验证：手机号必须有效
  const baseValid = isPhoneValid.value;
  
  if (loginMode.value === 1) {
    // 普通用户密码登录：必须有密码输入且同意协议
    return baseValid && formData.password.length > 0 && formData.agreed;
  } else if (loginMode.value === 2) {
    // 管理员登录：必须有密码输入（不需要同意协议）
    return baseValid && formData.password.length > 0;
  } else {
    // 验证码登录：必须有验证码输入且同意协议
    return baseValid && formData.verificationCode.length > 0 && formData.agreed;
  }
});

// 验证码相关状态
const resultMessage = ref("");
const countdown = ref(60);
const isCodeSent = ref(false);
let timer = null;
const messageTimeout = ref(null);

const countdownText = computed(() => {
  return `${countdown.value.toString().padStart(2, "0")}`;
});

// 自动聚焦功能
const phoneInputRef = ref(null);
const passwordInputRef = ref(null);
const verificationCodeInputRef = ref(null);

// 使用多种方法尝试聚焦元素，提高成功率
const tryFocusElement = (el) => {
  if (!el) return false;
  
  try {
    // 尝试多种方法聚焦
    el.focus();
    
    // 添加特殊处理，解决某些移动浏览器的问题
    setTimeout(() => {
      el.focus();
      // 一些浏览器需要触发点击才能显示软键盘
      el.click && el.click();
    }, 50);
    
    return true;
  } catch (e) {
    console.warn("聚焦输入框失败:", e);
    return false;
  }
};

const focusPhoneInput = () => {
  setTimeout(() => {
    if (phoneInputRef.value && phoneInputRef.value.$el) {
      // 先尝试直接聚焦组件
      if (typeof phoneInputRef.value.focus === 'function') {
        phoneInputRef.value.focus();
      }
      
      // 再尝试聚焦内部input元素
      const inputEl = phoneInputRef.value.$el.querySelector('input');
      if (inputEl) {
        tryFocusElement(inputEl);
      }
    }
  }, 100);
};

const focusPasswordInput = () => {
  setTimeout(() => {
    if (passwordInputRef.value && passwordInputRef.value.$el) {
      // 先尝试直接聚焦组件
      if (typeof passwordInputRef.value.focus === 'function') {
        passwordInputRef.value.focus();
      }
      
      // 再尝试聚焦内部input元素
      const inputEl = passwordInputRef.value.$el.querySelector('input');
      if (inputEl) {
        tryFocusElement(inputEl);
      }
    }
  }, 100);
};

const focusVerificationCodeInput = () => {
  setTimeout(() => {
    if (verificationCodeInputRef.value && verificationCodeInputRef.value.$el) {
      // 先尝试直接聚焦组件
      if (typeof verificationCodeInputRef.value.focus === 'function') {
        verificationCodeInputRef.value.focus();
      }
      
      // 再尝试聚焦内部input元素
      const inputEl = verificationCodeInputRef.value.$el.querySelector('input');
      if (inputEl) {
        tryFocusElement(inputEl);
      }
    }
  }, 100);
};

// 监听页面挂载
onMounted(() => {
  // 尝试从localStorage获取上次登录的手机号
  const savedPhone = localStorage.getItem("mobile");
  if (savedPhone) {
    formData.phone = savedPhone;
  }
  
  // 延迟执行聚焦，确保组件已完全渲染
  setTimeout(focusPhoneInput, 100);
  
  // 再次尝试聚焦，防止首次不成功
  setTimeout(focusPhoneInput, 300);
  setTimeout(focusPhoneInput, 500);
  
  // 设置一个标志来跟踪是否是首次加载
  const isFirstVisit = sessionStorage.getItem('visited') !== 'true';
  
  // 如果是首次访问，确保自动聚焦
  if (isFirstVisit) {
    // 多次尝试聚焦，提高成功率
    for (let i = 1; i <= 5; i++) {
      setTimeout(focusPhoneInput, i * 200);
    }
    sessionStorage.setItem('visited', 'true');
  }
});

onUnmounted(() => {
  if (timer) clearInterval(timer);
  if (messageTimeout.value) clearTimeout(messageTimeout.value);
});

// 添加消息类型状态
const messageType = ref("info");

// 显示全局消息的方法
const showMessage = (message, type = "info", duration = 3000) => {
  resultMessage.value = message;
  messageType.value = type;
  
  // 如果已经有定时器，清除它
  if (messageTimeout.value) clearTimeout(messageTimeout.value);
  
  // 设置自动关闭
  if (duration > 0) {
    messageTimeout.value = setTimeout(() => {
      resultMessage.value = "";
    }, duration);
  }
  
  // 添加动画效果的延迟处理
  setTimeout(() => {
    const messageIcon = document.querySelector('.message-icon');
    if (messageIcon) {
      messageIcon.classList.add('icon-animated');
      setTimeout(() => {
        messageIcon.classList.remove('icon-animated');
      }, 1000);
    }
  }, 100);
};
  
// 忘记密码处理
const forgotPassword = () => {
  // 检查手机号格式
  if (!isPhoneValid.value) {
    showMessage("请输入有效的11位手机号", "warning");
    return;
  }
  
  // 切换到验证码登录模式
  loginMode.value = 0;
  
  // 提示用户使用验证码重置密码
  showMessage("请使用验证码登录后重置密码", "info");
};

// 修改发送验证码函数，添加消息类型判断
const sendVerificationCode = async () => {
  // 检查是否同意协议（非管理员登录模式才需要检查）
  if (!formData.agreed) {
    showMessage("请勾选同意隐私协议", "warning");
    return;
  }

  // 检查手机号格式
  if (!isPhoneValid.value) {
    showMessage("请输入有效的11位手机号", "warning");
    return;
  }

  // 发送验证码
  try {
    // 提示用户正在发送验证码
    showMessage("正在发送验证码...", "info");

    const response = await fetch(`${API_BASE_URL}/send_sms/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        mobile: formData.phone // 发送的手机号
      })
    });
    
    const data = await response.json();

    if (response.ok) {
      showMessage("验证码已发送，请查收", "success");
      // 启动倒计时
      isCodeSent.value = true;
      countdown.value = 60;
      if (timer) clearInterval(timer);
      timer = setInterval(() => {
        countdown.value--;
        if (countdown.value <= 0) {
          clearInterval(timer);
          isCodeSent.value = false;
        }
      }, 1000);
    } else {
      showMessage(data.message || "验证码发送失败，请稍后重试", "error");
    }
  } catch (error) {
    // 网络请求失败的处理
    showMessage("网络连接异常，请检查网络", "error");
  }
};

// 登录处理
const login = async () => {
  // 检测手机号
  if (!isPhoneValid.value) {
    showMessage("请输入有效的11位手机号", "warning");
    focusPhoneInput();
    return;
  }

  // 根据登录模式检查输入
  if (loginMode.value === 1 || loginMode.value === 2) {
    if (!isPasswordValid.value) {
      showMessage("密码长度不能少于6位", "warning");
      focusPasswordInput();
      return;
    }
    if (loginMode.value === 1 && !formData.agreed) {
      showMessage("请勾选同意隐私协议", "warning");
      return;
    }
  } else {
    if (!isVerificationCodeValid.value) {
      showMessage("验证码必须是6位数字", "warning");
      focusVerificationCodeInput();
      return;
    }
    if (!formData.agreed) {
      showMessage("请勾选同意隐私协议", "warning");
      return;
    }
  }

  // 根据登录逻辑进行登录请求
  try {
    loading.value = true;
    showMessage("正在登录...", "info");
    
    let url, requestData;
    
    if (loginMode.value === 1) {
      // 普通用户密码登录
      url = `${API_BASE_URL}/login/`;
      requestData = {
        mobile: formData.phone,
        password: formData.password,
        mode: 'password'
      };
    } else if (loginMode.value === 2) {
      // 管理员密码登录
      url = `${API_BASE_URL}/admin_login/`;
      requestData = {
        username: formData.phone,
        password: formData.password
      };
    } else {
      // 验证码登录
      url = `${API_BASE_URL}/login/`;
      requestData = {
        mobile: formData.phone,
        code: formData.verificationCode,
        mode: 'code'
      };
    }
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });
    
    const data = await response.json();
    if (data.code === 200) {
      // 存储登录状态和Token
      if (data.data.token && loginMode.value !== 2) {
        localStorage.setItem("mobile", formData.phone);
        localStorage.setItem("token", data.data.token);
      }
      
      // 管理员状态处理
      if (loginMode.value === 2) {
        // 可能还需要保存其他管理员特定信息
        if(data.data.token){
          localStorage.setItem("token", data.data.token);
        }
      } else {
        // 非管理员登录时清除管理员状态
      }
      
      showMessage("登录成功，正在跳转...", "success");
      
      // 跳转到首页或管理员页面
      setTimeout(() => {
        if (loginMode.value === 2) {
          // 管理员登录成功后跳转到管理系统首页
          router.push("/admin");
        } else {
          router.push("/");
        }
      }, 1000);
    } else {
      showMessage(data.message || "登录失败", "error");
      
      // 登录失败后重新聚焦
      setTimeout(() => {
        if (loginMode.value === 0) {
          // 验证码登录失败，清空验证码并重新聚焦
          formData.verificationCode = "";
          focusVerificationCodeInput();
        } else {
          // 密码登录失败，清空密码并重新聚焦
          formData.password = "";
          focusPasswordInput();
        }
      }, 100);
    }
  } catch (error) {
    showMessage("网络连接异常，请检查网络", "error");
    
    // 添加网络异常时的特殊处理
    setTimeout(() => {
      if (loginMode.value === 0) {
        focusVerificationCodeInput();
      } else {
        focusPasswordInput();
      }
    }, 100);
  } finally {
    loading.value = false;
  }
};

// 在script setup中添加注册页面跳转
const goToRegister = () => {
  router.push('/register');
};

// 当登录模式改变时，清除错误消息和相关输入
const loginModeChangeHandler = (newMode) => {
  // 清空验证码或密码
  if (newMode === 0) {
    // 切换到验证码登录
    formData.password = "";
  } else if (newMode === 1 || newMode === 2) {
    // 切换到密码登录或管理员登录
    formData.verificationCode = "";
  }
  
  // 延迟执行，确保DOM更新完成
  setTimeout(() => {
    // 优先聚焦到手机号，如果手机号已有效则聚焦到下一个输入项
    if (!isPhoneValid.value) {
      focusPhoneInput();
    } else {
      if (newMode === 0) {
        // 验证码登录
        focusVerificationCodeInput();
      } else {
        // 密码登录
        focusPasswordInput();
      }
    }
  }, 200);
};

// 监听loginMode变化
watch(loginMode, (newValue) => {
  loginModeChangeHandler(newValue);
});
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
.login-page {
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

.node:nth-child(2) {
  top: 15%;
  right: 30%;
  animation-delay: 1s;
}

.node:nth-child(2) .node-connection:nth-child(2) {
  transform: rotate(120deg);
}

.node:nth-child(2) .node-connection:nth-child(3) {
  transform: rotate(200deg);
}

.node:nth-child(2) .node-connection:nth-child(4) {
  transform: rotate(300deg);
}

.node:nth-child(3) {
  bottom: 35%;
  right: 15%;
  animation-delay: 2s;
}

.node:nth-child(3) .node-connection:nth-child(2) {
  transform: rotate(60deg);
}

.node:nth-child(3) .node-connection:nth-child(3) {
  transform: rotate(150deg);
}

.node:nth-child(3) .node-connection:nth-child(4) {
  transform: rotate(240deg);
}

.node:nth-child(4) {
  bottom: 20%;
  left: 20%;
  animation-delay: 1.5s;
}

.node:nth-child(4) .node-connection:nth-child(2) {
  transform: rotate(30deg);
}

.node:nth-child(4) .node-connection:nth-child(3) {
  transform: rotate(170deg);
}

.node:nth-child(4) .node-connection:nth-child(4) {
  transform: rotate(260deg);
}

.node:nth-child(5) {
  top: 50%;
  left: 12%;
  animation-delay: 2.5s;
}

.node:nth-child(5) .node-connection:nth-child(2) {
  transform: rotate(70deg);
}

.node:nth-child(5) .node-connection:nth-child(3) {
  transform: rotate(190deg);
}

.node:nth-child(5) .node-connection:nth-child(4) {
  transform: rotate(290deg);
}

.node:nth-child(6) {
  top: 65%;
  right: 25%;
  animation-delay: 3s;
}

.node:nth-child(6) .node-connection:nth-child(2) {
  transform: rotate(100deg);
}

.node:nth-child(6) .node-connection:nth-child(3) {
  transform: rotate(210deg);
}

.node:nth-child(6) .node-connection:nth-child(4) {
  transform: rotate(320deg);
}

.node:nth-child(7) {
  top: 40%;
  right: 10%;
  animation-delay: 1s;
}

.node:nth-child(7) .node-connection:nth-child(2) {
  transform: rotate(80deg);
}

.node:nth-child(7) .node-connection:nth-child(3) {
  transform: rotate(160deg);
}

.node:nth-child(7) .node-connection:nth-child(4) {
  transform: rotate(290deg);
}

.node:nth-child(8) {
  top: 60%;
  left: 40%;
  animation-delay: 1s;
}

.node:nth-child(8) .node-connection:nth-child(2) {
  transform: rotate(20deg);
}

.node:nth-child(8) .node-connection:nth-child(3) {
  transform: rotate(130deg);
}

.node:nth-child(8) .node-connection:nth-child(4) {
  transform: rotate(250deg);
}

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

/* 全局消息提示 */
.global-message {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 14px 20px;
  border-radius: 10px;
  background-color: white;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
  display: flex;
  align-items: center;
  max-width: 400px;
  z-index: 9999;
  animation: message-slide-in 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transform-origin: top right;
  border: 1px solid rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
}

.global-message:hover {
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12), 0 3px 6px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.global-message.success {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.03) 100%);
  border-left: 4px solid var(--success);
}

.global-message.warning {
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.03) 100%);
  border-left: 4px solid var(--warning);
}

.global-message.error {
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.03) 100%);
  border-left: 4px solid var(--error);
}

.global-message.info {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(59, 130, 246, 0.03) 100%);
  border-left: 4px solid var(--info);
}

.message-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  margin-right: 12px;
  position: relative;
  border-radius: 50%;
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.message-icon.icon-animated {
  animation: icon-pulse 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes icon-pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}

.success .message-icon {
  background-color: rgba(16, 185, 129, 0.15);
}

.warning .message-icon {
  background-color: rgba(245, 158, 11, 0.15);
}

.error .message-icon {
  background-color: rgba(239, 68, 68, 0.15);
}

.info .message-icon {
  background-color: rgba(59, 130, 246, 0.15);
}

.message-icon::before {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-weight: bold;
  font-size: 14px;
}

.success .message-icon::before { 
  content: '✓'; 
  color: var(--success); 
}

.warning .message-icon::before { 
  content: '!'; 
  color: var(--warning);
  font-weight: 900;
}

.error .message-icon::before { 
  content: '✕'; 
  color: var(--error); 
}

.info .message-icon::before { 
  content: 'i'; 
  color: var(--info); 
  font-style: italic;
  font-weight: 900;
}

.message-text {
  flex: 1;
  font-size: 14px;
  color: var(--neutral-700);
  font-weight: 500;
  line-height: 1.5;
}

.success .message-text {
  color: #065f46;
}

.warning .message-text {
  color: #92400e;
}

.error .message-text {
  color: #b91c1c;
}

.info .message-text {
  color: #1e40af;
}

.message-close {
  margin-left: 12px;
  cursor: pointer;
  font-size: 18px;
  color: var(--neutral-500);
  transition: all 0.2s ease;
  opacity: 0.7;
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}

.message-close:hover {
  color: var(--neutral-800);
  background-color: rgba(0, 0, 0, 0.05);
  opacity: 1;
  transform: rotate(90deg);
}

@keyframes message-slide-in {
  from {
    transform: translateY(-20px) scale(0.95);
    opacity: 0;
  }
  to {
    transform: translateY(0) scale(1);
    opacity: 1;
  }
}

@keyframes message-slide-out {
  from {
    transform: translateY(0) scale(1);
    opacity: 1;
  }
  to {
    transform: translateY(-20px) scale(0.95);
    opacity: 0;
  }
}

.message-fade-enter-active {
  animation: message-slide-in 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.message-fade-leave-active {
  animation: message-slide-out 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.message-fade-enter-from, .message-fade-leave-to {
  opacity: 0;
  transform: translateY(-20px) scale(0.95);
}

/* 在移动设备上的消息样式调整 */
@media (max-width: 480px) {
  .global-message {
    max-width: calc(100% - 40px);
    top: 10px;
    right: 10px;
    padding: 12px 16px;
    border-radius: 8px;
  }
  
  .message-icon {
    width: 22px;
    height: 22px;
    margin-right: 10px;
  }
  
  .message-text {
    font-size: 13px;
  }
  
  .message-close {
    width: 20px;
    height: 20px;
    font-size: 16px;
  }
}

/* 登录面板 */
.login-panel {
  display: flex;
  width: 800px;
  max-width: 90%;
  height: 500px;
  max-height: 80vh;
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

/* 登录区域 */
.login-area {
  width: 60%;
  padding: 30px;
  display: flex;
  flex-direction: column;
}

.login-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--neutral-800);
  margin-bottom: 20px;
  text-align: center;
}

/* 登录选项卡 */
.login-tabs {
  display: flex;
  margin-bottom: 25px;
  border-bottom: 1px solid var(--neutral-200);
  position: relative;
  justify-content: center;
  padding-bottom: 2px;
  overflow: hidden;
  transition: all 0.3s ease;
}

.login-tabs::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 1px;
  background: linear-gradient(
    90deg, 
    transparent 0%, 
    rgba(66, 153, 225, 0.2) 25%, 
    rgba(66, 153, 225, 0.2) 75%, 
    transparent 100%
  );
}

.tab-item {
  padding: 10px 16px;
  margin: 0 8px;
  cursor: pointer;
  color: var(--neutral-600);
  font-size: 14px;
  position: relative;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  text-align: center;
  border-radius: 8px 8px 0 0;
  font-weight: 500;
  letter-spacing: 0.3px;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.tab-item::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(235, 248, 255, 0.8);
  z-index: -1;
  transform: translateY(100%);
  transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  border-radius: 8px 8px 0 0;
  box-shadow: inset 0 -1px 2px rgba(66, 153, 225, 0.1);
}

.tab-item:hover::after {
  transform: translateY(70%);
}

.tab-item.active::after {
  transform: translateY(0);
}

.tab-icon {
  margin-right: 6px;
  font-size: 16px;
  transition: transform 0.3s ease;
}

.tab-item:hover .tab-icon {
  transform: translateY(-2px);
}

.tab-item.active .tab-icon {
  transform: scale(1.1);
}

.tab-item:first-child {
  margin-left: 0;
}

.tab-item:last-child {
  margin-right: 0;
}

.tab-item:hover {
  color: #2c5282; /* 更深的蓝色，提高对比度 */
}

.tab-item.active {
  color: #2c5282; /* 更深的蓝色，提高对比度 */
  font-weight: 600;
  text-shadow: 0 1px 1px rgba(66, 153, 225, 0.1);
}

.tab-item::before {
  content: '';
  position: absolute;
  bottom: 0;
  left: 50%;
  width: 0;
  height: 2px;
  background-color: #2c5282; /* 更深的蓝色，提高对比度 */
  transition: all 0.3s ease;
  transform: translateX(-50%);
  opacity: 0;
}

.tab-item:hover::before {
  width: 30%;
  opacity: 0.6;
}

.tab-item.active::before {
  width: 100%;
  opacity: 1;
}

.tab-item.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: #2c5282; /* 更深的蓝色，提高对比度 */
  transition: transform 0.4s cubic-bezier(0.19, 1, 0.22, 1);
  transform-origin: left;
  animation: tabIndicatorIn 0.4s forwards;
}

@keyframes tabIndicatorIn {
  from {
    transform: scaleX(0);
  }
  to {
    transform: scaleX(1);
  }
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

/* 自动填充样式优化 */
.form-input:-webkit-autofill,
.form-input:-webkit-autofill:hover, 
.form-input:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0px 1000px var(--neutral-50) inset !important;
  -webkit-text-fill-color: var(--neutral-800) !important;
  transition: background-color 5000s ease-in-out 0s !important;
  caret-color: var(--primary) !important;
  border-color: var(--primary) !important;
}

.form-input:focus:-webkit-autofill {
  -webkit-box-shadow: 0 0 0px 1000px white inset !important;
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

/* 选项行 */
.options-row {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 16px;
}

.forgot-password {
  font-size: 13px;
}

.forgot-password a {
  color: var(--neutral-500);
  text-decoration: none;
  transition: color 0.2s ease;
}

.forgot-password a:hover {
  color: var(--primary);
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

.agreement-info {
  font-size: 12px;
  color: var(--neutral-500);
  margin-top: 6px;
  padding-left: 24px;
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
  transform: translateX(3px);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .login-panel {
    flex-direction: column;
    height: auto;
    max-height: 90vh;
  }
  
  .brand-area, .login-area {
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
  
  .login-area {
    padding: 25px 20px;
  }
  
  .login-tabs {
    justify-content: space-around;
    border-bottom-width: 1px;
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
  
  .submit-btn, .register-btn {
    height: 52px;
    font-size: 16px;
    box-shadow: 0 5px 10px rgba(49, 130, 206, 0.4); /* 增强阴影 */
  }
  
  .tab-item {
    font-size: 12px;
    margin: 0 2px;
    padding: 8px 6px;
    letter-spacing: 0;
  }
  
  .tab-icon {
    margin-right: 4px;
  font-size: 14px;
  }
  
  .login-title {
    font-size: 18px;
    margin-bottom: 15px;
  }
  
  .login-tabs {
    margin-bottom: 20px;
    padding-bottom: 0;
  }

  .register-icon {
    font-size: 14px;
  }
  
  .agreement-checkbox {
    font-size: 12px;
    line-height: 1.4;
  }
  
  .agreement-checkbox :deep(.el-checkbox__label) {
    padding-left: 4px;
    font-size: 12px;
  }
  
  .agreement-checkbox :deep(.el-checkbox__input) {
    margin-top: 2px;
  }
  
  .brand-content {
    padding: 20px;
    max-width: 280px;
  width: 100%;
  }
  
  .logo {
    width: 60px;
    height: 60px;
  }
  
  .title {
    font-size: 24px;
    margin-bottom: 10px;
  }
  
  .subtitle {
    font-size: 13px;
    padding: 0 16px;
    letter-spacing: 1px;
    min-width: 220px;
    white-space: nowrap;
    overflow: visible;
  }
  
  .subtitle::before,
  .subtitle::after {
    width: 12px;
  }

  .auth-input-container {
    min-height: 140px; /* 移动端更大的容器高度以适应验证码按钮的位置变化 */
  }
}

@keyframes patternMove {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/* 固定高度容器 */
.auth-input-container {
  min-height: 48px;
  margin-bottom: 20px;
}

/* 添加过渡效果 */
.fade-transform-enter-active,
.fade-transform-leave-active {
  transition: all 0.3s ease;
}

.fade-transform-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.fade-transform-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* 按钮区动画优化 */
.button-area {
  margin-bottom: 20px;
  display: flex;
    flex-direction: column;
  gap: 15px;
  transition: all 0.3s ease;
}

/* 确保移动端样式一致性 */
@media (max-width: 480px) {
  .auth-input-container {
    min-height: 140px; /* 移动端更大的容器高度以适应验证码按钮的位置变化 */
  }
  
  /* ... existing code ... */
}

/* ... existing code ... */

/* 添加化学分子结构背景 */
.login-page::before {
  content: '';
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 200 200'%3E%3Cg fill='none' stroke='%233B82F6' stroke-width='0.5' stroke-opacity='0.05'%3E%3Ccircle cx='100' cy='100' r='40'/%3E%3Ccircle cx='100' cy='100' r='60'/%3E%3Cline x1='60' y1='100' x2='140' y2='100'/%3E%3Cline x1='100' y1='60' x2='100' y2='140'/%3E%3Ccircle cx='60' cy='100' r='5'/%3E%3Ccircle cx='140' cy='100' r='5'/%3E%3Ccircle cx='100' cy='60' r='5'/%3E%3Ccircle cx='100' cy='140' r='5'/%3E%3Cline x1='65' y1='65' x2='135' y2='135'/%3E%3Cline x1='135' y1='65' x2='65' y2='135'/%3E%3Ccircle cx='65' cy='65' r='5'/%3E%3Ccircle cx='135' cy='65' r='5'/%3E%3Ccircle cx='65' cy='135' r='5'/%3E%3Ccircle cx='135' cy='135' r='5'/%3E%3C/g%3E%3C/svg%3E");
  background-size: 200px 200px;
  z-index: 0;
  opacity: 0.6;
}

/* 管理员登录提示样式 */
.admin-notice {
  display: flex;
  background-color: rgba(235, 248, 255, 0.5);
  border-radius: 8px;
  padding: 12px 16px;
  border: 1px dashed rgba(66, 153, 225, 0.4);
  margin-bottom: 5px;
}

.admin-notice-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background-color: rgba(66, 153, 225, 0.15);
  border-radius: 50%;
  margin-right: 14px;
  color: #2b6cb0;
}

.admin-notice-content {
  flex: 1;
}

.admin-notice-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c5282;
  margin: 0 0 4px;
}

.admin-notice-desc {
    font-size: 12px;
  color: var(--neutral-600);
  margin: 0;
  line-height: 1.4;
}

.admin-tip {
  text-align: center;
  font-size: 13px;
  color: var(--neutral-500);
  margin-top: 5px;
  padding: 8px;
  border-radius: 4px;
  background-color: rgba(235, 248, 255, 0.3);
}
</style>

