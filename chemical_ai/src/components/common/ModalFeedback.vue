<template>
  <div class="feedback-modal" v-if="show">
    <div class="modal-content">
      <div class="modal-header">
        <h3>信息反馈</h3>
        <button class="close-button" @click="closeFeedbackModal">×</button>
      </div>
      <div class="modal-body">
        <div class="feedback-type-selector">
          <span class="feedback-label">反馈类型:</span>
          <div class="feedback-type-options">
            <div 
              v-for="type in feedbackTypes" 
              :key="type.value" 
              :class="['feedback-type-option', { active: feedbackType === type.value }]"
              @click="feedbackType = type.value"
            >
              <span class="feedback-type-icon">{{ type.icon }}</span>
              <span>{{ type.label }}</span>
            </div>
          </div>
        </div>
        <div class="feedback-input-container">
          <textarea 
            placeholder="请输入您的建议或反馈..." 
            class="feedback-textarea"
            v-model="feedbackText"
            :class="{ 'has-error': showFeedbackError }"
            @input="showFeedbackError = false"
          ></textarea>
          <div class="feedback-textarea-footer">
            <span class="character-count" :class="{ 'near-limit': feedbackText.length > 450, 'at-limit': feedbackText.length >= 500 }">
              {{ feedbackText.length }}/500
            </span>
            <span class="error-message" v-if="showFeedbackError">
              请输入至少5个字符的反馈内容
            </span>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="cancel-button" @click="closeFeedbackModal" :disabled="feedbackSubmitting">取消</button>
        <button class="submit-button" @click="submitFeedback" :disabled="feedbackSubmitting">
          <span class="loading-spinner" v-if="feedbackSubmitting"></span>
          <span v-else>提交反馈</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
// API配置
import { API_BASE_URL } from "../../config";


// Props
const props = defineProps({
  show: {
    type: Boolean,
    required: true
  },
  message: {
    type: String,
    default: ''
  },
  question: {
    type: String,
    default: ''
  }
});

// Emits
const emit = defineEmits(['close', 'submit-success']);

// 状态变量
const feedbackText = ref('');
const feedbackType = ref('suggestion');
const feedbackSubmitting = ref(false);
const showFeedbackError = ref(false);

// 反馈类型选项
const feedbackTypes = [
  { value: 'suggestion', label: '功能建议', icon: '' },
  { value: 'bug', label: '问题反馈', icon: '' },
  { value: 'content', label: '内容改进', icon: '' },
  { value: 'other', label: '其他', icon: '' },
];

// 关闭反馈弹窗
const closeFeedbackModal = () => {
  if (feedbackSubmitting.value) return;
  resetForm();
  emit('close');
};

// 重置表单
const resetForm = () => {
  feedbackText.value = '';
  feedbackType.value = 'suggestion';
  showFeedbackError.value = false;
};

// 提交反馈
const submitFeedback = async () => {
  // 验证反馈内容
  if (feedbackText.value.trim().length < 5) {
    showFeedbackError.value = true;
    return;
  }
  
  // 限制反馈文本长度
  if (feedbackText.value.length > 500) {
    feedbackText.value = feedbackText.value.substring(0, 500);
  }
  
  // 设置提交状态
  feedbackSubmitting.value = true;
  const token = localStorage.getItem('token');
  
  try {
    const response = await fetch(`${API_BASE_URL}/submit-content-feedback/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        feedback_type: feedbackType.value,
        feedback_content: feedbackText.value,
      }),
    });

    if (!response.ok) {
      throw new Error(`提交失败: ${response.status}`);
    }

    const data = await response.json();
    if (data.code === 200) {
      // 发送成功信号
      emit('submit-success');
      // 关闭弹窗
      closeFeedbackModal();
    } else {
      throw new Error(data.message || '提交失败');
    }
  } catch (error) {
    console.error('提交反馈失败:', error);
    alert(`提交失败: ${error.message}`);
  } finally {
    feedbackSubmitting.value = false;
  }
};
</script>

<style scoped>
.feedback-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.modal-header {
  background: #f8f9fa;
  padding: 15px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #e9ecef;
}

.modal-header h3 {
  margin: 0;
  font-size: 1.1rem;
  color: #1a1f36;
}

.close-button {
  background: transparent;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  line-height: 1;
  color: #6c757d;
}

.modal-body {
  padding: 20px;
}

.feedback-type-selector {
  margin-bottom: 20px;
}

.feedback-label {
  display: block;
  font-size: 0.95rem;
  color: #4a5568;
  margin-bottom: 10px;
  font-weight: 500;
}

.feedback-type-options {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.feedback-type-option {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 8px;
  background: rgba(248, 250, 252, 0.8);
  border: 1px solid #e2e8f0;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.feedback-type-option:hover {
  background: rgba(248, 250, 252, 1);
  border-color: #cbd5e0;
  transform: translateY(-1px);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.feedback-type-option.active {
  background: rgba(79, 70, 229, 0.08);
  border-color: rgba(79, 70, 229, 0.4);
  color: #4f46e5;
  font-weight: 500;
  box-shadow: 0 2px 5px rgba(79, 70, 229, 0.1);
}

.feedback-type-icon {
  font-size: 1.1rem;
}

.feedback-input-container {
  position: relative;
  margin-bottom: 4px;
}

.feedback-textarea {
  width: 90%;
  min-height: 150px;
  padding: 16px;
  border: 1px solid #e9ecef;
  border-radius: 10px;
  font-size: 0.95rem;
  line-height: 1.5;
  resize: vertical;
  transition: all 0.2s ease;
  color: #1a1f36;
  background-color: rgba(248, 250, 252, 0.8);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

.feedback-textarea::placeholder {
  color: #a0aec0;
  font-style: italic;
}

.feedback-textarea:focus {
  outline: none;
  border-color: #4f46e5;
  box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
  background-color: #fff;
}

.feedback-textarea.has-error {
  border-color: #ef4444;
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.15);
}

.feedback-textarea-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  padding: 0 4px;
  font-size: 0.85rem;
}

.character-count {
  color: #6b7280;
  font-weight: 500;
  transition: all 0.2s ease;
  padding: 2px 6px;
  border-radius: 4px;
}

.character-count.near-limit {
  color: #f59e0b;
  background-color: rgba(245, 158, 11, 0.1);
}

.character-count.at-limit {
  color: #ef4444;
  background-color: rgba(239, 68, 68, 0.1);
  font-weight: 600;
}

.error-message {
  color: #ef4444;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 4px;
}

.error-message::before {
  content: "⚠️";
  font-size: 0.85rem;
}

.modal-footer {
  padding: 15px 20px;
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  border-top: 1px solid #e9ecef;
}

.cancel-button, .submit-button {
  padding: 8px 16px;
  border-radius: 6px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-button {
  background: transparent;
  border: 1px solid #e9ecef;
  color: #6c757d;
}

.submit-button {
  background: #4f46e5;
  border: none;
  color: white;
}

.cancel-button:hover {
  background: #f8f9fa;
}

.submit-button:hover {
  background: #4338ca;
}

.loading-spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: #fff;
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.submit-button:disabled,
.cancel-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

/* Dark theme styles */
body.dark-theme .modal-content {
  background: #1f2937;
}

body.dark-theme .modal-header {
  background: #111827;
  border-bottom: 1px solid #374151;
}

body.dark-theme .modal-header h3 {
  color: #f3f4f6;
}

body.dark-theme .close-button {
  color: #9ca3af;
}

body.dark-theme .modal-footer {
  border-top: 1px solid #374151;
}

body.dark-theme .cancel-button {
  border: 1px solid #374151;
  color: #9ca3af;
}

body.dark-theme .cancel-button:hover {
  background: #111827;
}

body.dark-theme .feedback-label {
  color: #d1d5db;
}

body.dark-theme .feedback-type-option {
  background: rgba(31, 41, 55, 0.7);
  border-color: #374151;
  color: #e5e7eb;
}

body.dark-theme .feedback-type-option:hover {
  background: rgba(31, 41, 55, 0.9);
  border-color: #4b5563;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

body.dark-theme .feedback-type-option.active {
  background: rgba(79, 70, 229, 0.2);
  border-color: rgba(79, 70, 229, 0.5);
  color: #a5b4fc;
  box-shadow: 0 2px 5px rgba(79, 70, 229, 0.15);
}

body.dark-theme .feedback-textarea {
  background-color: rgba(31, 41, 55, 0.7);
  border-color: #374151;
  color: #e5e7eb;
}

body.dark-theme .feedback-textarea::placeholder {
  color: #6b7280;
}

body.dark-theme .feedback-textarea:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
  background-color: #111827;
}

body.dark-theme .character-count {
  color: #9ca3af;
}

body.dark-theme .character-count.near-limit {
  color: #fbbf24;
  background-color: rgba(245, 158, 11, 0.15);
}

body.dark-theme .character-count.at-limit {
  color: #f87171;
  background-color: rgba(239, 68, 68, 0.15);
}

body.dark-theme .error-message {
  color: #f87171;
}
</style> 