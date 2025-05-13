<template>
  <div class="overlay" v-if="visible" @click="closeFeedback"></div>
  <div class="feedback-container" v-if="visible">
    <div class="feedback-box">
      <div class="feedback-header">
        <h2>反馈意见</h2>
      </div>
      
      <div class="feedback-content">
        <div v-if="isSubmitSuccess" class="submit-success-container">
          <div class="success-icon">
            <el-icon><Check /></el-icon>
          </div>
          <h3>感谢您的反馈！</h3>
          <p>您的反馈对我们非常重要，我们将持续优化AI助手的回答质量</p>
        </div>
        <div v-else class="feedback-layout">
          <div class="feedback-column content-column">
            <h3 class="section-title">天工AI智能助手的回答：</h3>
            <div class="ai-message">
              <div class="markdown-content" v-html="renderedMessage"></div>
            </div>
          </div>
          
          <div class="feedback-column options-column">
            <div class="feedback-section">
              <h3 class="section-title">反馈评分</h3>
              <div class="star-rating">
                <el-rate v-model="rating" :texts="['oops', 'disappointed', 'normal', 'good', 'great']" class="custom-rate" />
              </div>
            </div>
            
            <div class="feedback-section">
              <h3 class="section-title">请选择回答的不足之处（可选）：</h3>
              <div class="radio-group">
                <div class="radio-item">
                  <input 
                    type="radio" 
                    id="option-0" 
                    value="内容不准确" 
                    v-model="selectedFeedbackOption"
                    name="feedbackOption"
                  >
                  <label for="option-0">内容不准确</label>
                </div>
                <div class="radio-item">
                  <input 
                    type="radio" 
                    id="option-1" 
                    value="回答不完整" 
                    v-model="selectedFeedbackOption"
                    name="feedbackOption"
                  >
                  <label for="option-1">回答不完整</label>
                </div>
                <div class="radio-item">
                  <input 
                    type="radio" 
                    id="option-2" 
                    value="与问题不相关" 
                    v-model="selectedFeedbackOption"
                    name="feedbackOption"
                  >
                  <label for="option-2">与问题不相关</label>
                </div>
                <div class="radio-item">
                  <input 
                    type="radio" 
                    id="option-3" 
                    value="其他问题" 
                    v-model="selectedFeedbackOption"
                    name="feedbackOption"
                  >
                  <label for="option-3">其他问题</label>
                </div>
              </div>
            </div>
            
            <div class="feedback-section" style="flex: 1; display: flex; flex-direction: column; min-height: 0;">
              <h3 class="section-title">补充反馈：</h3>
              <div class="textarea-container">
                <textarea 
                  v-model="feedback" 
                  :placeholder="selectedFeedbackOption === '其他问题' ? '请详细描述您的反馈...' : '请输入您的补充反馈...'" 
                ></textarea>
              </div>
            </div>
            
            <div class="feedback-actions-inline">
              <button @click="submitFeedback">提交反馈</button>
              <button @click="closeFeedback">取消</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import { ElMessageBox, ElMessage } from "element-plus";
import { Check } from '@element-plus/icons-vue';
import { API_BASE_URL } from "../../config";
import axios from "axios";
import MarkdownIt from 'markdown-it';

const md = new MarkdownIt({
  html: true,
  breaks: true,
  linkify: true,
  typographer: true,
  highlight: function (str, lang) {
    return `<pre class="language-${lang}"><code>${str}</code></pre>`;
  }
});

const value = ref();
const props = defineProps({
  visible: Boolean,
  message: String,
  question: String,
});

const renderedMessage = computed(() => {
  if (!props.message) return '';
  return md.render(props.message);
});

const emit = defineEmits(["update:visible", "submitFeedback"]);

const feedback = ref("");
const rating = ref(0);
const isSubmitSuccess = ref(false);

const feedbackOptions = [
  { value: "内容不准确", label: "内容不准确" },
  { value: "回答不完整", label: "回答不完整" },
  { value: "与问题不相关", label: "与问题不相关" },
  { value: "其他问题", label: "其他问题" }
];

const selectedFeedbackOption = ref("");

const closeFeedback = () => {
  emit("update:visible", false);
  setTimeout(() => {
    isSubmitSuccess.value = false;
  }, 300);
};

const submitFeedback = async () => {
  if (rating.value) {
    try {
      const token = localStorage.getItem('token');
      
      // 创建要发送的数据对象
      const requestData = {
        rating: rating.value,
        feedback: feedback.value || " ", // 确保feedback始终有值
        feedback_option: selectedFeedbackOption.value || "其他问题", // 确保feedback_option始终有值
        message: props.message || " ", // 确保message始终有值
        question: props.question || " ", // 确保question始终有值
      };
      
      const response = await fetch(`${API_BASE_URL}/submit-feedback/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify(requestData),
      });

      const responseText = await response.text();
      
      // 尝试解析JSON
      let result;
      try {
        result = JSON.parse(responseText);
      } catch (e) {
        console.error('Failed to parse response as JSON:', e);
        ElMessage.error('服务器返回了无效的响应');
        return;
      }

      if (response.ok) {
        isSubmitSuccess.value = true;
        
        ElMessage({
          message: "感谢您的反馈！我们将持续优化AI助手的回答质量",
          type: "success",
          duration: 3000,
          showClose: true,
          offset: 80,
        });
        
        feedback.value = "";
        selectedFeedbackOption.value = "";
        rating.value = 0;
        
        setTimeout(() => {
          closeFeedback();
        }, 2000);
      } else {
        ElMessage.error(result.error || result.detail || "提交失败");
      }
    } catch (error) {
      ElMessage.error("提交失败，请稍后再试");
      console.error('Error submitting feedback:', error);
    }
  } else {
    ElMessage.error("请至少给出评分！");
  }
};
</script>

<style scoped>
.overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8));
  backdrop-filter: blur(6px);
  z-index: 1000;
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.feedback-container {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  width: 100%;
  height: 100vh;
  z-index: 1001;
  display: flex;
  justify-content: center;
  align-items: center;
  animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes fadeIn {
  from { opacity: 0; transform: scale(0.98); }
  to { opacity: 1; transform: scale(1); }
}

.feedback-box {
  width: 90%;
  max-width: 1200px;
  height: 90vh;
  background: linear-gradient(145deg, #ffffff, #f9fafb);
  border-radius: 24px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15), 0 1px 3px rgba(0, 0, 0, 0.05);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  border: 1px solid rgba(255, 255, 255, 0.8);
}

.feedback-header {
  padding: 22px 32px;
  background: linear-gradient(to right, #f8fafc, #f1f5f9);
  border-bottom: 1px solid rgba(226, 232, 240, 0.8);
}

.feedback-box h2 {
  font-size: 24px;
  font-weight: 700;
  color: #1a1a1a;
  margin: 0;
  text-align: center;
  position: relative;
  letter-spacing: 0.5px;
}

.feedback-box h2::after {
  content: '';
  position: absolute;
  bottom: -8px;
  left: 50%;
  transform: translateX(-50%);
  width: 40px;
  height: 3px;
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
  border-radius: 3px;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
}

.feedback-content {
  flex: 1;
  overflow: hidden;
  padding: 24px 32px;
}

.feedback-layout {
  display: flex;
  gap: 28px;
  height: 100%;
}

.content-column {
  flex: 0.9;
  overflow-y: auto;
  padding-right: 16px;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e1 transparent;
}

.options-column {
  flex: 1.1;
  background: linear-gradient(145deg, #f8fafc, #f1f5f9);
  border-radius: 18px;
  padding: 22px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  height: 100%;
  border: 1px solid rgba(226, 232, 240, 0.8);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}

.section-title {
  font-size: 14px;
  font-weight: 600;
  color: #3b82f6;
  margin: 0 0 10px 0;
  letter-spacing: 0.3px;
  display: flex;
  align-items: center;
}

.section-title::before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 14px;
  background: linear-gradient(to bottom, #3b82f6, #60a5fa);
  margin-right: 8px;
  border-radius: 2px;
}

.ai-message {
  background: linear-gradient(145deg, #f8fafc, #ffffff);
  border-radius: 14px;
  padding: 20px;
  border-left: 4px solid #3b82f6;
  flex: 1;
  overflow-y: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
  transition: all 0.3s ease;
  border: 1px solid rgba(226, 232, 240, 0.8);
  text-align: left;
}

.ai-message:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
  transform: translateY(-1px);
}

.star-rating {
  display: flex;
  justify-content: center;
  padding: 14px;
  background: linear-gradient(145deg, #ffffff, #f9fafb);
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(226, 232, 240, 0.8);
  transition: all 0.3s ease;
}

.star-rating:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
  transform: translateY(-1px);
}

.custom-rate {
  --el-rate-void-color: #cbd5e1;
  --el-rate-fill-color: #f59e0b;
  font-size: 26px;
}

.radio-group {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  background: linear-gradient(145deg, #ffffff, #f9fafb);
  padding: 14px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(226, 232, 240, 0.8);
  transition: all 0.3s ease;
}

.radio-group:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
  transform: translateY(-1px);
}

.radio-item {
  display: flex;
  align-items: center;
  cursor: pointer;
  padding: 8px 12px;
  border-radius: 8px;
  transition: all 0.2s ease;
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid transparent;
}

.radio-item:hover {
  background-color: rgba(241, 245, 249, 0.8);
  border-color: #e2e8f0;
}

.radio-item input[type="radio"] {
  appearance: none;
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border: 2px solid #cbd5e1;
  border-radius: 50%;
  margin-right: 10px;
  position: relative;
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;
  background: white;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.radio-item input[type="radio"]:checked {
  border-color: #3b82f6;
  background: #3b82f6;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}

.radio-item input[type="radio"]:checked::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 6px;
  height: 6px;
  background: white;
  border-radius: 50%;
}

.radio-item label {
  font-size: 14px;
  color: #4b5563;
  cursor: pointer;
  font-weight: 500;
}

.textarea-container {
  width: 100%;
  display: flex;
  flex: 1;
  min-height: 0;
  transition: all 0.3s ease;
}

textarea {
  width: 100%;
  padding: 14px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  background: linear-gradient(145deg, #ffffff, #f9fafb);
  font-size: 14px;
  line-height: 1.5;
  color: #1e293b;
  resize: none;
  transition: all 0.3s ease;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  box-sizing: border-box;
  flex: 1;
  min-height: 0;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}

textarea:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2), 0 4px 12px rgba(0, 0, 0, 0.03);
  background: #ffffff;
}

textarea::placeholder {
  color: #94a3b8;
  font-style: italic;
}

.feedback-actions-inline {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 18px;
}

button {
  padding: 10px 20px;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  border: none;
  outline: none;
  letter-spacing: 0.3px;
}

button:first-child {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: white;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
  position: relative;
  overflow: hidden;
}

button:first-child::after {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: all 0.6s ease;
}

button:first-child:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
}

button:first-child:hover::after {
  left: 100%;
}

button:first-child:active {
  transform: translateY(0);
}

button:last-child {
  background: linear-gradient(145deg, #f8fafc, #f1f5f9);
  color: #475569;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

button:last-child:hover {
  background: #e2e8f0;
  color: #334155;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
}

.content-column::-webkit-scrollbar,
.ai-message::-webkit-scrollbar {
  width: 6px;
}

.content-column::-webkit-scrollbar-track,
.ai-message::-webkit-scrollbar-track {
  background: transparent;
}

.content-column::-webkit-scrollbar-thumb,
.ai-message::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 6px;
}

.content-column::-webkit-scrollbar-thumb:hover,
.ai-message::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

@media (max-width: 900px) {
  .feedback-box {
    width: 95%;
    height: 95vh;
  }
  
  .feedback-header, .feedback-content {
    padding: 16px;
  }
  
  .feedback-layout {
    flex-direction: column;
  }
  
  .content-column, .options-column {
    flex: none;
    width: 100%;
    padding: 0;
    overflow-y: visible;
  }
  
  .options-column {
    padding: 16px;
    margin-top: 16px;
  }
  
  .radio-group {
    grid-template-columns: 1fr 1fr;
  }
  
  @media (max-width: 480px) {
    .radio-group {
      grid-template-columns: 1fr;
    }
    
    .radio-item:last-child {
      grid-column: auto;
    }
  }
  
  textarea {
    min-height: 120px;
  }
  
  .feedback-actions-inline {
    flex-direction: column;
    margin-top: 16px;
  }
  
  button {
    width: 100%;
  }
}

@media (min-width: 1200px) {
  .radio-group {
    grid-template-columns: repeat(3, 1fr);
  }
}

.markdown-content {
  font-size: 15px;
  line-height: 1.6;
  color: #4b5563;
  text-align: left;
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3,
.markdown-content h4,
.markdown-content h5,
.markdown-content h6 {
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  font-weight: 600;
  line-height: 1.25;
  color: #1a202c;
  text-align: left;
}

.markdown-content h1 {
  font-size: 1.8em;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 0.3em;
}

.markdown-content h2 {
  font-size: 1.5em;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 0.3em;
}

.markdown-content h3 {
  font-size: 1.25em;
}

.markdown-content h4 {
  font-size: 1em;
}

.markdown-content p,
.markdown-content ul,
.markdown-content ol {
  margin-top: 0;
  margin-bottom: 1em;
  text-align: left;
}

.markdown-content ul,
.markdown-content ol {
  padding-left: 2em;
}

.markdown-content li {
  margin-bottom: 0.25em;
}

.markdown-content ul li {
  list-style-type: disc;
}

.markdown-content ol li {
  list-style-type: decimal;
}

.markdown-content code {
  font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 0.9em;
  background-color: rgba(226, 232, 240, 0.5);
  padding: 0.2em 0.4em;
  border-radius: 3px;
}

.markdown-content pre {
  background-color: #f8fafc;
  border-radius: 6px;
  padding: 1em;
  overflow: auto;
  margin: 1em 0;
  border: 1px solid #e2e8f0;
}

.markdown-content pre code {
  background-color: transparent;
  padding: 0;
  font-size: 0.9em;
  color: #1a202c;
  white-space: pre;
}

.markdown-content table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0;
  text-align: left;
}

.markdown-content table th,
.markdown-content table td {
  border: 1px solid #e2e8f0;
  padding: 0.5em 1em;
  text-align: left;
}

.markdown-content table th {
  background-color: #f8fafc;
  font-weight: 600;
}

.markdown-content table tr:nth-child(even) {
  background-color: #f8fafc;
}

.markdown-content blockquote {
  margin: 1em 0;
  padding: 0.5em 1em;
  border-left: 4px solid #cbd5e1;
  background-color: #f8fafc;
  color: #4a5568;
}

.markdown-content a {
  color: #3b82f6;
  text-decoration: none;
  transition: color 0.2s ease;
}

.markdown-content a:hover {
  color: #2563eb;
  text-decoration: underline;
}

.markdown-content img {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  margin: 1em 0;
}

.markdown-content hr {
  height: 1px;
  background-color: #e2e8f0;
  border: none;
  margin: 2em 0;
}

.submit-success-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
  text-align: center;
  padding: 2rem;
}

.success-icon {
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, #4ade80, #22c55e);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.5rem;
  box-shadow: 0 10px 25px rgba(34, 197, 94, 0.3);
  animation: scaleIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s both;
}

.success-icon i,
.success-icon .el-icon {
  font-size: 40px;
  color: white;
}

.submit-success-container h3 {
  font-size: 24px;
  font-weight: 700;
  color: #16a34a;
  margin-bottom: 0.75rem;
  animation: fadeInUp 0.5s ease 0.4s both;
}

.submit-success-container p {
  font-size: 16px;
  color: #4b5563;
  max-width: 450px;
  line-height: 1.6;
  animation: fadeInUp 0.5s ease 0.6s both;
}

@keyframes scaleIn {
  from { transform: scale(0); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}

@keyframes fadeInUp {
  from { 
    opacity: 0;
    transform: translateY(20px);
  }
  to { 
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
