<template>
  <div class="feature-card ai-feature ai-feature-card">
    <div class="feature-header">
      <h2 class="fade-in">化工安全智能问答</h2>
    </div>
    
    <div class="ai-capabilities fade-in-delay">
      <div class="ai-capability">
        <span class="capability-text">专业知识查询</span>
      </div>
      <div class="ai-capability">
        <span class="capability-text">实时风险评估</span>
      </div>
      <div class="ai-capability">
        <span class="capability-text">安全标准对照</span>
      </div>
    </div>
    
    <div class="example-box fade-in-delay-2">
      <div class="example-header">示例问题</div>
      <ul class="example-list">
        <li>环氧乙烷储罐泄漏时的应急处置流程是什么？</li>
        <li>苯酚生产过程中如何控制硫酸雾的排放？</li>
        <li>高温高压釜的安全联锁系统应如何设计？</li>
        <li>双氧水与有机物混合可能产生哪些危险？</li>
      </ul>
    </div>
    
    <div class="button-container fade-in-delay-3">
      <router-link to="/chat" class="action-button ai-button">
        开始咨询
      </router-link>
    </div>
  </div>
</template>

<script setup>
// No props or state needed for this component
</script>

<style scoped>
.feature-card {
  background: white;
  border-radius: 12px;
  padding: 20px 20px 25px;
  display: flex;
  flex-direction: column;
  position: relative;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  overflow: hidden;
  text-align: center;
  box-sizing: border-box;
}

.ai-feature {
  grid-column: 1;
  grid-row: 1;
  height: auto;
  min-height: 400px;
  max-height: none;
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
  border: 1px solid rgba(233, 236, 239, 0.7);
  position: relative;
  padding-bottom: 70px;
}

.ai-feature:hover {
  box-shadow: 0 6px 24px rgba(79, 70, 229, 0.15);
  transform: translateY(-2px);
}

/* 添加背景动效 */
.ai-feature::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    radial-gradient(circle at 90% 10%, rgba(79, 70, 229, 0.05) 0%, transparent 30%),
    radial-gradient(circle at 10% 90%, rgba(99, 102, 241, 0.03) 0%, transparent 40%);
  z-index: 0;
  opacity: 0.8;
  transition: opacity 0.3s ease;
}

.ai-feature:hover::before {
  opacity: 1;
}

.feature-header {
  margin-bottom: 15px;
  text-align: center;
  position: relative;
  z-index: 1;
}

.feature-card h2 {
  margin: 0;
  font-size: 1.5rem;
  color: #1a1f36;
  text-align: center;
  font-weight: 600;
  background: linear-gradient(135deg, #4f46e5, #6366f1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  position: relative;
}

/* AI能力样式 */
.ai-capabilities {
  display: flex;
  justify-content: space-between;
  border-radius: 12px;
  padding: 16px;
  position: relative;
  z-index: 1;
  transition: all 0.3s ease;
  border: 1px solid rgba(79, 70, 229, 0.15);
  box-shadow: 0 2px 10px rgba(79, 70, 229, 0.05);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(247, 248, 255, 0.9));
}

.ai-capabilities:hover {
  transform: translateY(-1px);
  border-color: rgba(79, 70, 229, 0.25);
  box-shadow: 0 4px 15px rgba(79, 70, 229, 0.1);
}

.ai-capability {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  width: 30%;
  position: relative;
  transition: transform 0.3s ease;
}

.ai-capability:hover {
  transform: translateY(-2px);
}

.capability-text {
  font-size: 0.9rem;
  font-weight: 600;
  color: #4f46e5;
  text-align: center;
}

/* 示例问题样式 */
.example-box {
  border-radius: 12px;
  padding: 16px;
  margin: 15px 0;
  flex: 0 1 auto;
  max-height: 200px;
  text-align: center;
  position: relative;
  z-index: 1;
  transition: all 0.3s ease;
  border: 1px solid rgba(79, 70, 229, 0.15);
  box-shadow: 0 2px 10px rgba(79, 70, 229, 0.05);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(247, 248, 255, 0.9));
  display: flex;
  flex-direction: column;
}

.example-box:hover {
  border-color: rgba(79, 70, 229, 0.25);
  box-shadow: 0 4px 15px rgba(79, 70, 229, 0.1);
}

.example-header {
  font-size: 1rem;
  font-weight: 600;
  color: #1a1f36;
  margin-bottom: 16px;
  text-align: center;
}

.example-list {
  margin: 0;
  padding-left: 15px;
  font-size: 0.9rem;
  color: #4a5568;
  text-align: left;
  list-style-position: outside;
}

.example-list li {
  margin-bottom: 8px;
  line-height: 1.5;
  text-align: left;
  padding: 8px;
  border-radius: 8px;
  transition: all 0.2s ease;
  position: relative;
  border-left: 2px solid transparent;
}

.example-list li:hover {
  background: linear-gradient(90deg, rgba(79, 70, 229, 0.03), transparent);
  transform: scale(1.02);
  cursor: pointer;
  border-left: 2px solid #4f46e5;
  padding-left: 12px;
}

/* 按钮容器 */
.button-container {
  margin-top: auto;
  padding-top: 15px;
  padding-bottom: 12px;
  display: flex;
  justify-content: center;
  width: 100%;
  position: absolute;
  bottom: 10px;
  left: 0;
  z-index: 2;
}

/* 按钮样式 */
.action-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #4f46e5;
  color: white;
  border: none;
  border-radius: 10px;
  padding: 10px 15px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.3s ease;
  width: 70%;
  text-align: center;
  position: relative;
  overflow: hidden;
  max-height: 42px;
}

.action-button::before {
  content: "";
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0) 0%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0) 100%
  );
  transition: all 0.6s ease;
}

.action-button:hover::before {
  left: 100%;
}

.action-button:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3);
}

.ai-button {
  background: linear-gradient(135deg, #4f46e5, #6366f1);
}

/* Dark theme styles */
body.dark-theme .feature-card {
  background: rgba(31, 41, 55, 0.8);
}

body.dark-theme .ai-feature {
  background: rgba(31, 41, 55, 0.8);
  border-color: rgba(55, 65, 81, 0.5);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

body.dark-theme .ai-feature::before {
  background: 
    radial-gradient(circle at 90% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 30%),
    radial-gradient(circle at 10% 90%, rgba(79, 70, 229, 0.06) 0%, transparent 40%);
}

body.dark-theme .feature-card h2 {
  background: linear-gradient(135deg, #a5b4fc, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

body.dark-theme .example-header, 
body.dark-theme .feature-card h2 {
  color: #f3f4f6;
}

body.dark-theme .example-list li {
  color: #d1d5db;
}

body.dark-theme .example-list li:hover {
  background: linear-gradient(90deg, rgba(79, 70, 229, 0.08), transparent);
  border-left: 2px solid #818cf8;
}

body.dark-theme .example-box {
  background: linear-gradient(135deg, rgba(31, 41, 55, 0.7), rgba(17, 24, 39, 0.7));
  border-color: rgba(79, 70, 229, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

body.dark-theme .example-box:hover {
  border-color: rgba(99, 102, 241, 0.3);
  box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
}

body.dark-theme .ai-capabilities {
  background: linear-gradient(135deg, rgba(31, 41, 55, 0.7), rgba(17, 24, 39, 0.7));
  border-color: rgba(79, 70, 229, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

body.dark-theme .ai-capabilities:hover {
  border-color: rgba(99, 102, 241, 0.3);
  box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
}

body.dark-theme .capability-text {
  color: #a5b4fc;
}

/* Responsive design */
@media (max-width: 900px) {
  .ai-feature {
    max-height: none;
    padding-bottom: 25px;
  }
  
  .action-button {
    width: 90%;
    padding: 10px;
  }
  
  .feature-card {
    padding: 15px 15px 20px;
  }
}

/* 添加渐入动画效果 */
.fade-in {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
}

.fade-in-delay {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
  animation-delay: 0.3s;
}

.fade-in-delay-2 {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
  animation-delay: 0.6s;
}

.fade-in-delay-3 {
  opacity: 0;
  animation: fadeIn 0.8s ease forwards;
  animation-delay: 0.9s;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style> 