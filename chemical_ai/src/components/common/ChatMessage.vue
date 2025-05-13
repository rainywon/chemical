<template>
  <div class="message-container" :class="messageType">
    <template v-if="messageType === 'ai'">
      <div class="ai-message-content">
        <img src="@/assets/product_logo.png" alt="AI Image" class="ai-image" />
        <!-- 显示文档参考的小方框 -->
        <template v-if="(Array.isArray(references) && references.length > 0) || (!Array.isArray(references) && Object.keys(references).length > 0)">
          <div class="references-box">
            <References :references="Array.isArray(references) ? references : Object.values(references)" />
          </div>
        </template>
        <div class="chat-message ai">
          <!-- 使用 v-html 渲染 Markdown 后的 HTML 内容 -->
          <div class="message-content" ref="messageContentRef" v-html="parsedMessage" @click="handleThinkBlockClick"></div>
        </div>
        <!-- 加载指示器 -->
        <div v-if="isLoading" class="loading-indicator">
          <div class="typing-animation">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <div class="loading-text">天工AI正在思考中...</div>
        </div>
      </div>
    </template>

    <template v-else>
      <div class="chat-message user">
        <div class="message-content" v-html="parsedMessage"></div>
      </div>
    </template>
  </div>

  <div class="message-actions" :style="actionStyle">
    <!-- 用户消息操作 -->
    <template v-if="messageType === 'user'">
      <el-tooltip content="复制" effect="dark">
        <img src="@/assets/copy.png" class="action-icon" @click="copyMessage" />
      </el-tooltip>
    </template>

    <!-- AI 消息操作 -->
    <template v-if="messageType === 'ai' && !isLoading">
      <el-tooltip content="复制" effect="dark">
        <img src="@/assets/copy.png" class="action-icon" @click="copyMessage" />
      </el-tooltip>
      <el-tooltip content="反馈" effect="dark">
        <img src="@/assets/feedback.png" class="action-icon" @click="openFeedbackDialog" />
      </el-tooltip>
      <el-tooltip v-if="isLast" content="重新生成" effect="dark">
        <img src="@/assets/regenerate.png" class="action-icon" @click="emit('regenerate', messageId)" />
      </el-tooltip>
    </template>
  </div>

  <!-- Feedback Component -->
  <Teleport to="body">
    <FeedbackForm :visible="dialogVisible" :message="message" @update:visible="dialogVisible = $event"
      @submitFeedback="handleFeedbackSubmit" :question="question" />
  </Teleport>
</template>

<script setup>
import { computed, onMounted, ref, nextTick, watch, onUpdated, shallowRef, markRaw, onBeforeUnmount } from "vue";
import { ElMessage } from "element-plus";
import { marked } from "marked"; // 用于 Markdown 渲染
import DOMPurify from "dompurify";
import References from "./References.vue";
import FeedbackForm from "./FeedbackForm.vue";
import hljs from "highlight.js";
import Prism from "prismjs"; //导入代码高亮插件的core（里面提供了其他官方插件及代码高亮样式主题，你只需要引入即可）
import "prismjs/themes/prism-tomorrow.min.css";
// 导入额外的Prism插件和组件
import "prismjs/plugins/line-numbers/prism-line-numbers.min.css";
import "prismjs/plugins/line-numbers/prism-line-numbers.min.js";
import "prismjs/plugins/toolbar/prism-toolbar.min.css";
import "prismjs/plugins/toolbar/prism-toolbar.min.js";
import "prismjs/plugins/copy-to-clipboard/prism-copy-to-clipboard.min.js";
import "prismjs/plugins/show-language/prism-show-language.min.js";
// 导入KaTeX数学公式渲染支持
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { renderMathInElement } from 'katex/dist/contrib/auto-render.min.js';

// 自定义Prism插件文本
if (window.Prism && window.Prism.plugins && window.Prism.plugins.toolbar) {
  // 检查按钮是否已经存在
  if (!window.Prism.plugins.toolbar.buttons || !window.Prism.plugins.toolbar.buttons['copy-to-clipboard']) {
    try {
      window.Prism.plugins.toolbar.registerButton('copy-to-clipboard', function(env) {
        const linkCopy = document.createElement('button');
        linkCopy.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>复制</span>';
        linkCopy.className = 'copy-to-clipboard-button';
        
        registerClipboard(linkCopy, env.element);
        return linkCopy;
        
        function registerClipboard(copyButton, element) {
          copyButton.addEventListener('click', function() {
            const textToCopy = element.textContent;
            
            navigator.clipboard.writeText(textToCopy).then(function() {
              copyButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg><span>已复制!</span>';
              
              setTimeout(function () {
                copyButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>复制</span>';
              }, 2000);
            }, function() {
              copyButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg><span>复制失败!</span>';
            });
          });
        }
      });
      console.log('成功注册复制按钮');
    } catch (error) {
      console.warn('复制按钮注册失败:', error);
    }
  } else {
    console.log('复制按钮已存在，跳过注册');
  }
  
  // 检查按钮是否已经存在
  if (!window.Prism.plugins.toolbar.buttons || !window.Prism.plugins.toolbar.buttons['show-language']) {
    try {
      window.Prism.plugins.toolbar.registerButton('show-language', function(env) {
        const languageMap = {
          'javascript': '语言: JavaScript',
          'js': '语言: JavaScript',
          'typescript': '语言: TypeScript',
          'ts': '语言: TypeScript',
          'jsx': '语言: React JSX',
          'tsx': '语言: React TSX',
          'html': '语言: HTML',
          'xml': '语言: XML',
          'svg': '语言: SVG',
          'css': '语言: CSS',
          'scss': '语言: SCSS',
          'less': '语言: Less',
          'python': '语言: Python',
          'py': '语言: Python',
          'java': '语言: Java',
          'c': '语言: C',
          'cpp': '语言: C++',
          'csharp': '语言: C#',
          'cs': '语言: C#',
          'php': '语言: PHP',
          'ruby': '语言: Ruby',
          'rb': '语言: Ruby',
          'go': '语言: Go',
          'rust': '语言: Rust',
          'swift': '语言: Swift',
          'kotlin': '语言: Kotlin',
          'dart': '语言: Dart',
          'sql': '语言: SQL',
          'bash': '语言: Bash',
          'sh': '语言: Shell',
          'json': '语言: JSON',
          'yaml': '语言: YAML',
          'yml': '语言: YAML',
          'markdown': '语言: Markdown',
          'md': '语言: Markdown',
          'plaintext': '语言: 纯文本',
          'txt': '语言: 纯文本'
        };
        
        const pre = env.element.parentNode;
        if (!pre || !/pre/i.test(pre.nodeName)) {
          return;
        }
        
        let language = pre.getAttribute('data-language') || 
                      Array.from(env.element.classList)
                        .find(cls => cls.startsWith('language-'))?.replace('language-', '');
                        
        if (!language) {
          return;
        }
        
        language = language.toLowerCase();
        
        const element = document.createElement('span');
        element.className = 'show-language';
        element.textContent = languageMap[language] || `语言: ${language.toUpperCase()}`;
        
        return element;
      });
      console.log('成功注册显示语言按钮');
    } catch (error) {
      console.warn('显示语言按钮注册失败:', error);
    }
  } else {
    console.log('显示语言按钮已存在，跳过注册');
  }
}

// 使用markRaw避免Vue对这些大型对象进行响应式处理
const markedInstance = markRaw(marked);
const prismInstance = markRaw(Prism);
const domPurify = markRaw(DOMPurify);
const katexInstance = markRaw(katex);

const emit = defineEmits(["regenerate"]);

const props = defineProps({
  message: String,
  messageType: String,
  isLoading: Boolean,
  messageId: [Number, String],
  references: [Array, Object],
  question: String,
  isLast: Boolean,
});

// 使用shallowRef来存储thinkBlocksState，避免深层响应式
const thinkBlocksState = shallowRef(new Map());
// 缓存解析后的消息，避免频繁重新解析
const cachedParsedMessage = shallowRef('');
// 标记是否需要重新解析消息
const needsReparse = ref(true);
// 消息容器引用，用于后续处理数学公式
const messageContentRef = ref(null);

// 配置DOMPurify以允许KaTeX生成的标签和属性
DOMPurify.addHook('afterSanitizeAttributes', function(node) {
  if (node.hasAttribute('data-latex') || 
      node.classList && (node.classList.contains('katex') || 
                        node.classList.contains('math-inline') || 
                        node.classList.contains('math-block'))) {
    node.setAttribute('data-latex-safe', 'true');
  }
});

// 使用防抖函数优化事件处理
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

// 处理 think 块的点击事件
const handleThinkBlockClick = (event) => {
  // 寻找最近的think-header或think-toggle-btn元素
  const header = event.target.closest('.think-header');
  const toggleBtn = event.target.closest('.think-toggle-btn');
  
  if (header || toggleBtn) {
    const container = event.target.closest('.think-container');
    if (container) {
      const thinkId = container.dataset.thinkId;
      toggleThinkBlock(thinkId);
    }
  }
};

// 切换 think 块的展开/收起状态 - 优化版本
const toggleThinkBlock = (thinkId) => {
  // 创建新的Map而不是修改原有Map，以触发响应式更新
  const newState = new Map(thinkBlocksState.value);
  const currentState = newState.get(thinkId);
  const newExpandState = !currentState;
  
  // 设置新状态
  newState.set(thinkId, newExpandState);
  thinkBlocksState.value = newState;
  
  // 使用setTimeout确保DOM已更新
  setTimeout(() => {
    const thinkContent = document.querySelector(`.think-content[data-think-id="${thinkId}"]`);
    const thinkContainer = document.querySelector(`.think-container[data-think-id="${thinkId}"]`);
    const toggleBtn = document.querySelector(`.think-toggle-btn[data-think-id="${thinkId}"]`);
    
    if (!thinkContent || !thinkContainer) {
      console.error('思考过程框元素未找到:', thinkId);
      return;
    }
    
    try {
      if (newExpandState) {
        // 展开状态
        thinkContainer.classList.add('expanded');
        
        // 计算内容实际高度
        thinkContent.style.maxHeight = 'none';
        thinkContent.style.visibility = 'hidden';
        thinkContent.style.position = 'absolute';
        thinkContent.style.padding = '16px';
        
        const contentHeight = thinkContent.scrollHeight;
        
        thinkContent.style.visibility = '';
        thinkContent.style.position = '';
        thinkContent.style.padding = '';
        thinkContent.style.maxHeight = '0';
        
        // 使用requestAnimationFrame来确保在下一帧进行动画
        requestAnimationFrame(() => {
          thinkContent.style.maxHeight = `${contentHeight}px`;
          
          // 动画完成后，设置maxHeight为none，以允许内容自适应变化
          setTimeout(() => {
            thinkContent.style.maxHeight = 'none';
          }, 500); // 与CSS过渡时间匹配
        });
        
        if (toggleBtn) {
          toggleBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg>';
        }
      } else {
        // 收起状态
        // 先设置确切的高度值，再过渡到0
        const contentHeight = thinkContent.scrollHeight;
        thinkContent.style.maxHeight = `${contentHeight}px`;
        
        // 先进行一次重绘，确保浏览器记录了当前高度
        thinkContent.offsetHeight; 
        
        // 设置收起动画的目标高度
        requestAnimationFrame(() => {
          thinkContainer.classList.remove('expanded');
          thinkContent.style.maxHeight = '0';
        });
        
        if (toggleBtn) {
          toggleBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>';
        }
      }
    } catch (error) {
      console.error('切换思考过程框状态失败:', error);
    }
  }, 0);
};

// 设置Marked选项 - 优化代码高亮性能
markedInstance.setOptions({
  highlight: function (code, lang) {
    if (lang && prismInstance.languages[lang]) {
      try {
        return prismInstance.highlight(code, prismInstance.languages[lang], lang);
      } catch (err) {
        console.error('Prism highlighting error:', err);
        return code;
      }
    }
    return code;
  },
  gfm: true,
  breaks: true,
  silent: true // 避免在解析错误时抛出异常
});

// 修改计算属性，使用缓存提高性能
const parsedMessage = computed(() => {
  // 如果消息未更改且已有缓存，直接返回缓存的结果
  if (!needsReparse.value && cachedParsedMessage.value) {
    return cachedParsedMessage.value;
  }
  
  // 空消息直接返回空字符串
  if (!props.message) {
    cachedParsedMessage.value = '';
    needsReparse.value = false;
    return '';
  }
  
  try {
    // 处理Markdown内容前先保护数学公式
    let processedMessage = props.message || "";
    
    // 1. 保存所有数学公式到安全区域
    const mathExpressions = [];
    
    // 处理块级公式
    processedMessage = processedMessage.replace(/\$\$([\s\S]+?)\$\$/g, function(match, formula) {
      const id = mathExpressions.length;
      mathExpressions.push({type: 'block', formula: formula});
      return `@@MATH_BLOCK_${id}@@`;
    });
    
    // 处理行内公式
    processedMessage = processedMessage.replace(/\$([^$\n]+?)\$/g, function(match, formula) {
      const id = mathExpressions.length;
      mathExpressions.push({type: 'inline', formula: formula});
      return `@@MATH_INLINE_${id}@@`;
    });
    
    // 2. 使用Marked处理Markdown
    let rawHtml = markedInstance(processedMessage);
    
    // 3. 将公式占位符替换回KaTeX渲染后的HTML
    mathExpressions.forEach((item, index) => {
      let renderedFormula = '';
      try {
        if (item.type === 'block') {
          renderedFormula = katexInstance.renderToString(item.formula, {
            displayMode: true,
            throwOnError: false,
            output: 'html'
          });
          renderedFormula = `<div class="math-block" data-latex="true">${renderedFormula}</div>`;
        } else {
          renderedFormula = katexInstance.renderToString(item.formula, {
            displayMode: false,
            throwOnError: false,
            output: 'html'
          });
          renderedFormula = `<span class="math-inline" data-latex="true">${renderedFormula}</span>`;
        }
      } catch (err) {
        console.error('KaTeX rendering error:', err);
        renderedFormula = item.type === 'block' 
          ? `<div class="math-error">$$${item.formula}$$</div>` 
          : `<span class="math-error">$${item.formula}$</span>`;
      }
      
      const blockRegex = new RegExp(`@@MATH_BLOCK_${index}@@`, 'g');
      const inlineRegex = new RegExp(`@@MATH_INLINE_${index}@@`, 'g');
      
      if (item.type === 'block') {
        rawHtml = rawHtml.replace(blockRegex, renderedFormula);
      } else {
        rawHtml = rawHtml.replace(inlineRegex, renderedFormula);
      }
    });
    
    // 替换 <think> 标签，添加展开/收起功能
    let thinkBlockCounter = 0;
    rawHtml = rawHtml.replace(/<think>([\s\S]*?)(<\/think>|$)/g, (match, content, endTag) => {
      const thinkId = `think-${props.messageId}-${thinkBlockCounter++}`;
      
      // 初始化 think 块的状态（默认展开）
      if (!thinkBlocksState.value.has(thinkId)) {
        const newState = new Map(thinkBlocksState.value);
        // 默认展开所有思考块
        newState.set(thinkId, true);
        thinkBlocksState.value = newState;
      }
      
      // 检查是否有结束标签，如果没有，添加一个占位符
      const isComplete = endTag === '</think>';
      
      return `
        <div class="think-container expanded" data-think-id="${thinkId}">
          <div class="think-header" data-think-id="${thinkId}">
            <div class="think-title">思考过程</div>
            <div class="think-toggle-btn" data-think-id="${thinkId}">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg>
            </div>
          </div>
          <div class="think-content" data-think-id="${thinkId}" style="max-height: none;">
            ${content}
          </div>
        </div>
      `;
    });

    // 增强图片处理，添加可点击放大功能
    rawHtml = rawHtml.replace(/<img src="([^"]+)"([^>]*)>/g, (match, src, attrs) => {
      return `<div class="md-image-container">
        <img src="${src}" ${attrs} loading="lazy" />
        <div class="md-image-caption">点击查看大图</div>
      </div>`;
    });
    
    // 为代码块添加语言标签和更好的复制按钮
    rawHtml = rawHtml.replace(/<pre class="line-numbers" data-language="([^"]*)">([\s\S]*?)<\/pre>/g, (match, lang, code) => {
      // 获取友好的语言名称
      const langDisplay = getLangDisplayName(lang);
      return `<div class="code-block-container">
        <div class="code-block-header">
          <span class="code-lang-tag">${langDisplay}</span>
          <button class="code-copy-btn" onclick="(function(e){e.preventDefault();const code=this.parentElement.parentElement.querySelector('pre code').textContent;navigator.clipboard.writeText(code).then(()=>{this.innerHTML='<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'16\' height=\'16\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'><polyline points=\'20 6 9 17 4 12\'></polyline></svg><span>已复制!</span>';setTimeout(()=>{this.innerHTML='<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'16\' height=\'16\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'><rect x=\'9\' y=\'9\' width=\'13\' height=\'13\' rx=\'2\' ry=\'2\'></rect><path d=\'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1\'></path></svg><span>复制</span>';},2000);}).catch(()=>{this.innerHTML='<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'16\' height=\'16\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'><circle cx=\'12\' cy=\'12\' r=\'10\'></circle><line x1=\'12\' y1=\'8\' x2=\'12\' y2=\'12\'></line><line x1=\'12\' y1=\'16\' x2=\'12.01\' y2=\'16\'></line></svg><span>复制失败!</span>';});})(event)">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>复制</span>
          </button>
        </div>
        ${match}
      </div>`;
    });
    
    // 配置DOMPurify允许KaTeX相关标签
    const sanitizedHtml = domPurify.sanitize(rawHtml, {
      ADD_TAGS: ['math', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 'mfrac', 'mspace', 'mtable', 'mtr', 'mtd', 'annotation', 'semantics', 'svg', 'mstyle'],
      ADD_ATTR: ['display', 'encoding', 'data-latex', 'data-latex-safe', 'class', 'style', 'id', 'viewBox', 'width', 'height', 'xmlns', 'd', 'preserveAspectRatio']
    });
    
    // 更新缓存和状态
    cachedParsedMessage.value = sanitizedHtml;
    needsReparse.value = false;
    
    // 使用requestAnimationFrame优化DOM操作
    requestAnimationFrame(() => {
      handlePostRender();
    });
    
    return sanitizedHtml;
  } catch (error) {
    console.error("Error parsing message:", error);
    return props.message || ""; // 解析失败时返回原始消息
  }
});

// 获取语言的友好显示名称
const getLangDisplayName = (lang) => {
  const langMap = {
    'js': 'JavaScript',
    'jsx': 'React JSX',
    'ts': 'TypeScript',
    'tsx': 'React TSX',
    'html': 'HTML',
    'css': 'CSS',
    'scss': 'SCSS',
    'less': 'Less',
    'json': 'JSON',
    'py': 'Python',
    'python': 'Python',
    'java': 'Java',
    'c': 'C',
    'cpp': 'C++',
    'cs': 'C#',
    'go': 'Go',
    'rust': 'Rust',
    'rb': 'Ruby',
    'ruby': 'Ruby',
    'php': 'PHP',
    'sh': '命令行',
    'bash': '命令行',
    'sql': 'SQL',
    'swift': 'Swift',
    'kotlin': 'Kotlin',
    'dart': 'Dart',
    'vue': 'Vue',
    'xml': 'XML',
    'yaml': 'YAML',
    'yml': 'YAML',
    'md': 'Markdown',
    'markdown': 'Markdown',
    'docker': 'Dockerfile',
    'dockerfile': 'Dockerfile',
    'plaintext': '纯文本',
    'txt': '纯文本',
    'text': '纯文本'
  };
  
  return langMap[lang.toLowerCase()] || lang.toUpperCase() || '代码';
};

// DOM操作优化 - 将所有DOM操作合并到一个函数
const handlePostRender = debounce(() => {
  // 使用IntersectionObserver判断元素是否可见，只处理可见元素
  if ('IntersectionObserver' in window) {
    // 代码高亮处理
    const codeBlocks = document.querySelectorAll('.message-content pre code');
    if (codeBlocks.length > 0) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            try {
              // 激活Prism的代码高亮和插件功能
              prismInstance.highlightElement(entry.target);
              
              // 获取代码块wrapper
              const codeWrapper = entry.target.closest('.code-block-container');
              if (codeWrapper) {
                // 绑定复制按钮事件
                const copyBtn = codeWrapper.querySelector('.code-copy-btn');
                if (copyBtn && !copyBtn.hasAttribute('data-listener-attached')) {
                  copyBtn.setAttribute('data-listener-attached', 'true');
                  copyBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const code = entry.target.textContent;
                    navigator.clipboard.writeText(code).then(() => {
                      // 显示复制成功提示
                      copyBtn.classList.add('copied');
                      copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg><span class="copy-text">已复制!</span>';
                      
                      setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span class="copy-text">复制</span>';
                      }, 2000);
                    }).catch(err => {
                      console.error('复制失败:', err);
                    });
                  });
                }
              }
            } catch (e) {
              console.warn('代码高亮失败:', e);
            }
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.1 });
      
      codeBlocks.forEach(block => observer.observe(block));
    }
    
    // 处理think块 - 确保默认展开
    const thinkContents = document.querySelectorAll(".think-content");
    if (thinkContents.length > 0) {
      thinkContents.forEach((content) => {
        const thinkId = content.dataset.thinkId;
        const thinkContainer = content.closest('.think-container');
        
        // 确保思考块展开
        if (thinkContainer) {
          thinkContainer.classList.add('expanded');
          content.style.maxHeight = 'none';
        }
      });
    }

    // 处理图片点击放大
    const images = document.querySelectorAll('.message-content .md-image-container img');
    images.forEach(img => {
      if (!img.hasAttribute('data-zoom-initialized')) {
        img.setAttribute('data-zoom-initialized', 'true');
        img.addEventListener('click', () => {
          const overlay = document.createElement('div');
          overlay.className = 'image-zoom-overlay';
          
          const imgClone = document.createElement('img');
          imgClone.src = img.src;
          imgClone.className = 'zoomed-image';
          
          overlay.appendChild(imgClone);
          document.body.appendChild(overlay);
          
          overlay.addEventListener('click', () => {
            document.body.removeChild(overlay);
          });
        });
      }
    });

    // 确保数学公式正确渲染（备用方案）
    const messageContentElement = document.querySelector('.message-content');
    if (messageContentElement) {
      // 检查是否有未渲染的公式占位符
      const unrenderedBlocks = messageContentElement.innerHTML.match(/@@MATH_(BLOCK|INLINE)_\d+@@/g);
      if (unrenderedBlocks && unrenderedBlocks.length > 0) {
        console.warn('Found unrendered math blocks, applying auto-render fallback');
        try {
          renderMathInElement(messageContentElement, {
            delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$', right: '$', display: false}
            ],
            throwOnError: false
          });
        } catch (err) {
          console.error('Auto-render fallback failed:', err);
        }
      }
    }
  } else {
    // 降级处理 - 直接处理DOM
    document.querySelectorAll("pre code").forEach((block) => {
      try {
        prismInstance.highlightElement(block);
      } catch (e) {
        console.warn('代码高亮失败:', e);
      }
    });
    
    // 确保思考块展开
    document.querySelectorAll(".think-content").forEach((content) => {
      const thinkContainer = content.closest('.think-container');
      if (thinkContainer) {
        thinkContainer.classList.add('expanded');
        content.style.maxHeight = 'none';
      }
    });
  }
}, 50);

// 监听消息变化，触发重新解析
watch(() => props.message, (newMessage, oldMessage) => {
  // 检查是否有新的<think>标签开始或结束
  const hasNewThinkStart = newMessage && newMessage.includes('<think>') && 
                          (!oldMessage || !oldMessage.includes('<think>'));
  const hasNewThinkEnd = newMessage && newMessage.includes('</think>') && 
                        (!oldMessage || !oldMessage.includes('</think>'));
  
  // 如果有新的<think>标签开始或结束，或者消息内容变化，则重新解析
  if (hasNewThinkStart || hasNewThinkEnd || newMessage !== oldMessage) {
    needsReparse.value = true;
  }
}, { immediate: true });

// 组件挂载完成后处理DOM
onMounted(() => {
  handlePostRender();
});

// 组件更新后处理DOM，使用防抖减少频繁调用
const debouncedPostUpdate = debounce(() => {
  handlePostRender();
}, 100);

onUpdated(() => {
  debouncedPostUpdate();
});

// 组件卸载前清理
onBeforeUnmount(() => {
  // 清理任何可能的订阅、定时器等
});

const dialogVisible = ref(false);

const actionStyle = computed(() => ({
  display: "flex",
  justifyContent: props.messageType === "user" ? "flex-end" : "flex-start",
  gap: "12px",
}));

// 复制消息方法 - 添加错误处理
const copyMessage = async () => {
  let messageToCopy = props.message;
  if (props.messageType === "user") {
    messageToCopy = messageToCopy.trim();
  }
  
  try {
    await navigator.clipboard.writeText(messageToCopy);
    ElMessage.success("复制成功");
  } catch (error) {
    console.error("复制失败:", error);
    // 尝试降级处理
    try {
      const textarea = document.createElement('textarea');
      textarea.value = messageToCopy;
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      ElMessage.success("复制成功");
    } catch (fallbackError) {
      ElMessage.error("复制失败");
    }
  }
};

// 打开反馈对话框
const openFeedbackDialog = () => {
  dialogVisible.value = true;
};

// 处理用户反馈
const handleFeedbackSubmit = (feedback) => {
  ElMessage.success("感谢您的反馈！");
};
</script>

<style scoped>
/* 消息容器 */
.message-container {
  display: flex;
  align-items: flex-start;
  margin: 20px 0;
  flex-direction: column;
  contain: content; /* 添加包含属性提高渲染性能 */
  will-change: transform; /* 暗示浏览器元素将改变，优化渲染 */
}

/* AI 消息样式 */
.message-container.ai {
  width: 100%;
  display: flex;
  align-items: flex-start;
  text-align: left;
  transform: translateZ(0); /* 启用GPU加速 */
}

.message-container .ai-image {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  margin-right: 10px;
  margin-bottom: 10px;
  contain: strict; /* 严格包含，避免布局变化影响其他元素 */
  will-change: transform; /* 启用GPU加速 */
}

/* AI 对话框 */
.ai-message-content {
  position: relative;
  width: 100%;
  contain: layout; /* 包含布局变化 */
}

.chat-message.ai {
  width: 100%;
  border-radius: 12px;
  margin-left: 5px;
  text-align: left;
  box-sizing: border-box;
  transform: translateZ(0); /* 启用GPU加速 */
}

/* 用户消息 */
.message-container.user {
  flex-direction: row-reverse;
  transform: translateZ(0); /* 启用GPU加速 */
}

.chat-message.user {
  background-color: #f1f0f0;
  max-width: 70%;
  min-width: 220px;
  border-radius: 12px;
  padding: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-left: auto;
  text-align: left;
}

/* 操作按钮图标样式 */
.action-icon {
  width: 20px;
  height: 20px;
  cursor: pointer;
  transition: opacity 0.2s, background-color 0.2s;
  padding: 4px;
  transform: translateZ(0); /* 启用GPU加速 */
}

.action-icon:hover {
  opacity: 0.8;
  background-color: rgba(0, 0, 0, 0.1);
  border-radius: 35%;
}

/* 调整消息操作容器 */
.message-actions {
  margin-top: 8px;
  padding: 4px 0;
}

/* 加载指示器 */
.loading-indicator {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-top: 8px;
  margin-left: 20px;
}

.typing-animation {
  display: flex;
  align-items: center;
}

.typing-animation span {
  height: 8px;
  width: 8px;
  margin: 0 2px;
  background-color: #2563eb;
  border-radius: 50%;
  display: inline-block;
  animation: bounce 1.5s infinite ease-in-out;
}

.typing-animation span:nth-child(1) {
  animation-delay: 0s;
}

.typing-animation span:nth-child(2) {
  animation-delay: 0.3s;
}

.typing-animation span:nth-child(3) {
  animation-delay: 0.6s;
}

.loading-text {
  margin-top: 5px;
  color: #6b7280;
  font-size: 14px;
}

@keyframes bounce {
  0%, 80%, 100% { 
    transform: scale(0);
    opacity: 0.5;
  }
  40% { 
    transform: scale(1);
    opacity: 1;
  }
}

/* 参考框样式 */
.references-box {
  margin-bottom: 20px;
  contain: content; /* 包含内容，提高渲染性能 */
}

/* 保留其余样式，但使用CSS contain属性优化渲染性能 */
/* Markdown 内容样式 - 使用GPU加速和优化渲染属性 */
.message-content :deep(*) {
  /* font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; */
  line-height: 1.6;
  -webkit-font-smoothing: antialiased; /* 提高文本渲染 */
  text-rendering: optimizeLegibility; /* 优化字体渲染 */
}

/* 增强的代码块样式 */
.message-content :deep(pre) {
  position: relative;
  padding: 30px !important;
  border-radius: 15px;
  transform: translateZ(0); /* 启用GPU加速 */
  contain: content; /* 包含内容，提高渲染性能 */
  will-change: transform; /* 告知浏览器变化特性，优化渲染 */
  background-color: #282c34;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  margin: 24px 0;
  overflow: hidden;
  font-family: 'SF Mono', 'Fira Code', 'Menlo', 'Monaco', 'Courier New', monospace;
}

.message-content :deep(pre.line-numbers) {
  padding-left: 60px !important;
  counter-reset: linenumber;
}

.message-content :deep(pre .line-numbers-rows) {
  position: absolute;
  left: 0;
  width: 50px;
  text-align: center;
  background-color: rgba(0, 0, 0, 0.15);
  padding-top: 30px;
  padding-bottom: 30px;
}

.message-content :deep(.code-copy-button) {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 6px;
  color: rgba(255, 255, 255, 0.8);
  padding: 5px 10px;
  cursor: pointer;
  transition: all 0.2s ease;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  font-size: 12px;
}

.message-content :deep(.code-copy-button::after) {
  content: "复制";
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  font-weight: 500;
}

.message-content :deep(.code-copy-button:hover) {
  background: rgba(255, 255, 255, 0.2);
  color: white;
}

.message-content :deep(.code-copy-button:active) {
  transform: scale(0.95);
}

/* 增强的图片样式 */
.message-content :deep(.md-image-container) {
  display: inline-block;
  margin: 16px 0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  position: relative;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.message-content :deep(.md-image-container:hover) {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.message-content :deep(.md-image-container img) {
  display: block;
  max-width: 100%;
  height: auto;
  transition: filter 0.2s ease;
}

.message-content :deep(.md-image-caption) {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 8px;
  font-size: 12px;
  text-align: center;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.message-content :deep(.md-image-container:hover .md-image-caption) {
  opacity: 1;
}

.message-content :deep(.image-zoom-overlay) {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  cursor: zoom-out;
}

.message-content :deep(.zoomed-image) {
  max-width: 90%;
  max-height: 90%;
  object-fit: contain;
  box-shadow: 0 0 40px rgba(0, 0, 0, 0.5);
}

/* 增强的表格样式 */
.message-content :deep(table) {
  border-collapse: separate;
  border-spacing: 0;
  margin: 20px 0;
  width: 100%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
  border-radius: 12px;
  overflow: hidden;
}

.message-content :deep(th),
.message-content :deep(td) {
  border: none;
  border-bottom: 1px solid #eaeef2;
  padding: 12px 16px;
  text-align: left;
}

.message-content :deep(th) {
  background: linear-gradient(to right, #f8f9fc, #eef1f6);
  font-weight: 600;
  color: #4a5173;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}

.message-content :deep(tr:last-child td) {
  border-bottom: none;
}

.message-content :deep(tr:nth-child(even)) {
  background-color: #f9fafd;
}

.message-content :deep(tr:hover) {
  background-color: #f3f5fa;
}

/* 增强的引用块样式 */
.message-content :deep(blockquote) {
  border-left: 4px solid #4a6ee0;
  margin: 20px 0;
  padding: 14px 20px;
  background: linear-gradient(to right, rgba(74, 110, 224, 0.05), rgba(74, 110, 224, 0.01));
  border-radius: 0 12px 12px 0;
  color: #4a5173;
  font-style: italic;
  position: relative;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

.message-content :deep(blockquote::before) {
  content: "\201C"; /* 使用Unicode转义序列代替直接使用引号 */
  font-size: 48px;
  position: absolute;
  left: -12px;
  top: -12px;
  color: rgba(74, 110, 224, 0.2);
  font-family: Georgia, serif;
}

.message-content :deep(blockquote p) {
  margin: 0 !important;
}

/* 增强的列表样式 */
.message-content :deep(ul),
.message-content :deep(ol) {
  padding-left: 1.6em;
  margin: 16px 0;
}

.message-content :deep(li) {
  margin: 8px 0;
  padding-left: 0.5em;
  position: relative;
}

.message-content :deep(ul) {
  list-style: none;
}

.message-content :deep(ul li::before) {
  content: "•";
  color: #4a6ee0;
  font-weight: bold;
  display: inline-block;
  width: 1em;
  margin-left: -1em;
  position: absolute;
  left: 0;
}

.message-content :deep(ol) {
  list-style-type: decimal;
  counter-reset: item;
}

.message-content :deep(ol li) {
  counter-increment: item;
}

.message-content :deep(ol li::marker) {
  color: #4a6ee0;
  font-weight: bold;
}

/* 增强的链接样式 */
.message-content :deep(a) {
  color: #4a6ee0;
  text-decoration: none;
  border-bottom: 1px solid rgba(74, 110, 224, 0.3);
  transition: all 0.2s ease;
  font-weight: 500;
  padding: 0 1px;
}

.message-content :deep(a:hover) {
  color: #3a56b4;
  background-color: rgba(74, 110, 224, 0.08);
  border-bottom-color: rgba(58, 86, 180, 0.6);
  border-radius: 2px;
}

/* 增强的标题样式 */
.message-content :deep(h1),
.message-content :deep(h2),
.message-content :deep(h3),
.message-content :deep(h4),
.message-content :deep(h5),
.message-content :deep(h6) {
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  color: #2c3e50;
  line-height: 1.3;
  position: relative;
}

.message-content :deep(h1) {
  font-size: 2em;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 0.3em;
  margin-top: 0.8em;
  font-weight: 700;
}

.message-content :deep(h2) {
  font-size: 1.5em;
  border-bottom: 1px solid #f0f0f0;
  padding-bottom: 0.2em;
  font-weight: 600;
}

.message-content :deep(h3) {
  font-size: 1.25em;
  font-weight: 600;
}

.message-content :deep(h4) {
  font-size: 1.1em;
  font-weight: 500;
}

.message-content :deep(h1::before),
.message-content :deep(h2::before),
.message-content :deep(h3::before) {
  content: ""; /* 确保字符串正确闭合 */
  display: block;
  height: 4px;
  width: 100%;
  margin-bottom: 5px;
  background: linear-gradient(to right, rgba(74, 110, 224, 0.7), transparent);
  border-radius: 2px;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.message-content :deep(h1:hover::before),
.message-content :deep(h2:hover::before),
.message-content :deep(h3:hover::before) {
  opacity: 1;
}

/* 增强分割线样式 */
.message-content :deep(hr) {
  height: 1px;
  background: linear-gradient(to right, transparent, #e1e4e8, transparent);
  border: none;
  margin: 30px 0;
}

/* 增强的内联代码样式 */
.message-content :deep(:not(pre) > code) {
  font-family: 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
  background-color: rgba(74, 110, 224, 0.08);
  color: #4a6ee0;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.9em;
  white-space: nowrap;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  margin: 0 2px;
}

/* 更新 think 相关样式 */
.message-content :deep(.think-container) {
  margin: 1.2rem 0;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  background-color: #f9fafb;
  overflow: hidden;
  transition: all 0.3s ease;
}

.message-content :deep(.think-container.expanded) {
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

.message-content :deep(.think-header) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background-color: #f3f4f6;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid #e5e7eb;
  position: relative;
}

.message-content :deep(.think-header:hover) {
  background-color: #e5e7eb;
}

.message-content :deep(.think-title) {
  font-weight: 500;
  color: #6b7280;
  font-size: 14px;
  pointer-events: none; /* 确保标题不会干扰点击 */
}

.message-content :deep(.think-toggle-btn) {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  padding: 6px;
  min-height: 24px;
  min-width: 24px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.message-content :deep(.think-toggle-btn:hover) {
  background-color: rgba(107, 114, 128, 0.2);
}

.message-content :deep(.think-toggle-btn svg) {
  transition: transform 0.3s ease;
  pointer-events: none; /* 确保SVG不会干扰点击 */
}

.message-content :deep(.think-container:not(.expanded) .think-toggle-btn svg) {
  transform: rotate(180deg);
}

.message-content :deep(.think-content) {
  padding: 0;
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.5s cubic-bezier(0.4, 0, 0.2, 1), padding 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  font-size: 14px;
  line-height: 1.5;
  color: #4b5563;
  background-color: #f9fafb;
}

.message-content :deep(.think-loading) {
  display: flex;
  align-items: center;
  margin: 10px 0;
  color: #6b7280;
  font-style: italic;
  font-size: 13px;
}

.message-content :deep(.think-loading::before) {
  content: "";
  display: inline-block;
  width: 14px;
  height: 14px;
  margin-right: 8px;
  border: 2px solid rgba(107, 114, 128, 0.3);
  border-top-color: #6b7280;
  border-radius: 50%;
  animation: think-spin 1s linear infinite;
}

.message-content :deep(.think-container.expanded .think-content) {
  padding: 16px;
  overflow: auto;
  max-height: 10000px; /* 设置一个足够大的值 */
  transition: max-height 0.5s cubic-bezier(0.4, 0, 0.2, 1), padding 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* 内容区块的内部元素样式 */
.message-content :deep(.think-content p) {
  margin-bottom: 12px !important;
}

.message-content :deep(.think-content p:last-child) {
  margin-bottom: 0 !important;
}

@keyframes think-spin {
  to {
    transform: rotate(360deg);
  }
}

/* 数学公式样式 */
.message-content :deep(.math-inline) {
  display: inline-flex !important;
  align-items: center;
  margin: 0 3px;
  vertical-align: middle;
  font-size: 1.1em;
}

.message-content :deep(.math-block) {
  display: flex !important;
  justify-content: center;
  margin: 20px 0;
  padding: 16px;
  background-color: rgba(247, 247, 252, 0.7);
  border-radius: 8px;
  overflow-x: auto;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.message-content :deep(.math-block:hover) {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.message-content :deep(.katex) {
  font-size: 1.1em !important;
  font-family: 'KaTeX_Math', 'Times New Roman', serif !important;
  line-height: 1.3 !important;
}

.message-content :deep(.katex-display) {
  margin: 0 !important;
  padding: 4px 0 !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
}

.message-content :deep(.katex-display > .katex) {
  white-space: nowrap !important;
}

.message-content :deep(.math-error) {
  color: #e53935;
  background-color: rgba(229, 57, 53, 0.1);
  padding: 8px 12px;
  border-radius: 6px;
  border-left: 3px solid #e53935;
  font-family: 'Courier New', monospace;
  white-space: pre-wrap;
  margin: 10px 0;
}

/* 代码块容器样式 */
.message-content :deep(.code-block-container) {
  position: relative;
  margin: 24px 0;
  border-radius: 12px;
  overflow: hidden;
  background-color: #f8fafc;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  border: 1px solid rgba(59, 130, 246, 0.1);
}

.message-content :deep(.code-block-container:hover) {
  transform: translateY(-2px);
  box-shadow: 0 6px 24px rgba(59, 130, 246, 0.12);
}

/* 代码块头部 */
.message-content :deep(.code-block-header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background-color: #1e3a8a;
  color: #e0e7ff;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

/* 语言标签 */
.message-content :deep(.code-lang-tag) {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 6px;
  background-color: rgba(255, 255, 255, 0.15);
  color: #e0e7ff;
  font-weight: 500;
  letter-spacing: 0.3px;
  font-size: 12px;
}

/* 复制按钮 */
.message-content :deep(.code-copy-btn) {
  background: rgba(255, 255, 255, 0.15);
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border-radius: 6px;
  padding: 6px 12px;
  color: #e0e7ff;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 12px;
  font-weight: 500;
}

.message-content :deep(.code-copy-btn:hover) {
  background: rgba(255, 255, 255, 0.2);
  color: #ffffff;
}

.message-content :deep(.code-copy-btn.copied) {
  background-color: rgba(46, 213, 115, 0.2);
  color: #2ed573;
}

/* 代码块样式 */
.message-content :deep(pre) {
  margin: 0 !important;
  border-radius: 0 !important;
  background-color: #f8fafc !important;
  padding: 16px !important;
  overflow-x: auto;
  font-family: 'Fira Code', 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.6;
  tab-size: 2;
}

/* 行号样式 */
.message-content :deep(pre.line-numbers) {
  padding-left: 3.8em !important;
  counter-reset: linenumber;
}

.message-content :deep(pre.line-numbers > code) {
  position: relative;
  white-space: inherit;
  font-family: 'Fira Code', 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace !important;
}

.message-content :deep(.line-numbers-rows) {
  position: absolute;
  top: 0;
  left: -3.8em;
  width: 3em;
  letter-spacing: -1px;
  border-right: 1px solid rgba(59, 130, 246, 0.1);
  user-select: none;
  pointer-events: none;
  font-size: 14px;
  font-family: 'Fira Code', 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
  padding-top: 16px;
  padding-bottom: 16px;
}

.message-content :deep(.line-numbers-rows > span) {
  display: block;
  counter-increment: linenumber;
  color: rgba(59, 130, 246, 0.4);
  text-align: right;
  padding: 0 8px;
}

.message-content :deep(.line-numbers-rows > span::before) {
  content: counter(linenumber);
  color: rgba(59, 130, 246, 0.4);
}

/* 代码高亮调优 - 深蓝白色主题 */
.message-content :deep(.token.comment),
.message-content :deep(.token.prolog),
.message-content :deep(.token.doctype),
.message-content :deep(.token.cdata) {
  color: #64748b;
  font-style: italic;
}

.message-content :deep(.token.punctuation) {
  color: #475569;
}

.message-content :deep(.token.property),
.message-content :deep(.token.tag),
.message-content :deep(.token.boolean),
.message-content :deep(.token.number),
.message-content :deep(.token.constant),
.message-content :deep(.token.symbol) {
  color: #0f766e;
}

.message-content :deep(.token.selector),
.message-content :deep(.token.attr-name),
.message-content :deep(.token.string),
.message-content :deep(.token.char),
.message-content :deep(.token.builtin) {
  color: #0369a1;
}

.message-content :deep(.token.operator),
.message-content :deep(.token.entity),
.message-content :deep(.token.url),
.message-content :deep(.language-css .token.string),
.message-content :deep(.style .token.string) {
  color: #475569;
}

.message-content :deep(.token.atrule),
.message-content :deep(.token.attr-value),
.message-content :deep(.token.keyword) {
  color: #1e40af;
  font-weight: bold;
}

.message-content :deep(.token.function) {
  color: #0d9488;
}

.message-content :deep(.token.regex),
.message-content :deep(.token.important),
.message-content :deep(.token.variable) {
  color: #be123c;
}

.message-content :deep(.token.important),
.message-content :deep(.token.bold) {
  font-weight: bold;
}

.message-content :deep(.token.italic) {
  font-style: italic;
}

.message-content :deep(.token.entity) {
  cursor: help;
}

/* Prism工具栏样式优化 */
.message-content :deep(div.code-toolbar) {
  position: relative;
}

.message-content :deep(div.code-toolbar > .toolbar) {
  position: absolute;
  top: 10px;
  right: 10px;
  transition: opacity 0.3s ease-in-out;
  opacity: 0;
}

.message-content :deep(div.code-toolbar:hover > .toolbar) {
  opacity: 1;
}

.message-content :deep(div.code-toolbar > .toolbar > .toolbar-item) {
  display: inline-block;
  margin-left: 8px;
}

.message-content :deep(div.code-toolbar > .toolbar > .toolbar-item > button) {
  display: flex;
  align-items: center;
  gap: 4px;
  background: rgba(93, 116, 209, 0.12);
  color: #3a4b7c;
  border: none;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
  cursor: pointer;
  transition: all 0.2s ease;
}

.message-content :deep(div.code-toolbar > .toolbar > .toolbar-item > button:hover) {
  background: rgba(93, 116, 209, 0.18);
  color: #2b4098;
}

.message-content :deep(div.code-toolbar > .toolbar > .toolbar-item > button:focus) {
  outline: none;
}

.message-content :deep(div.code-toolbar > .toolbar > .toolbar-item > button > span) {
  margin-left: 4px;
}

.message-content :deep(div.code-toolbar > .toolbar > .toolbar-item > .show-language) {
  display: inline-block;
  background: rgba(93, 116, 209, 0.08);
  color: #3a4b7c;
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.3px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
}

/* 针对移动设备的响应式样式 */
@media (max-width: 768px) {
  .message-content :deep(.think-container) {
    margin: 1.2rem 0;
    border-radius: 14px;
  }
  
  .message-content :deep(.think-header) {
    padding: 12px 16px;
  }
  
  .message-content :deep(.think-icon) {
    margin-right: 10px;
    width: 32px;
    height: 32px;
    font-size: 18px;
  }
  
  .message-content :deep(.think-title) {
    font-size: 15px;
  }
  
  .message-content :deep(.think-toggle-btn) {
    height: 28px;
    width: 28px;
    padding: 6px;
  }
  
  .message-content :deep(.think-container.expanded .think-content) {
    padding: 18px;
  }
  
  .message-content :deep(.think-content) {
    font-size: 14px;
    line-height: 1.6;
  }
}

@media (max-width: 480px) {
  .message-content :deep(.think-container) {
    margin: 1rem 0;
    border-radius: 12px;
  }
  
  .message-content :deep(.think-header) {
    padding: 10px 14px;
  }
  
  .message-content :deep(.think-icon) {
    margin-right: 8px;
    width: 28px;
    height: 28px;
    font-size: 16px;
  }
  
  .message-content :deep(.think-title) {
    font-size: 14px;
  }
  
  .message-content :deep(.think-toggle-btn) {
    height: 24px;
    width: 24px;
    padding: 4px;
  }
  
  .message-content :deep(.think-container.expanded .think-content) {
    padding: 14px;
  }
  
  .message-content :deep(.think-content) {
    font-size: 13.5px;
    line-height: 1.5;
  }
}

/* 提升思考块内部元素的样式 */
.message-content :deep(.think-content h1, 
                      .think-content h2, 
                      .think-content h3, 
                      .think-content h4) {
  font-weight: 600;
  margin: 1.2em 0 0.6em;
  color: #4b5563;
}

.message-content :deep(.think-content h1) {
  font-size: 1.5em;
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 0.3em;
}

.message-content :deep(.think-content h2) {
  font-size: 1.3em;
}

.message-content :deep(.think-content h3) {
  font-size: 1.1em;
}

.message-content :deep(.think-content h4) {
  font-size: 1em;
}

.message-content :deep(.think-content strong) {
  font-weight: 600;
  color: #4b5563;
}

.message-content :deep(.think-content em) {
  font-style: italic;
  color: #6b7280;
}

.message-content :deep(.think-content code) {
  background-color: #e5e7eb;
  padding: 2px 4px;
  border-radius: 4px;
  font-family: 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
  font-size: 0.9em;
  color: #4b5563;
}

.message-content :deep(.think-content blockquote) {
  border-left: 3px solid #9ca3af;
  background-color: #f3f4f6;
  padding: 0.8em 1em;
  margin: 1em 0;
  color: #4b5563;
  border-radius: 0 4px 4px 0;
}

.message-content :deep(.think-content a) {
  color: #6b7280;
  text-decoration: none;
  border-bottom: 1px solid #9ca3af;
  transition: all 0.2s ease;
}

.message-content :deep(.think-content a:hover) {
  color: #4b5563;
  border-bottom-color: #6b7280;
}
</style>
