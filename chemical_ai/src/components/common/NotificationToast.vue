<template>
  <div class="notification-toast" v-if="show" @click="$emit('close')">
    <div class="notification-content" :class="type">
      <div class="notification-icon">{{ icon }}</div>
      <div class="notification-message">
        <h4>{{ title }}</h4>
        <p>{{ message }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, watch } from 'vue';

// Props
const props = defineProps({
  show: {
    type: Boolean,
    required: true
  },
  title: {
    type: String,
    default: '提交成功'
  },
  message: {
    type: String,
    default: '操作已完成'
  },
  type: {
    type: String,
    default: 'success',
    validator: (value) => ['success', 'error', 'warning', 'info'].includes(value)
  },
  duration: {
    type: Number,
    default: 3000
  },
  autoClose: {
    type: Boolean,
    default: true
  }
});

// Emits
const emit = defineEmits(['close']);

// Computed
const icon = () => {
  switch (props.type) {
    case 'success': return '✓';
    case 'error': return '✕';
    case 'warning': return '!';
    case 'info': return 'i';
    default: return '✓';
  }
};

// Auto close
watch(() => props.show, (newVal) => {
  if (newVal && props.autoClose) {
    setTimeout(() => {
      emit('close');
    }, props.duration);
  }
});

// Auto close on mount if shown immediately
onMounted(() => {
  if (props.show && props.autoClose) {
    setTimeout(() => {
      emit('close');
    }, props.duration);
  }
});
</script>

<style scoped>
.notification-toast {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 2000;
  animation: slideIn 0.3s ease-out forwards;
  cursor: pointer;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.notification-content {
  display: flex;
  align-items: flex-start;
  background: white;
  border-radius: 10px;
  padding: 16px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  min-width: 300px;
  max-width: 400px;
  border-left: 4px solid #10b981;
}

.notification-content.success {
  border-left-color: #10b981;
}

.notification-content.error {
  border-left-color: #ef4444;
}

.notification-content.warning {
  border-left-color: #f59e0b;
}

.notification-content.info {
  border-left-color: #3b82f6;
}

.notification-icon {
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: bold;
  margin-right: 12px;
  flex-shrink: 0;
}

.notification-content.error .notification-icon {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
}

.notification-content.warning .notification-icon {
  background: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

.notification-content.info .notification-icon {
  background: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
}

.notification-message {
  flex: 1;
}

.notification-message h4 {
  margin: 0 0 4px 0;
  color: #1a1f36;
  font-size: 1rem;
}

.notification-message p {
  margin: 0;
  color: #6b7280;
  font-size: 0.9rem;
  line-height: 1.4;
}

/* Dark theme styles */
body.dark-theme .notification-content {
  background: #1f2937;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

body.dark-theme .notification-icon {
  background: rgba(16, 185, 129, 0.2);
}

body.dark-theme .notification-content.error .notification-icon {
  background: rgba(239, 68, 68, 0.2);
}

body.dark-theme .notification-content.warning .notification-icon {
  background: rgba(245, 158, 11, 0.2);
}

body.dark-theme .notification-content.info .notification-icon {
  background: rgba(59, 130, 246, 0.2);
}

body.dark-theme .notification-message h4 {
  color: #f3f4f6;
}

body.dark-theme .notification-message p {
  color: #d1d5db;
}
</style> 