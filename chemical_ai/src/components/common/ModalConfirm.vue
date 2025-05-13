<template>
  <div class="modal-overlay" v-if="show">
    <div class="modal-content">
      <div class="modal-header">
        <h3>确认操作</h3>
        <button class="close-button" @click="$emit('cancel')">×</button>
      </div>
      <div class="modal-body">
        <p class="confirm-message">{{ message }}</p>
      </div>
      <div class="modal-footer">
        <button class="cancel-button" @click="$emit('cancel')">取消</button>
        <button class="submit-button danger" @click="$emit('confirm')">确认</button>
      </div>
    </div>
  </div>
</template>

<script setup>
// Props
defineProps({
  show: {
    type: Boolean,
    required: true
  },
  message: {
    type: String,
    required: true
  }
});

// Emits
defineEmits(['confirm', 'cancel']);
</script>

<style scoped>
.modal-overlay {
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
  max-width: 400px;
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

.confirm-message {
  font-size: 1rem;
  color: #1a1f36;
  text-align: center;
  margin: 10px 0;
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

.submit-button.danger {
  background: #ef4444;
}

.cancel-button:hover {
  background: #f8f9fa;
}

.submit-button:hover {
  background: #4338ca;
}

.submit-button.danger:hover {
  background: #dc2626;
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

body.dark-theme .confirm-message {
  color: #f3f4f6;
}
</style> 