<template>
  <el-dialog v-model="visible" title="登录提示" width="30%" :show-close="true">
    <template #header>
      <div class="dialog-header">
        <span>登录提示</span>
        <i class="el-icon-close" @click="handleClose" style="cursor: pointer;"></i>
      </div>
    </template>
    <div style="text-align: center">
      <p style="margin-bottom: 20px; font-size: 16px;">您需要登录后才能继续操作</p>
      <el-button type="primary" @click="handleLogin" style="width: 120px;">
        立即登录
      </el-button>
    </div>
  </el-dialog>
</template>

<script setup>
import { ref, defineProps, defineEmits } from "vue";
import { useRouter } from "vue-router";

// 定义接收的 props
const props = defineProps({
  visible: {
    type: Boolean,
    default: false,
  },
});

// 定义触发的事件
const emits = defineEmits(["close"]);

// 获取路由实例
const router = useRouter();

// 关闭弹窗的方法
const handleClose = () => {
  emits("close");
};

// 处理登录跳转的方法
const handleLogin = () => {
  router.push("/login");
  emits("close");
};
</script>

<style scoped>
.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-right: 20px;
}

.dialog-header .el-icon-close {
  font-size: 18px;
  color: #999;
  transition: color 0.2s;
}

.dialog-header .el-icon-close:hover {
  color: #666;
}
</style>