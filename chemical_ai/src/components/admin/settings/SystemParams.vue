<template>
  <div class="system-params-container">

    <el-card class="config-card">
      <template #header>
        <div class="card-header">
          <h3>系统配置参数</h3>
          <el-button type="primary" size="small" @click="refreshConfigs">刷新</el-button>
        </div>
      </template>
      
      <el-table v-loading="loading.configs" :data="configList" style="width: 100%" border>
        <el-table-column prop="description" label="描述" width="250" />
        <el-table-column prop="config_value" label="配置值" />
        <el-table-column prop="updated_at" label="更新时间" width="180">
          <template #default="scope">
            {{ formatDateTime(scope.row.updated_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120" align="center">
          <template #default="scope">
            <el-button 
              type="primary" 
              size="small" 
              plain
              round
              class="edit-button"
              @click="handleEditConfig(scope.row)"
            >
              <el-icon class="button-icon"><Edit /></el-icon>
              编辑
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card class="version-card">
      <template #header>
        <div class="card-header">
          <h3>系统版本信息</h3>
          <el-button type="primary" size="small" @click="refreshVersions">刷新</el-button>
        </div>
      </template>
      
      <el-table v-loading="loading.versions" :data="versionList" style="width: 100%" border>
        <el-table-column prop="version_id" label="ID" width="80" />
        <el-table-column prop="version_number" label="版本号" width="120" />
        <el-table-column prop="knowledge_base_version" label="知识库版本" width="120" />
        <el-table-column prop="release_date" label="发布日期" width="120" />
        <el-table-column prop="update_notes" label="更新说明" />
        <el-table-column label="当前版本" width="120">
          <template #default="scope">
            <el-tag v-if="scope.row.is_current" type="success">当前版本</el-tag>
            <el-tag v-else type="info">历史版本</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120">
          <template #default="scope">
            <el-button 
              size="small" 
              :disabled="scope.row.is_current" 
              @click="setAsCurrentVersion(scope.row)"
            >
              设为当前版本
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 添加新版本的按钮 -->
    <div class="action-bar">
      <el-button type="primary" @click="showAddVersionDialog">添加新版本</el-button>
    </div>

    <!-- 编辑配置对话框 -->
    <el-dialog v-model="dialogs.editConfig" title="编辑系统配置" width="550px" destroy-on-close custom-class="config-edit-dialog">
      <div class="dialog-content">
        <div class="dialog-header-info">
          <el-alert
            type="info"
            :closable="false"
            show-icon
          >
            <template #title>
              <div class="alert-title">
                <span>您正在编辑 <strong>{{ currentConfig.config_key }}</strong> 配置项</span>
              </div>
            </template>
            <p class="alert-desc">系统配置项修改后会立即生效，请谨慎操作。</p>
          </el-alert>
        </div>

        <el-form :model="configForm" ref="configFormRef" :rules="configRules" label-width="90px" class="config-form" label-position="top">
          <el-form-item label="配置键名" prop="config_key">
            <el-input v-model="configForm.config_key" disabled>
              <template #prefix>
                <el-icon><Key /></el-icon>
              </template>
            </el-input>
            <div class="form-item-hint">配置键名不可修改</div>
          </el-form-item>
          
          <el-form-item label="配置值" prop="config_value">
            <!-- 根据配置键名显示不同的输入控件 -->
            <el-date-picker 
              v-if="currentConfig.config_key === 'knowledge_base_update_date'"
              v-model="configForm.dateValue"
              type="date"
              placeholder="请选择日期"
              format="YYYY-MM-DD"
              value-format="YYYY-MM-DD"
              style="width: 100%;"
              @update:model-value="handleDateChange"
            />
            <el-input 
              v-else
              v-model="configForm.config_value" 
              placeholder="请输入配置值"
            >
              <template #prefix>
                <el-icon><Setting /></el-icon>
              </template>
            </el-input>
            <div class="form-item-hint">
              <template v-if="currentConfig.config_key === 'knowledge_base_update_date'">
                请选择知识库最后更新日期
              </template>
              <template v-else>
                配置值将直接应用于系统
              </template>
            </div>
          </el-form-item>
          
          <el-form-item label="描述" prop="description" class="description-item">
            <el-input v-model="configForm.description" placeholder="请输入配置描述" type="textarea" rows="2">
            </el-input>
            <div class="form-item-hint">描述将帮助其他管理员理解该配置项的用途</div>
          </el-form-item>
        </el-form>
        
        <div class="config-update-info">
          <p class="update-warning">
            <el-icon><Warning /></el-icon>
            请注意：配置修改会记录在系统日志中，并可能影响系统运行。
          </p>
        </div>
      </div>
      
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="closeConfigDialog" plain>取消</el-button>
          <el-button type="primary" @click="submitConfigChanges" :loading="loading.saveConfig">
            <el-icon><Check /></el-icon> 保存
          </el-button>
        </div>
      </template>
    </el-dialog>

    <!-- 添加新版本对话框 -->
    <el-dialog v-model="dialogs.addVersion" title="添加新版本" width="500px">
      <el-form :model="newVersion" label-width="120px">
        <el-form-item label="版本号" required>
          <el-input v-model="newVersion.version_number" placeholder="例如: 1.0.1" />
        </el-form-item>
        <el-form-item label="知识库版本">
          <el-input v-model="newVersion.knowledge_base_version" placeholder="例如: v3.2.2" />
        </el-form-item>
        <el-form-item label="发布日期" required>
          <el-date-picker v-model="newVersion.release_date" type="date" placeholder="选择日期" />
        </el-form-item>
        <el-form-item label="更新说明">
          <el-input v-model="newVersion.update_notes" type="textarea" rows="4" />
        </el-form-item>
        <el-form-item label="设为当前版本">
          <el-switch v-model="newVersion.is_current" />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogs.addVersion = false">取消</el-button>
          <el-button type="primary" @click="saveNewVersion">保存</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import axios from 'axios';
import { API_BASE_URL } from '../../../config';
import { Edit, Check, Key, Setting, InfoFilled, Warning } from '@element-plus/icons-vue';

// 加载状态
const loading = reactive({
  configs: false,
  versions: false,
  saveConfig: false
});

// 数据列表
const configList = ref([]);
const versionList = ref([]);

// 对话框控制
const dialogs = reactive({
  editConfig: false,
  addVersion: false
});

// 当前编辑的配置
const currentConfig = ref({});
const configForm = ref({
  config_key: '',
  config_value: '',
  description: '',
  dateValue: null
});
const configFormRef = ref(null);

// 表单验证规则
const configRules = {
  config_value: [
    { required: true, message: '配置值不能为空', trigger: 'blur' },
  ],
  description: [
    { max: 255, message: '描述不能超过255个字符', trigger: 'blur' }
  ]
};

// 新版本表单
const newVersion = ref({
  version_number: '',
  knowledge_base_version: '',
  release_date: new Date(),
  update_notes: '',
  is_current: false
});

// 获取管理员ID
const getAdminId = () => {
  // 从localStorage或其他存储方式获取管理员ID
  const adminId = localStorage.getItem('adminId') || localStorage.getItem('userId');
  return adminId ? parseInt(adminId) : 1; // 默认返回1，根据实际情况调整
};

// 加载系统配置
const loadConfigs = async () => {
  loading.configs = true;
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/settings/system-configs`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });

    if (response.data.code === 200) {
      configList.value = response.data.data;
    } else {
      ElMessage.error(response.data.message || '获取系统配置失败');
    }
  } catch (error) {
    console.error('获取系统配置失败:', error);
    ElMessage.error('获取系统配置失败');
  } finally {
    loading.configs = false;
  }
};

// 加载系统版本信息
const loadVersions = async () => {
  loading.versions = true;
  try {
    const response = await axios.get(`${API_BASE_URL}/admin/settings/system-versions`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });

    if (response.data.code === 200) {
      versionList.value = response.data.data;
    } else {
      ElMessage.error(response.data.message || '获取系统版本失败');
    }
  } catch (error) {
    console.error('获取系统版本失败:', error);
    ElMessage.error('获取系统版本失败');
  } finally {
    loading.versions = false;
  }
};

// 刷新配置列表
const refreshConfigs = () => {
  loadConfigs();
};

// 刷新版本列表
const refreshVersions = () => {
  loadVersions();
};

// 处理编辑配置
const handleEditConfig = (config) => {
  currentConfig.value = { ...config };
  configForm.value = {
    config_key: config.config_key,
    config_value: config.config_value,
    description: config.description,
    dateValue: config.config_key === 'knowledge_base_update_date' ? config.config_value : null
  };
  dialogs.editConfig = true;
};

// 处理日期变更
const handleDateChange = (date) => {
  configForm.value.config_value = date;
};

// 关闭配置对话框
const closeConfigDialog = () => {
  if (configFormRef.value) {
    configFormRef.value.resetFields();
  }
  dialogs.editConfig = false;
};

// 提交配置更改前验证
const submitConfigChanges = async () => {
  if (!configFormRef.value) return;
  
  await configFormRef.value.validate((valid) => {
    if (valid) {
      saveConfigChanges();
    } else {
      return false;
    }
  });
};

// 保存配置更改
const saveConfigChanges = async () => {
  loading.saveConfig = true;
  try {
    // 对于日期类型的配置，确保使用正确的格式
    let configValue = configForm.value.config_value;
    if (currentConfig.value.config_key === 'knowledge_base_update_date' && configForm.value.dateValue) {
      configValue = configForm.value.dateValue;
    }
    
    const payload = {
      config_value: configValue,
      description: configForm.value.description
    };
    
    const response = await axios.put(
      `${API_BASE_URL}/admin/settings/system-configs/${currentConfig.value.config_id}`, 
      payload,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      }
    );
    
    if (response.data.code === 200) {
      ElMessage({
        type: 'success',
        message: '配置已成功更新',
        duration: 2000
      });
      dialogs.editConfig = false;
      
      // 刷新配置列表
      loadConfigs();
    } else {
      ElMessage.error(response.data.message || '保存配置失败');
    }
  } catch (error) {
    console.error('保存配置失败', error);
    ElMessage.error('保存配置失败');
  } finally {
    loading.saveConfig = false;
  }
};

// 设置当前版本
const setAsCurrentVersion = async (version) => {
  try {
    ElMessageBox.confirm(
      `确定要将版本 ${version.version_number} 设置为当前版本吗？`,
      '确认操作',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }
    ).then(async () => {
      const response = await axios.put(
        `${API_BASE_URL}/admin/settings/system-versions/${version.version_id}/set-current`, 
        {},
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      if (response.data.code === 200) {
        ElMessage.success('当前版本已更新');
        // 刷新版本列表
        loadVersions();
      } else {
        ElMessage.error(response.data.message || '设置当前版本失败');
      }
    }).catch(() => {
      // 用户取消操作
    });
  } catch (error) {
    console.error('设置当前版本失败', error);
    ElMessage.error('设置当前版本失败');
  }
};

// 显示添加版本对话框
const showAddVersionDialog = () => {
  newVersion.value = {
    version_number: '',
    knowledge_base_version: '',
    release_date: new Date(),
    update_notes: '',
    is_current: false
  };
  dialogs.addVersion = true;
};

// 保存新版本
const saveNewVersion = async () => {
  if (!newVersion.value.version_number || !newVersion.value.release_date) {
    ElMessage.warning('请填写必填字段');
    return;
  }
  
  try {
    // 格式化日期
    const formattedDate = newVersion.value.release_date instanceof Date 
      ? newVersion.value.release_date.toISOString().split('T')[0]
      : newVersion.value.release_date;
    
    const payload = {
      version_number: newVersion.value.version_number,
      knowledge_base_version: newVersion.value.knowledge_base_version,
      release_date: formattedDate,
      update_notes: newVersion.value.update_notes,
      is_current: newVersion.value.is_current
    };
    
    const response = await axios.post(
      `${API_BASE_URL}/admin/settings/system-versions`, 
      payload,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      }
    );
    
    if (response.data.code === 200) {
      ElMessage.success('新版本已添加');
      dialogs.addVersion = false;
      
      // 刷新版本列表
      loadVersions();
    } else {
      ElMessage.error(response.data.message || '保存新版本失败');
    }
  } catch (error) {
    console.error('保存新版本失败', error);
    ElMessage.error('保存新版本失败');
  }
};

// 格式化日期时间显示
const formatDateTime = (dateTimeStr) => {
  if (!dateTimeStr) return '';
  
  try {
    const date = new Date(dateTimeStr);
    
    // 使用本地日期时间格式
    return new Intl.DateTimeFormat('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    }).format(date);
    
    // 或者使用简单的自定义格式化：
    // const year = date.getFullYear();
    // const month = String(date.getMonth() + 1).padStart(2, '0');
    // const day = String(date.getDate()).padStart(2, '0');
    // const hours = String(date.getHours()).padStart(2, '0');
    // const minutes = String(date.getMinutes()).padStart(2, '0');
    // const seconds = String(date.getSeconds()).padStart(2, '0');
    // return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  } catch (error) {
    console.error('日期格式化错误:', error);
    return dateTimeStr; // 如果出错返回原字符串
  }
};

// 初始化
onMounted(() => {
  loadConfigs();
  loadVersions();
});
</script>

<style scoped>
.system-params-container {
  padding: 20px;
}

.page-header {
  margin-bottom: 20px;
}

.config-card,
.version-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3 {
  margin: 0;
}

.action-bar {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

:deep(.el-dialog__body) {
  padding-top: 10px;
}

:deep(.el-table .cell) {
  word-break: break-word;
}

.config-edit-container {
  padding: 20px;
}

.config-info-alert {
  margin-bottom: 20px;
}

.alert-title {
  display: flex;
  align-items: center;
}

.alert-title .el-icon {
  margin-right: 8px;
}

.config-form {
  width: 100%;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  padding-top: 20px;
}

:deep(.el-input-number) {
  width: 100%;
}

:deep(.el-date-editor) {
  width: 100% !important;
}

:deep(.el-divider__text) {
  font-weight: bold;
  color: var(--el-color-primary);
}

/* 移除按钮点击时的黑色边框 */
:deep(.el-button) {
  outline: none !important;
}

:deep(.el-button:focus),
:deep(.el-button:active) {
  outline: none !important;
  border-color: transparent;
  box-shadow: none !important;
}

:deep(.el-button.is-circle:focus),
:deep(.el-button.is-circle:active) {
  outline: none !important;
  border-color: var(--el-button-hover-border-color);
}

/* 为圆形按钮保持悬浮时的边框颜色 */
:deep(.el-button.is-circle.el-button--primary:focus),
:deep(.el-button.is-circle.el-button--primary:active) {
  border-color: var(--el-button-hover-bg-color);
}

/* 编辑按钮样式 */
.edit-button {
  transition: all 0.3s;
}

.edit-button:hover {
  transform: scale(1.05);
}

.button-icon {
  margin-right: 4px;
  vertical-align: middle;
}

/* 弹窗样式 */
.dialog-header-info {
  margin-bottom: 25px;
}

.alert-desc {
  font-size: 12px;
  margin-top: 6px;
  color: #606266;
}

.form-item-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
  margin-left: 2px;
}

.dialog-divider {
  margin: 20px 0;
}

.config-update-info {
  padding: 10px 0;
}

.update-warning {
  display: flex;
  align-items: flex-start;
  color: #e6a23c;
  font-size: 13px;
  margin: 0;
  line-height: 1.5;
}

.update-warning .el-icon {
  margin-right: 10px;
  font-size: 16px;
  color: #e6a23c;
  flex-shrink: 0;
}

/* 配置编辑对话框样式 */
:deep(.config-edit-dialog) {
  max-height: 90vh;
}

:deep(.config-edit-dialog .el-dialog__body) {
  padding: 16px 20px;
  max-height: calc(90vh - 110px);
  overflow-y: auto;
}

.dialog-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.dialog-header-info {
  margin-bottom: 5px;
}

:deep(.el-alert) {
  border-radius: 6px;
  padding: 10px 15px;
}

:deep(.el-alert__title) {
  font-size: 14px;
  line-height: 1.4;
}

.alert-desc {
  font-size: 12px;
  margin-top: 4px;
  color: #606266;
  line-height: 1.4;
}

.config-form {
  margin: 0;
}

:deep(.el-form-item) {
  margin-bottom: 12px;
}

:deep(.el-form-item__label) {
  padding-bottom: 4px;
  line-height: 1.2;
  font-weight: 500;
}

.form-item-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 3px;
  line-height: 1.2;
}

:deep(.el-textarea__inner) {
  min-height: 60px;
}

.description-item {
  margin-bottom: 10px;
}

.config-update-info {
  background-color: #fef9e7;
  border-radius: 6px;
  padding: 8px 12px;
  border: 1px solid #faecd8;
  margin-top: 0;
}

.update-warning {
  display: flex;
  align-items: flex-start;
  color: #e6a23c;
  font-size: 12px;
  margin: 0;
  line-height: 1.4;
}

.update-warning .el-icon {
  margin-right: 8px;
  font-size: 16px;
  color: #e6a23c;
  flex-shrink: 0;
  margin-top: 1px;
}

:deep(.el-dialog__footer) {
  padding: 10px 20px 16px;
  border-top: 1px solid #f0f0f0;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>