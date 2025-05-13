<template>
  <div class="admin-management">
    
    <!-- 搜索和筛选区域 -->
    <div class="search-container">
      <el-form :inline="true" :model="searchForm" class="search-form">
        <el-form-item label="手机号">
          <el-input v-model="searchForm.phoneNumber" placeholder="请输入手机号"></el-input>
        </el-form-item>
        <el-form-item label="姓名">
          <el-input v-model="searchForm.fullName" placeholder="请输入姓名"></el-input>
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="searchForm.role" placeholder="请选择角色">
            <el-option label="全部" value=""></el-option>
            <el-option label="管理员" value="admin"></el-option>
            <el-option label="操作员" value="operator"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="searchForm.status" placeholder="请选择状态">
            <el-option label="全部" value=""></el-option>
            <el-option label="正常" value="1"></el-option>
            <el-option label="禁用" value="0"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">查询</el-button>
          <el-button @click="resetSearchForm">重置</el-button>
          <el-button type="success" @click="handleAddAdmin">添加管理员</el-button>
        </el-form-item>
      </el-form>
    </div>
    
    <!-- 管理员列表 -->
    <div class="table-container">
      <el-table 
        :data="adminList" 
        border 
        style="width: 100%" 
        v-loading="loading"
        :cell-style="{ padding: '12px 8px', fontSize: '14px', whiteSpace: 'nowrap' }"
        :header-cell-style="{ background:'#f5f7fa', color:'#606266', fontWeight: 'bold' }">
        <el-table-column prop="admin_id" label="ID" min-width="80" align="center"></el-table-column>
        <el-table-column prop="phone_number" label="手机号" min-width="120" align="center">
          <template #default="scope">
            <span class="ellipsis-cell" :title="scope.row.phone_number">{{ scope.row.phone_number }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="full_name" label="姓名" min-width="120" align="center"></el-table-column>
        <el-table-column prop="email" label="邮箱" min-width="180" align="center">
          <template #default="scope">
            <span class="ellipsis-cell" :title="scope.row.email">{{ scope.row.email }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="role" label="角色" min-width="100" align="center">
          <template #default="scope">
            <div class="tag-container">
              <el-tag :type="scope.row.role === 'admin' ? 'danger' : 'success'" class="admin-tag">
                {{ scope.row.role === 'admin' ? '管理员' : '操作员' }}
              </el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" min-width="100" align="center">
          <template #default="scope">
            <div class="tag-container">
              <el-tag :type="scope.row.status === 1 ? 'success' : 'danger'" class="admin-tag">
                {{ scope.row.status === 1 ? '正常' : '禁用' }}
              </el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="last_login_time" label="最后登录时间" min-width="180" align="center">
          <template #default="scope">
            <span class="ellipsis-cell" :title="scope.row.last_login_time">{{ formatDateTime(scope.row.last_login_time) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" min-width="180" align="center">
          <template #default="scope">
            <span class="ellipsis-cell" :title="scope.row.created_at">{{ formatDateTime(scope.row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" min-width="220" align="center">
          <template #default="scope">
            <el-button 
              size="small"
              type="primary"
              @click="handleEdit(scope.row)"
              style="margin-right: 8px"
            >
              编辑
            </el-button>
            <el-button 
              size="small"
              :type="scope.row.status === 1 ? 'danger' : 'success'"
              @click="handleStatusChange(scope.row)"
              style="margin-right: 8px"
            >
              {{ scope.row.status === 1 ? '禁用' : '启用' }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <!-- 分页 -->
      <div class="pagination-container">
        <el-pagination
          background
          layout="total, sizes, prev, pager, next, jumper"
          :total="total"
          :page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :current-page="currentPage"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        ></el-pagination>
      </div>
    </div>
    
    <!-- 添加/编辑管理员对话框 -->
    <el-dialog 
      :title="isEdit ? '编辑管理员' : '添加管理员'" 
      v-model="dialogVisible" 
      width="500px"
      :close-on-click-modal="false"
    >
      <el-form 
        :model="adminForm" 
        :rules="adminRules" 
        ref="adminFormRef" 
        label-width="100px"
        :disabled="formLoading"
      >
        <el-form-item label="手机号" prop="phone_number">
          <el-input v-model="adminForm.phone_number" placeholder="请输入手机号"></el-input>
        </el-form-item>
        <el-form-item label="姓名" prop="full_name">
          <el-input v-model="adminForm.full_name" placeholder="请输入姓名"></el-input>
        </el-form-item>
        <el-form-item label="邮箱" prop="email">
          <el-input v-model="adminForm.email" placeholder="请输入邮箱"></el-input>
        </el-form-item>
        <el-form-item label="角色" prop="role">
          <el-select v-model="adminForm.role" placeholder="请选择角色" style="width: 100%;">
            <el-option label="管理员" value="admin"></el-option>
            <el-option label="操作员" value="operator"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-select v-model="adminForm.status" placeholder="请选择状态" style="width: 100%;">
            <el-option label="正常" :value="1"></el-option>
            <el-option label="禁用" :value="0"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item v-if="!isEdit" label="密码" prop="password">
          <el-input v-model="adminForm.password" type="password" placeholder="请输入密码" show-password></el-input>
        </el-form-item>
        <el-form-item v-if="!isEdit" label="确认密码" prop="confirmPassword">
          <el-input v-model="adminForm.confirmPassword" type="password" placeholder="请确认密码" show-password></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" @click="submitForm" :loading="formLoading">确认</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { API_BASE_URL } from '../../../config';

// 状态变量
const loading = ref(false);
const formLoading = ref(false);
const adminList = ref([]);
const total = ref(0);
const pageSize = ref(10);
const currentPage = ref(1);
const dialogVisible = ref(false);
const isEdit = ref(false);
const adminFormRef = ref(null);

// 日期格式化函数
const formatDateTime = (dateTimeStr) => {
  if (!dateTimeStr) return '未记录';
  
  const date = new Date(dateTimeStr);
  if (isNaN(date.getTime())) return dateTimeStr;
  
  return `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
};

// 搜索表单
const searchForm = reactive({
  phoneNumber: '',
  fullName: '',
  role: '',
  status: ''
});

// 管理员表单
const adminForm = reactive({
  admin_id: '',
  phone_number: '',
  full_name: '',
  email: '',
  role: 'operator',
  status: 1,
  password: '',
  confirmPassword: ''
});

// 表单验证规则
const adminRules = reactive({
  phone_number: [
    { required: true, message: '请输入手机号', trigger: 'blur' },
    { pattern: new RegExp('^1[3-9]\\d{9}$'), message: '请输入有效的手机号', trigger: 'blur' }
  ],
  full_name: [
    { required: true, message: '请输入姓名', trigger: 'blur' },
    { min: 2, max: 20, message: '姓名长度应在2-20个字符之间', trigger: 'blur' }
  ],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '请输入有效的邮箱地址', trigger: 'blur' }
  ],
  role: [
    { required: true, message: '请选择角色', trigger: 'change' }
  ],
  status: [
    { required: true, message: '请选择状态', trigger: 'change' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度至少为6个字符', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    {
      validator: (rule, value, callback) => {
        if (value !== adminForm.password) {
          callback(new Error('两次输入的密码不一致'));
        } else {
          callback();
        }
      },
      trigger: 'blur'
    }
  ]
});

// 获取管理员列表
const fetchAdminList = async () => {
  loading.value = true;
  try {
    // 构建查询参数
    const params = new URLSearchParams();
    params.append('page', currentPage.value);
    params.append('page_size', pageSize.value);
    
    if (searchForm.phoneNumber) {
      params.append('phone_number', searchForm.phoneNumber);
    }
    
    if (searchForm.fullName) {
      params.append('full_name', searchForm.fullName);
    }
    
    if (searchForm.role) {
      params.append('role', searchForm.role);
    }
    
    if (searchForm.status !== '') {
      params.append('status', searchForm.status);
    }
    
    const response = await fetch(`${API_BASE_URL}/admin/admins?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
      adminList.value = data.data.admins;
      total.value = data.data.total;
    } else {
      ElMessage.error(data.message || '获取管理员列表失败');
    }
  } catch (error) {
    console.error('获取管理员列表错误:', error);
    ElMessage.error('网络连接异常，请稍后再试');
  } finally {
    loading.value = false;
  }
};

// 搜索处理
const handleSearch = () => {
  currentPage.value = 1;
  fetchAdminList();
};

// 重置搜索表单
const resetSearchForm = () => {
  searchForm.phoneNumber = '';
  searchForm.fullName = '';
  searchForm.role = '';
  searchForm.status = '';
  currentPage.value = 1;
  fetchAdminList();
};

// 分页大小改变
const handleSizeChange = (size) => {
  pageSize.value = size;
  fetchAdminList();
};

// 当前页码改变
const handleCurrentChange = (page) => {
  currentPage.value = page;
  fetchAdminList();
};

// 添加管理员
const handleAddAdmin = () => {
  isEdit.value = false;
  resetForm();
  dialogVisible.value = true;
};

// 编辑管理员
const handleEdit = (admin) => {
  isEdit.value = true;
  resetForm();
  Object.assign(adminForm, {
    admin_id: admin.admin_id,
    phone_number: admin.phone_number,
    full_name: admin.full_name,
    email: admin.email,
    role: admin.role,
    status: admin.status
  });
  dialogVisible.value = true;
};

// 重置表单
const resetForm = () => {
  adminForm.admin_id = '';
  adminForm.phone_number = '';
  adminForm.full_name = '';
  adminForm.email = '';
  adminForm.role = 'operator';
  adminForm.status = 1;
  adminForm.password = '';
  adminForm.confirmPassword = '';
  
  // 重置表单验证
  if (adminFormRef.value) {
    adminFormRef.value.resetFields();
  }
};

// 提交表单
const submitForm = () => {
  if (!adminFormRef.value) return;
  
  adminFormRef.value.validate(async (valid) => {
    if (!valid) return;
    
    formLoading.value = true;
    try {
      const url = isEdit.value 
        ? `${API_BASE_URL}/admin/admins/${adminForm.admin_id}` 
        : `${API_BASE_URL}/admin/admins`;
      
      const method = isEdit.value ? 'PUT' : 'POST';
      
      // 移除确认密码字段，不需要发送到后端
      const submitData = { ...adminForm };
      delete submitData.confirmPassword;
      
      // 如果是编辑模式，不发送密码字段
      if (isEdit.value) {
        delete submitData.password;
      }
      
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(submitData)
      });
      
      const data = await response.json();
      
      if (data.code === 200) {
        ElMessage({
          type: 'success',
          message: isEdit.value ? '管理员信息更新成功' : '添加管理员成功',
          duration: 2000,
          showClose: true
        });
        dialogVisible.value = false;
        fetchAdminList();
      } else {
        ElMessage.error(data.message || (isEdit.value ? '更新管理员信息失败' : '添加管理员失败'));
      }
    } catch (error) {
      console.error('提交表单错误:', error);
      ElMessage.error('网络连接异常，请稍后再试');
    } finally {
      formLoading.value = false;
    }
  });
};

// 改变管理员状态
const handleStatusChange = (admin) => {
  if (!admin) return;
  
  const action = admin.status === 1 ? '禁用' : '启用';
  const newStatus = admin.status === 1 ? 0 : 1;
  
  // 自定义确认对话框样式
  ElMessageBox.confirm(
    `确定要${action}管理员 ${admin.full_name} 的账号吗？`,
    `${action}确认`,
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: admin.status === 1 ? 'warning' : 'success',
      customClass: 'custom-message-box',
      distinguishCancelAndClose: true,
      center: true,
      roundButton: true,
      showClose: false,
      draggable: true
    }
  ).then(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/admin/admins/status`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          admin_id: admin.admin_id,
          status: newStatus
        })
      });
      
      const data = await response.json();
      
      if (data.code === 200) {
        ElMessage({
          type: 'success',
          message: `管理员账号${action}成功`,
          duration: 2000,
          showClose: true
        });
        
        // 刷新列表
        fetchAdminList();
      } else {
        ElMessage.error(data.message || `管理员账号${action}失败`);
      }
    } catch (error) {
      console.error(`${action}管理员账号错误:`, error);
      ElMessage.error('网络连接异常，请稍后再试');
    }
  }).catch(() => {
    // 用户取消操作
  });
};

// 页面加载时获取管理员列表
onMounted(() => {
  fetchAdminList();
});
</script>

<style scoped>
.admin-management {
  padding: 20px;
}

.search-container {
  background-color: #fff;
  padding: 20px;
  border-radius: 4px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.table-container {
  background-color: #fff;
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

/* 添加表格单元格文本省略样式 */
.ellipsis-cell {
  display: block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 鼠标悬停时显示完整内容的提示 */
.ellipsis-cell:hover {
  cursor: pointer;
}

.pagination-container {
  margin-top: 20px;
  text-align: right;
}

/* 添加操作按钮间距 */
.el-button + .el-button {
  margin-left: 8px;
}

/* 确保表格在小屏幕上也能水平滚动 */
@media (max-width: 1200px) {
  .table-container {
    overflow-x: auto;
  }
}

/* 移除按钮点击后的黑色边框 */
.el-button:focus {
  outline: none !important;
  box-shadow: none !important;
}

/* 改善按钮点击状态 */
.el-button:active {
  opacity: 0.9;
  transform: scale(0.98);
}

/* 确保表格内的按钮也没有黑框 */
.el-table .el-button:focus {
  outline: none !important;
  box-shadow: none !important;
}

/* 改进表单样式 */
:deep(.el-form-item__label) {
  font-weight: 500;
}

:deep(.el-input__inner) {
  border-radius: 4px;
}

:deep(.el-select .el-input__inner) {
  border-radius: 4px;
}

/* 对话框样式 */
:deep(.el-dialog) {
  border-radius: 8px;
  overflow: hidden;
}

:deep(.el-dialog__header) {
  background-color: #f8f9fa;
  padding: 15px 20px;
  margin: 0;
  border-bottom: 1px solid #ebeef5;
}

:deep(.el-dialog__title) {
  font-weight: bold;
  color: #333;
}

:deep(.el-dialog__body) {
  padding: 20px;
}

:deep(.el-dialog__footer) {
  border-top: 1px solid #ebeef5;
  padding: 15px 20px;
  background-color: #f8f9fa;
}

/* 响应式布局 */
@media (max-width: 1200px) {
  .search-form .el-col {
    width: 100%;
  }
  
  .search-form .el-col:last-child {
    display: flex;
    justify-content: flex-start;
    margin-top: 10px;
  }
  
  .button-container {
    justify-content: flex-start;
    margin-top: 0;
  }
  
  .table-container {
    overflow-x: auto;
  }
}
</style>

<style>
/* 自定义对话框样式 */
.custom-message-box {
  border-radius: 10px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  padding: 0;
  border: none;
}

.custom-message-box .el-message-box__header {
  padding: 16px 20px 8px;
  border-bottom: 1px solid #ebeef5;
  background-color: #f8f9fa;
}

.custom-message-box .el-message-box__title {
  font-size: 18px;
  font-weight: bold;
  display: flex;
  align-items: center;
}

.custom-message-box .el-message-box__content {
  padding: 24px 20px;
  font-size: 15px;
  line-height: 1.6;
  color: #5a5a5a;
}

.custom-message-box .el-message-box__btns {
  padding: 12px 20px;
  border-top: 1px solid #ebeef5;
  background-color: #f8f9fa;
}

.custom-message-box .el-button {
  border-radius: 24px;
  padding: 9px 22px;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.25s ease;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}

.custom-message-box .el-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
}

.custom-message-box .el-button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* 为不同类型确认框定制样式 */
.el-message-box.el-message-box--warning .el-message-box__title {
  color: #e6a23c;
}

.el-message-box.el-message-box--success .el-message-box__title {
  color: #67c23a;
}

.el-message-box__status.el-icon-warning {
  color: #e6a23c;
  font-size: 24px;
  margin-right: 10px;
}

.el-message-box__status.el-icon-success {
  color: #67c23a;
  font-size: 24px;
  margin-right: 10px;
}

/* 去除按钮焦点状态 - 全局应用到所有对话框按钮 */
.el-message-box .el-button:focus,
.el-dialog .el-button:focus,
.el-dialog__wrapper .el-button:focus {
  outline: none !important;
  box-shadow: none !important;
  border-color: transparent;
}

/* 改进禁用按钮样式 */
.custom-message-box .el-button--danger {
  background-color: #f56c6c;
  border-color: #f56c6c;
  color: white;
}

.custom-message-box .el-button--success {
  background-color: #67c23a;
  border-color: #67c23a;
  color: white;
}

/* 改进确认按钮动效 */
.custom-message-box .el-button--primary {
  position: relative;
  overflow: hidden;
}

.custom-message-box .el-button--primary:after {
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
  transition: all 0.5s ease;
}

.custom-message-box .el-button--primary:hover:after {
  left: 100%;
}

/* 标签相关样式 */
.tag-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}

.admin-tag {
  min-width: 70px;
  text-align: center;
  padding: 2px 12px;
  font-weight: 500;
  letter-spacing: 1px;
  border-radius: 12px;
  white-space: nowrap;
  font-size: 13px;
}

:deep(.el-tag--danger.admin-tag) {
  background-color: rgba(245, 108, 108, 0.9);
}

:deep(.el-tag--success.admin-tag) {
  background-color: rgba(103, 194, 58, 0.9);
}
</style> 