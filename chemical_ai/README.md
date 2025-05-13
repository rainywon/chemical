# 天工AI智能助手

## 项目概述

天工AI智能助手是一个基于Vue 3和FastAPI构建的化工安全智能问答平台。该平台整合了大型语言模型和专业知识库，为用户提供化工安全领域的智能问答服务，帮助用户快速获取和理解化工安全相关知识。

## 功能特点

- **智能问答**：支持大模型生成，提供化工安全领域专业问题的解答
- **历史对话**：保存并管理用户的历史对话记录，支持会话查看与管理
- **资料库**：提供化工安全相关的文档资料查阅，包括MSDS和安全操作规程
- **用户系统**：包含注册、登录功能，支持短信验证码登录
- **实时响应**：采用流式响应技术，实现打字机效果的回答生成
- **优雅的UI**：现代化设计的用户界面，提供良好的用户体验与深色模式支持
- **用户反馈**：支持用户提交系统功能和AI内容反馈
- **多端适配**：响应式设计，适配桌面和移动设备
- **后台管理系统**：全面的管理功能，包括用户管理、内容管理、系统监控等

## 技术栈

### 前端
- **前端框架**：Vue 3 + Vite
- **UI组件库**：Element Plus
- **路由管理**：Vue Router
- **HTTP客户端**：Axios
- **代码高亮**：Prismjs
- **日期格式化**：Intl.DateTimeFormat

### 后端
- **Web框架**：FastAPI
- **数据库**：MySQL 8.0+
- **ORM**：PyMySQL
- **认证**：JWT
- **API文档**：Swagger UI, ReDoc
- **日志管理**：Python logging

## 安装与运行

### 环境要求

- Node.js 14.0+
- npm 6.0+
- Python 3.8+
- MySQL 8.0+

### 前端安装步骤

1. 克隆项目到本地:

```bash
git clone https://github.com/yourusername/chemical_ai
cd chemical_ai
```

2. 安装依赖:

```bash
npm install
```

3. 配置环境变量:
创建 `.env` 文件并设置以下变量:
```
VITE_API_BASE_URL=http://localhost:8000
```

4. 启动开发服务器:

```bash
npm run dev
```

5. 构建生产版本:

```bash
npm run build
```

### 后端安装步骤

1. 创建并激活虚拟环境:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

2. 安装Python依赖:
```bash
pip install -r requirements.txt
```

3. 配置数据库:
```bash
mysql -u root -p < chemical_server.sql
```

4. 启动后端服务器:
```bash
python server.py
```

## 项目结构

```
chemical_ai/
├── public/             # 静态资源
├── src/                # 前端源代码
│   ├── assets/         # 资源文件（图片等）
│   ├── components/     # 公共组件
│   │   ├── common/     # 通用组件
│   │   └── admin/      # 管理后台组件
│   │       ├── admins/     # 管理员管理组件
│   │       ├── content/    # 内容管理组件
│   │       ├── feedback/   # 反馈管理组件
│   │           ├── UserFeedback.vue     # 系统功能反馈
│   │           └── AIContentFeedback.vue # AI内容反馈
│   │       ├── monitor/    # 系统监控组件
│   │       ├── settings/   # 系统设置组件
│   │           └── SystemParams.vue     # 系统参数配置
│   │       └── users/      # 用户管理组件
│   ├── router/         # 路由配置
│   │   └── index.js    # 路由定义文件
│   ├── views/          # 页面视图
│   │   ├── Chat.vue    # 聊天界面
│   │   ├── Login.vue   # 登录页面
│   │   ├── Register.vue# 注册页面
│   │   ├── Files.vue   # 资料库页面
│   │   ├── Welcome.vue # 欢迎页面
│   │   └── admin/      # 管理后台视图
│   │       └── AdminLayout.vue  # 管理后台布局
│   ├── App.vue         # 根组件
│   ├── config.js       # 配置文件
│   ├── main.js         # 入口文件
│   └── style.css       # 全局样式
├── server.py           # 后端主应用
├── database.py         # 数据库连接配置
├── chemical_server.sql # 数据库设计
├── requirements.txt    # Python依赖
├── index.html          # HTML 模板
├── vite.config.js      # Vite 配置
└── package.json        # 项目依赖
```

## API文档

启动后端服务器后，可以访问以下API文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

主要API端点包括：

### 用户相关
- 登录: `POST /api/login/`
- 注册: `POST /api/register/`
- 发送验证码: `POST /api/send_sms/`
- 验证验证码: `POST /api/verify_code/`
- 获取用户信息: `GET /api/user/info/`
- 登出: `POST /api/logout/`

### 对话相关
- 聊天会话: `POST/GET /api/conversations/`
- 聊天消息: `POST/GET /api/conversations/{conversation_id}/messages/`

### 反馈相关
- 用户反馈: `POST /api/submit-feedback/`
- 内容反馈: `POST /api/content-feedback/`

### 管理后台
- 用户管理: `GET/PUT /admin/users/`
- 系统配置: `GET/PUT /admin/settings/system-configs`
- 系统版本: `GET/POST/PUT /admin/settings/system-versions`
- 反馈管理: `GET/PUT /admin/feedback/`
- 操作日志: `GET /admin/operation-logs/`

## 使用指南

### 用户端

1. **首页**：访问应用首页可查看系统简介和功能入口
2. **登录/注册**：新用户需先注册账号，支持手机号验证码注册
3. **聊天**：
   - 在聊天界面可以向AI助手提问化工安全相关问题
   - 支持历史对话查看和管理
4. **资料库**：可浏览和下载化工安全相关的文档资料
5. **反馈**：可以提交使用反馈和建议

### 管理后台

管理员可通过登录进入后台管理系统，包含以下功能模块：

1. **系统监控**
   - 对话数据统计：查看平台对话量和质量数据
   - 用户活跃度监控：分析用户活跃情况和使用趋势
   - 操作日志查询：记录和查询系统所有操作日志

2. **用户管理**
   - 用户查询与管理：管理所有注册用户信息，支持启用/禁用操作
   - 用户登录历史：查看用户登录日志

3. **管理员管理**
   - 管理员与角色管理：查看所有管理员账号并进行权限编辑
   - 管理员操作日志：记录管理员的操作历史

4. **内容管理**
   - 知识库分类管理：管理知识库的分类体系
   - 知识文档管理：上传、编辑和删除知识文档
   - 应急处理方案管理：管理安全应急预案

5. **反馈管理**
   - 系统功能反馈：查看用户提交的功能反馈，支持处理状态更新和回复
   - AI内容反馈：查看用户对AI回答的评价和反馈，支持按评分和类型筛选

6. **系统设置**
   - 系统参数配置：管理和调整系统运行参数，如知识库更新日期、数据源数量等
   - 系统版本管理：添加和管理系统版本信息，支持设置当前版本
   - 接口稳定性监控：监测后端API响应时间和状态

## 数据库设计

系统使用MySQL数据库，主要表结构包括：

1. **用户相关**
   - `users`: 用户基本信息
   - `user_tokens`: 用户登录令牌
   - `verification_codes`: 验证码记录

2. **管理员相关**
   - `admins`: 管理员信息
   - `operation_logs`: 操作日志

3. **对话相关**
   - `chat_sessions`: 对话会话
   - `chat_messages`: 对话消息

4. **反馈相关**
   - `user_feedback`: 用户功能反馈
   - `content_feedbacks`: AI内容反馈
   - `feedback_types`: 反馈类型

5. **内容与配置相关**
   - `knowledge_categories`: 知识分类
   - `knowledge_documents`: 知识文档
   - `emergency_plans`: 应急方案
   - `system_configs`: 系统配置
   - `system_versions`: 系统版本
   - `ai_models`: AI模型配置

## 系统参数配置

系统支持以下配置参数的管理：

1. **知识库最后更新日期** (knowledge_base_update_date)：记录知识库的最后更新时间，支持日期选择器
2. **知识库数据源数量** (data_sources_count)：记录系统知识库的数据源总数
3. **系统响应时间限制** (response_time_limit)：设置系统响应的最大时间限制
4. **系统运行状态** (system_status)：标识系统当前的运行状态，可设为正常、维护中或错误

## 贡献指南

欢迎对本项目进行贡献。请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 维护者

- 维护团队 - [email@example.com](mailto:email@example.com)

## 许可证

[MIT License](LICENSE)
