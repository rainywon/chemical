# 化工安全AI助手系统 - 后端设置指南

本文档提供了设置和运行化工安全AI助手系统后端的详细说明。

## 系统要求

- Python 3.8+
- MySQL 8.0+
- 虚拟环境工具(可选，但推荐)

## 数据库设置

1. 确保您已安装并启动了MySQL服务器

2. 使用以下命令创建数据库和所需表:

```bash
mysql -u root -p < database_design.sql
```

或者在MySQL客户端中直接运行`database_design.sql`文件中的SQL语句。

## 安装依赖

1. 创建并激活虚拟环境(可选):

```bash
# 创建虚拟环境
python -m venv venv

# 在Windows上激活
venv\Scripts\activate

# 在Linux/macOS上激活
source venv/bin/activate
```

2. 安装所需的Python包:

```bash
pip install -r requirements.txt
```

## 配置

编辑`database.py`文件中的数据库连接配置，根据您的MySQL设置修改以下参数:

```python
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'your_password'),
    'db': os.environ.get('DB_NAME', 'chemical_ai_db'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True
}
```

您可以直接修改默认值，或者通过环境变量设置这些参数。

## 运行服务器

使用以下命令启动FastAPI服务器:

```bash
python server.py
```

服务器将在`http://localhost:8000`上运行，并开启自动重载功能。

## API文档

启动服务器后，您可以访问自动生成的API文档:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 目录结构

- `server.py` - 主应用入口点
- `database.py` - 数据库连接和操作工具
- `database_design.sql` - 数据库设计和初始化脚本
- `routers/` - API路由模块
  - `__init__.py` - 路由注册
  - `login.py` - 用户登录认证
  - `register.py` - 用户注册
  - `sms.py` - 验证码服务
  - `submit_feedback.py` - 用户反馈
  - `chat_history.py` - 聊天历史记录

## 与前端集成

前端应该使用以下格式的API端点:

- 登录: `POST /api/login/`
- 注册: `POST /api/register/`
- 发送验证码: `POST /api/send_sms/`
- 验证验证码: `POST /api/verify_code/`
- 获取用户信息: `GET /api/user/info/`
- 登出: `POST /api/logout/`
- 反馈: `POST /api/submit-feedback/`
- 聊天会话: `POST/GET /api/conversations/`
- 聊天消息: `POST/GET /api/conversations/{conversation_id}/messages/` 