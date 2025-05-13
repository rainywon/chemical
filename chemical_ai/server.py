from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from routers import routers
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("chemical_ai")

# 创建FastAPI应用
app = FastAPI(
    title="化工安全AI助手系统API",
    description="提供用户认证、聊天、反馈等功能的API接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的域名，生产环境应当限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许的HTTP方法
    allow_headers=["*"],  # 允许的HTTP头
)

# 注册所有路由
for router in routers:
    app.include_router(router, prefix="/api")

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    logger.info("化工安全AI助手系统API服务启动")

# 应用关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("化工安全AI助手系统API服务关闭")

# 运行服务器
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True) 