import pymysql
import pymysql.cursors
from pymysql.cursors import DictCursor
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库连接配置
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'password'),
    'db': os.environ.get('DB_NAME', 'chemical_ai_db'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True
}

def get_connection():
    """
    获取数据库连接
    """
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        raise

def execute_query(query, params=None):
    """
    执行查询操作，并返回结果
    
    Args:
        query (str): SQL查询语句
        params (tuple): 查询参数
        
    Returns:
        list: 查询结果列表
    """
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
        connection.close()
        return result
    except Exception as e:
        logger.error(f"查询执行失败: {str(e)} - 查询: {query} - 参数: {params}")
        raise

def execute_update(query, params=None):
    """
    执行更新操作，包括插入、更新和删除
    
    Args:
        query (str): SQL更新语句
        params (tuple): 更新参数
        
    Returns:
        int: 受影响的行数
    """
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            affected_rows = cursor.execute(query, params)
            if query.strip().upper().startswith('INSERT'):
                # 如果是INSERT操作，返回自增ID
                last_id = cursor.lastrowid
                connection.close()
                return last_id
            connection.close()
            return affected_rows
    except Exception as e:
        logger.error(f"更新执行失败: {str(e)} - 查询: {query} - 参数: {params}")
        raise

def execute_transaction(queries_and_params):
    """
    执行事务操作，保证多个查询的原子性
    
    Args:
        queries_and_params (list): 查询和参数的列表，每个元素为 (query, params) 元组
        
    Returns:
        bool: 事务是否成功
    """
    try:
        connection = get_connection()
        connection.autocommit(False)  # 关闭自动提交
        try:
            with connection.cursor() as cursor:
                results = []
                for query, params in queries_and_params:
                    cursor.execute(query, params)
                    if query.strip().upper().startswith('SELECT'):
                        results.append(cursor.fetchall())
                    elif query.strip().upper().startswith('INSERT'):
                        results.append(cursor.lastrowid)
                    else:
                        results.append(cursor.rowcount)
            connection.commit()
            connection.close()
            return results
        except Exception as e:
            connection.rollback()
            connection.close()
            logger.error(f"事务执行失败，已回滚: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"事务初始化失败: {str(e)}")
        raise 