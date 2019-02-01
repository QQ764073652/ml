from sqlalchemy import create_engine
import pandas as pd
import requests
from xdl.utils import *

# import MySQLdb
engine = create_engine('mysql+pymysql://root:hadoop1234@112.31.12.175:33306/cnbita?charset=utf8')
# 本地数据存储
local_engine = create_engine('mysql+pymysql://root:P@ssw0rd@127.0.0.1:3306/pai?charset=utf8')
token = ''
domain = "https://cloud.bitahub.com"


# 获取关系型用户数据
def get_user_basic_info_online():
    with engine.connect() as conn:
        df_user_info = pd.read_sql(
            "select id user_id,username,email,company,active_time,is_developer,create_date from ai_user_basic_info",
            conn)
        df_user_info.to_csv("user_basic_info.csv", index=False)
        return df_user_info


def get_user_basic_info_offline():
    df_user_info = pd.read_csv("user_basic_info.csv")
    return df_user_info


# auth
def authentication():
    global token
    # auth
    auth_api = domain + "/rest-server/api/v1/token"
    params = {
        "username": "mengjiaxiang@leinao.ai",
        "password": "K&3hsU7D*t",
        "expiration": "7200"
    }
    auth_response = requests.post(auth_api, data=params).json()
    token = auth_response.get('token', '')
    return token


# save to db
def save_to_db(df, table):
    try:
        with local_engine.connect() as conn:
            # df.apply(transferContent)
            df.to_sql(table, conn, if_exists="append", index=False)
    except Exception as e:
        print(e)


# sql转义
def transferContent(content):
    content = str(content)
    if content is None:
        return None
    else:
        string = ""
        for c in content:
            if c == '"':
                string += '\"'
            elif c == "'":
                string += "\'"
            elif c == "\\":
                string += "\\\\"
            else:
                string += c
        return string


# 获取用户任务列表
def get_job_list_online():
    # get job list
    job_api = domain + "/rest-server/api/v1/jobs"
    header = {
        "Authorization": "Bearer " + token
    }
    job_list_response = requests.get(job_api, headers=header).json()
    df_user_jobs = pd.DataFrame(job_list_response)
    df_user_jobs.to_csv("user_jobs.csv", index=False)
    return df_user_jobs


def get_job_list_offline():
    df_user_jobs = pd.read_csv('user_jobs.csv')
    return df_user_jobs


def map_list(data, prefix, sep='_'):
    for task in data:
        task_dic = {}
        _task = flatten(task)
        for k, v in _task.items():
            task_dic[prefix + sep + k] = transferContent(v)
    return task_dic


# 获取任务配置信息
def get_user_job_info(job_dic):
    job_config_api = domain + "/rest-server/api/v1/jobs/%s/config" % job_dic.get('name', '')
    job_info_api = domain + "/rest-server/api/v1/jobs/%s" % job_dic.get('name', '')
    header = {
        "Authorization": "Bearer " + token
    }
    # job info
    # job_info_response = requests.get(job_info_api, headers=header).json()
    # job_info_response = flatten(job_info_response)
    # job_info_list = []
    # for k,v in job_info_response.copy().items():
    #     if isinstance(v,list):
    #         task_dic = map_list(job_info_response[k],'task')
    #         job_info_response.pop(k)
    #         job_info_dic = {**job_info_response, **task_dic}
    #         job_info_list.append(job_info_dic)
    #
    # save_to_db(pd.DataFrame(job_info_list), 'user_job_info')
    # print(job_info_list)

    # job config
    job_config_respose = requests.get(job_config_api, headers=header).json()
    job_config_respose = flatten(job_config_respose)
    job_config_list = []
    for k, v in job_config_respose.copy().items():
        if isinstance(v, list):
            task_dic = map_list(job_config_respose[k], 'task')
            job_config_respose.pop(k)
            job_info_dic = {**job_config_respose, **task_dic}
            job_config_list.append(job_info_dic)
    save_to_db(pd.DataFrame(job_config_list), 'user_job_config')
    print(job_config_list)


# data merge
def data_merge_by_userid():
    df_user_basic_info = get_user_basic_info_online()
    df_job_list = get_job_list_online()
    df_job_info = get_user_job_info()
    user_data = pd.merge(df_user_basic_info, df_job_list, how='inner', on=['user_id'])


# 一些统计
def static():
    # 统计每个人任务成功、失败等任务数量
    sql = 'select a1.user_id,a1.email,count(*) num,a2.state from user_basic_info a1,user_jobs_list a2 where a1.user_id=a2.userId group by a1.user_id,a1.email,a2.state order by a1.user_id,num desc;'
    # 统计任务耗时情况
    'select (completedTime-createdTime)cost from user_jobs_list order by cost desc'
    # 通过聚合用户任务信息
    'select * from user_jobs_list a1,user_job_config a2 where a1.name = a2.jobName limit 10'
    # 统计镜像使用次数
    'select t1.image,count(*) num from (select image,jobName from user_job_config group by image,jobname )t1 GROUP BY t1.image order by num desc'
    'select image,count(*) num from user_job_config GROUP BY image order by num desc'


# 训练xdl
def train_xdl():
    pass

def multi_task(df):
    df.apply(get_user_job_info)

def multi():
    from multiprocessing import Pool

    df_job_list = get_job_list_online()[15822:]
    process_size = 4  # 进程数量(默认cpu核数)
    pool = Pool(process_size)
    for i in range(process_size):
        # map_async异步执行，io密集型可以用协程，计算密集型用进程
        pool.map_async(multi_task, [df_job_list, 4])
    pool.close()
    pool.join()

if __name__ == '__main__':
    authentication()
    # df_user_basic_info = get_user_basic_info_online()
    # save_to_db(df_user_basic_info,'user_basic_info')
    # df_job_list = get_job_list_online()
    # save_to_db(df_job_list,'user_jobs_list')

    df_job_list = get_job_list_online()
    df_job_list[31000:].T.apply(get_user_job_info)
    # multi()