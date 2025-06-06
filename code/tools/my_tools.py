from datetime import datetime
import shutil
import os


def copy_code(save_path):
    now = datetime.now()
    current_time = now.strftime("%m%d%H%M")
    dir_path = save_path + f'/code_{current_time}'
    # 删除目录及其内容
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    shutil.copytree('.', dir_path)


