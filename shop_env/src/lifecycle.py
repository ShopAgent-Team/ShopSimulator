import signal
import os

root_path = os.path.expanduser("~")

def offline(signum, frame):
    print(f"Received signal: {signum}, frame: {frame}, ready to offline")
    # do something to offline

def check():
    # 返回app.online的内容
    online_file = f'{root_path}/app.online'
    if os.path.exists(online_file):
        with open(online_file, 'r') as f:
            return f.read()
    else:
        return None

def online():
    # 检查~/app.online是否存在
    online_file = f'{root_path}/app.online'
    if not os.path.exists(online_file):
        print('generate online file:', online_file)
        os.system('touch ' + online_file)
        os.system('echo success > ' + online_file)
    print('online')

def register_signal():
    signal.signal(signal.SIGTERM, offline)
    signal.signal(signal.SIGINT, offline)
    signal.signal(signal.SIGQUIT, offline)