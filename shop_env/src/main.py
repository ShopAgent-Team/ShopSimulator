import os
import lifecycle
from web.blueprint.item import item_blueprint
from flask import Flask


lifecycle.register_signal()

online=False
app = Flask(__name__)
root_path = os.path.expanduser("~")

blueprints = [item_blueprint]
for bp in blueprints:
    app.register_blueprint(bp)

@app.route('/hello-world')
def hello():
    env_vars = os.environ
    print('env vars:')
    for key, value in env_vars.items():
        print(f'{key}={value}')
    return "Hello, World!"

@app.route('/status.taobao')
def status_taobao():
    '''
    如果check返回None代表第一次请求，调用online函数，返回online文件内容
    如果check返回online文件内容，说明已经上线
    :return:
    '''
    global online
    result = lifecycle.check()
    if not result and not online:
        lifecycle.online()
        online = True
        return lifecycle.check()
    else:
        if result:
            return result
        else:
            return 'not fount', 404

if __name__ == '__main__':
    app.run(port=8000, host='127.0.0.1')
