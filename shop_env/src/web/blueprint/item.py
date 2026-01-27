from flask import Blueprint, jsonify
import os

# 需要将 item_blueprint 注册到 server.py 中
item_blueprint = Blueprint('item', __name__)


@item_blueprint.route('/api/items', methods=['GET'])
def get_items():
    '''
    获取所有 items
    :return:
    '''
    return jsonify({"items": ["item1", "item2", "item3"]})


@item_blueprint.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    '''
    获取一个 item
    :param item_id:
    :return:
    '''
    # 打印环境变量
    print(os.environ)
    return jsonify({"item_id": item_id, "name": f"item{item_id}"})
