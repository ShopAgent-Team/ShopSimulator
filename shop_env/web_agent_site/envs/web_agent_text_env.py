import gym
import json
import random
import time
import torch
import numpy as np

from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import defaultdict
from flask import Flask
from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    ACTION_TO_TEMPLATE,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
from web_agent_site.engine.goal import get_reward, get_goals
from web_agent_site.utils import (DEFAULT_FILE_PATH, random_idx)

PROMPT_TEMPLATE_zh="""你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。请注意，商品库中存在大量同类商品，你必须通过合理操作，最终购买到最符合要求的目标商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。

规则说明：
1. 在点击【购买】前，必须至少选择一个商品规格（如颜色、尺码等）。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 如果你给出的操作在当前环境中不可执行，则页面不会有任何变化。
4. 提示：当你点击某一个商品规格时，页面其他信息不会发生变化，但该规格会被选中。请据此判断后续操作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""

PROMPT_TEMPLATE_zh_persona="""你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。
我会提供他们的目标商品（例如“一双鞋”）以及个人文档（例如偏好、预算、使用场景等）请注意，商品库中存在大量同类商品，你必须通过合理操作，，综合分析用户需求和当前页面信息，最终购买到最符合用户要求的商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。

规则说明：
1. 在点击【购买】前，必须至少选择一个商品规格（如颜色、尺码等）。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 如果你给出的操作在当前环境中不可执行，则页面不会有任何变化。
4. 提示：当你点击某一个商品规格时，页面其他信息不会发生变化，但该规格会被选中。请据此判断后续操作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""

app = Flask(__name__)
class WebAgentTextEnv(gym.Env):
    """Gym environment for Text mode of WebShop environment"""
    def __init__(
            self,
            observation_mode='html',
            file_path=DEFAULT_FILE_PATH,
            server=None,
            **kwargs
        ):
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(WebAgentTextEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        self.file_path = file_path

        self.base_url = 'http://127.0.0.1:3000'
        self.server = SimServer(
            self.base_url,
            self.file_path,
            self.kwargs.get('filter_goals'),
            self.kwargs.get('limit_goals', -1),
            self.kwargs.get('num_products'),
            self.kwargs.get('human_goals'),
            self.kwargs.get('show_attrs', False),
            self.kwargs.get('shuffle_goals', False),
            self.kwargs.get('shuffle_num', 20),
            self.kwargs.get('shift_goals', False),
            self.kwargs.get('if_persona', False),
        ) if server is None else server
        self.browser = SimBrowser(self.server)
        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')
        if self.kwargs.get('get_image', 0):
            # TODO: self.ids 需要从外部传入或初始化，否则这里会报错
            # self.ids = {url: idx for idx, url in enumerate(self.ids)}
            pass
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        self.split = self.kwargs.get('split', "test")
        self.if_persona = self.kwargs.get('if_persona', False)
        if self.if_persona:
            self.prompt_template = PROMPT_TEMPLATE_zh_persona
        else:
            self.prompt_template = PROMPT_TEMPLATE_zh
        self.history_init = [
            {'role': 'user', 'content': self.prompt_template},
            {'role': 'assistant', 'content': 'ok'},
        ]
        if self.split == 'test':
            self.init_idx = -1
        else:
            self.init_idx = 194
        self.idx = self.init_idx

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        info = None
        self.get_available_actions()

        # Determine action type (click, search) and argument
        try:
            action_name, action_arg = parse_action(action)
        except (ValueError, TypeError, AttributeError):
            # 更具体的异常处理，避免捕获所有异常
            action_name, action_arg = "", ""
        if action_arg is not None:
            action_arg = action_arg.lower()
        if (action_name == 'search' and
            action_arg is not None and
            action_arg != ''):
            #执行搜索
            status = self.browser.search(action_arg)
        elif (action_name == 'click' and
              action_arg in self.text_to_clickable.keys() and
              action_arg != 'search'):
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:
            status = dict(reward=0, done=False)

        # Update observation, state with the new action
        ob = self.observation
        text_list = [ob]
        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text_list.append(self.prev_obs[-i])
        state = ' [SEP] '.join(text_list[::-1])
        self.prev_obs.append(ob)

        #return state, status['reward'], status['done'], info
        return state, status, info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links  = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f'{b.get_text()}'.lower(): b
            for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get('value')
            self.text_to_clickable[f'{opt_value}'] = opt
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def get_image(self):
        """Scrape image from page HTML and return as a list of pixel values"""
        html_obj = self._parse_html(self.browser.page_source)
        image_url = html_obj.find(id='product-image')
        if image_url is not None:
            image_url = image_url['src']
            if image_url in self.ids:
                image_idx = self.ids[image_url]
                image = self.feats[image_idx]
                return image
        return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            for t in visible_texts:
                if t == '\n':
                    continue
                if t.parent.name == 'button':  # button
                    processed_t = f'[button] {t} [button_]'
                elif t.parent.name == 'label':  # options
                    if f'"{t}"' in self.state['url']:
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'  [button] {t} [button_]'
                elif t.parent.get('class') == ["product-link"]: # product asins
                    if f'{t}' in self.server.user_sessions[self.session]['asins']:
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    else:
                        processed_t = f'\n[button] {t} [button_]'
                else: # regular, unclickable text
                    processed_t =  str(t)
                observation += processed_t + '\n'
            return observation

    def reset(self,idx=None,session=None, instruction_text=None):
        """Create a new session and reset environment variables"""
        session_int = None
    
        if session is not None:
            self.session = str(session)
        else:
            self.session = idx

        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        self.text_to_clickable = None
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []
        self.history = self.history_init[:]
        self.instruction_simple = self.server.goals[idx]['instruction_simple']
        self.goal_options = self.server.goals[idx]['goal_options']
        if self.if_persona:
            user_persona = self.server.goals[idx]['user_persona'].copy()
            if '__reasoning__' in user_persona:
                del user_persona['__reasoning__']
            self.history.extend([
                {'role': 'user', 'content': f"\n用户个人文档：{json.dumps(user_persona, ensure_ascii=False)}"},
                {'role': 'assistant', 'content': 'ok'},
            ])
        if self.server.goals[idx]['user_persona']:
            self.user_persona = self.server.goals[idx]['user_persona'].copy()
            self.reason_key = self.server.goals[idx]['reason_key']
        return obs, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

class SimServer:
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""
    def __init__(
        self,
        base_url,
        file_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
        shuffle_goals=False,
        shuffle_num=20,
        shift_goals=False,
        if_persona=False,
    ):
        """
        Constructor for simulated server serving WebShop application

        Arguments:
        filter_goals (`func`) -- Select specific goal(s) for consideration based on criteria of custom function
        limit_goals (`int`) -- Limit to number of goals available
        num_products (`int`) -- Number of products to search across
        human_goals (`bool`) -- If true, load human goals; otherwise, load synthetic goals
        """
        # Load all products, goals, and search engine
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices, _ = \
            load_products(filepath=file_path, num_products=num_products, human_goals=human_goals)
        self.search_engine = init_search_engine(num_products=num_products)
        self.existed_goals = True
        self.goals = get_goals(self.all_products, self.product_prices, if_persona=if_persona)
        self.show_attrs = show_attrs
        self.shuffle_goals = shuffle_goals
        self.shuffle_num = shuffle_num
        self.shift_goals = shift_goals
        # Fix outcome for random shuffling of goals
        #random.seed(233)
        #random.shuffle(self.goals)

        # Apply `filter_goals` parameter if exists to select speific goal(s)
        if filter_goals is not None:
            self.goals = [
                goal for (i, goal) in enumerate(self.goals)
                if filter_goals(i, goal)
            ]

        # Imposes `limit` on goals via random selection
        if limit_goals != -1 and limit_goals < len(self.goals):
            self.weights = [goal['weight'] for goal in self.goals]
            self.cum_weights = [0] + np.cumsum(self.weights).tolist()
            idxs = []
            while len(idxs) < limit_goals:
                idx = random_idx(self.cum_weights)
                if idx not in idxs:
                    idxs.append(idx)
            self.goals = [self.goals[i] for i in idxs]
        print(f'Loaded {len(self.goals)} goals.')
        # Set extraneous housekeeping variables
        self.weights = [goal['weight'] for goal in self.goals]
        #累积权重
        self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        self.user_sessions = dict()
        self.search_time = 0
        self.render_time = 0
        self.sample_time = 0
        self.assigned_instruction_text = None  # TODO: very hacky, should remove

    @app.route('/', methods=['GET', 'POST'])
    def index(self, session_id, **kwargs):
        """Redirect to the search page with the given session ID"""
        html = map_action_to_html(
            'start',
            session_id=session_id,
            instruction_text=kwargs['instruction_text'],
        )
        url = f'{self.base_url}/{session_id}'
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def search_results(self, session_id, **kwargs):
        """Initialize session and return the search results page"""
        session = self.user_sessions[session_id]
        keywords = kwargs['keywords']  # TODO: why is this using kwargs? why not session?
        assert isinstance(keywords, list)
        page = 1 if 'page' not in kwargs else kwargs['page']
        session["page"] = page
        session["keywords"] = keywords
        session["actions"]["search"] += 1
        session["asin"] = None
        session["options"] = {}

        # Perform search on keywords from items and record amount of time it takes
        old_time = time.time()
        top_n_products = get_top_n_product_from_keywords(
            keywords,
            self.search_engine,
            self.all_products,
            self.product_item_dict,
        )
        self.search_time += time.time() - old_time

        if self.shuffle_goals and self.shuffle_num != 0:
            if len(top_n_products) > self.shuffle_num:
                first = top_n_products[:self.shuffle_num]
                random.shuffle(first)
                top_n_products = first + top_n_products[self.shuffle_num:]
            else:
                random.shuffle(top_n_products)
        if self.shift_goals:
            if len(top_n_products) > 40:
                # 取出前20和21-40
                first_20 = top_n_products[:20]
                second_20 = top_n_products[20:40]
                rest = top_n_products[40:]
                # 交换
                top_n_products = second_20 + first_20 + rest

        # Get product list from search result asins and get list of corresponding URLs
        products = get_product_per_page(top_n_products, page)

        keywords_url_string = '+'.join(keywords)
        url = (
            f'{self.base_url}/search_results/{session_id}/'
            f'{keywords_url_string}/{page}'
        )

        # Render HTML search page and record amount of time taken
        old_time = time.time()
        html = map_action_to_html(
            'search',
            session_id=session_id,
            products=products,
            keywords=session["keywords"],
            page=page,
            total=len(top_n_products),
            instruction_text=session["goal"]["instruction_text"],
        )
        self.render_time += time.time() - old_time
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_page(self, session_id, **kwargs):
        """Render and return the HTML for a product item page"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        text_to_clickable = kwargs['text_to_clickable']
        clickable = text_to_clickable[clickable_name]

        # Update session logs with information of last product asin selected
        if (clickable.get('class') is not None and
            clickable.get('class')[0] == 'product-link'):
            session["asin"] = clickable_name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif clickable.get('name') is not None:
            clickable_key = clickable['name'].lower()
            session["options"][clickable_key] = clickable_name
            session["actions"]["options"] += 1

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        keywords_url_string = '+'.join(session["keywords"])
        option_string = json.dumps(session['options'])

        # 新增：获取当前选中的option和价格
        selected_option = None
        selected_price = product_info.get('Price', 0)
        option_to_price = product_info.get('option_to_price', {})
        if session["options"]:
            # 假设只有一个option
            selected_option = list(session["options"].values())[-1]
            selected_price = option_to_price.get(selected_option, selected_price)

        url = (
            f'{self.base_url}/item_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/'
            f'{session["page"]}/{option_string}'
        )

        html = map_action_to_html(
            'click',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
            show_attrs=self.show_attrs,
            selected_option=selected_option,  # 新增
            selected_price=selected_price,    # 新增
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_sub_page(self, session_id, **kwargs):
        """Render and return the HTML for a product's sub page (i.e. description, features)"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        session["actions"][clickable_name] += 1
        keywords_url_string = '+'.join(session["keywords"])
        url = (
            f'{self.base_url}/item_sub_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/{session["page"]}/'
            f'{clickable_name}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{clickable_name}]',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def done(self, session_id, **kwargs):
        """Render and return HTML for done page"""
        session = self.user_sessions[session_id]
        goal = self.user_sessions[session_id]['goal']
        purchased_product = self.product_item_dict[session["asin"]]
        session["actions"]["purchase"] += 1
        price = self.product_prices.get(session["asin"])

        # Calculate reward for selected product and set variables for page details
        reward, info = get_reward(
            purchased_product,
            goal,
            price=price,
            options=session["options"],
            verbose=True
        )

        self.user_sessions[session_id]['verbose_info'] = info
        self.user_sessions[session_id]['done'] = True
        self.user_sessions[session_id]['reward'] = reward

        url = (
            f'{self.base_url}/done/{session_id}/'
            f'{session["asin"]}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{END_BUTTON}]',
            session_id=session_id,
            reward=reward,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url, reward, info, purchased_product, goal, session["options"]

    def receive(self, session_id, current_url, session_int=None, **kwargs):
        """Map action to the corresponding page"""
        status = dict(reward=0.0, done=False)
        with app.app_context(), app.test_request_context():
            # Create/determine goal, instruction_text from current session
            if session_id not in self.user_sessions:
                idx = int(session_id)
                goal = self.goals[idx]
                instruction_text = goal['instruction_text']
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            else:
                instruction_text = \
                    self.user_sessions[session_id]['goal']['instruction_text']
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text
                self.user_sessions[session_id]['goal']['instruction_text'] = instruction_text
            session = self.user_sessions[session_id]

            if not kwargs:
                # If no action, reset the session variables
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update(
                    {
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'options': dict(),
                        'actions': defaultdict(int)
                    }
                )
            elif 'keywords' in kwargs:
                # If search keywords are available, run a search
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    # If "buy now" clicked, calculate reward and flag session as terminated
                    html, url, reward, reward_detail, purchased_product, goal, options = self.done(session_id, **kwargs)
                    status['reward'] = reward
                    status['reward_detail'] = reward_detail
                    status['done'] = True
                    status['purchase'] = self.get_purchase_info(purchased_product, options)
                    status['goal'] = goal
                elif clickable_name == BACK_TO_SEARCH.lower():
                    # If "back to search" clicked, recursively reset the session back to search page
                    html, url, status = self.receive(session_id, current_url)
                elif (clickable_name == NEXT_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "next page" clicked from search results, re-render with `page` enumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] + 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "prev page" clicked from search results, re-render with `page` denumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] - 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_sub_page'):
                    # If "prev page" clicked from sub page, return to corresponding item page
                    html, url = self.item_page(session_id, **kwargs)
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_page'):
                    # If "prev page" clicked from item page, return to search results page
                    html, url = self.search_results(
                        session_id,
                        keywords=session["keywords"],
                        page=session["page"],
                        **kwargs
                    )
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    # Render item_sub_page if clickable is description, features, or reviews
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    # Otherwise, render current item page
                    html, url = self.item_page(session_id, **kwargs)
            return html, url, status

    def get_purchase_info(self, purchase, option):
        purchase_light = {"asin": purchase["asin"],
                          "category": purchase["category"],
                          "query": purchase["query"],
                          "name": purchase["title"],
                          "product_category": purchase["product_category"],
                          "instruction_text": purchase["instruction_text"],
                          "attributes": purchase["Attributes"],
                           "price": purchase["price"],
                           "options":option}
        return purchase_light


    def get_page_name(self, url):
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        if url is None:
            return None
        page_names = [
            'search_results',
            'item_page',
            'item_sub_page',
            'done'
        ]
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''  # index page


class SimBrowser:
    """Simulated browser for rendering the HTML source of WebShop environment pages"""
    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(self, url, session_id=None, session_int=None):
        """Set browser variables to corresponding link, page HTML for URL"""
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        self.page_source, _, _ = \
            self.server.receive(self.session_id, self.current_url, session_int=session_int)
        self.current_url = url

    def click(self, clickable_name, text_to_clickable):
        """Wrapper for `receive` handler for performing click action on current page"""
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                clickable_name=clickable_name,
                text_to_clickable=text_to_clickable,
            )
        return status

    def search(self, keywords):
        """Wrapper for `receive` handler for performing search action on current page"""
        if isinstance(keywords, str):
            keywords = keywords.split(' ')
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                keywords=keywords,
        )
        return status
