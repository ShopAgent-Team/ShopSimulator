"""
Functions for specifying goals and reward calculations.
"""
import random
import spacy
from collections import defaultdict
from rich import print
from thefuzz import fuzz
from web_agent_site.engine.normalize import normalize_color
import math
import pdb

nlp = spacy.load("zh_core_web_sm")

def get_price_range_above(price, count=4):
    """
    Get the first N price options above the specified price.
    Round up the current price to the nearest multiple of 10, then generate the next N prices.

    Args:
        price: Base price
        count: Number of prices to return

    Returns:
        list: List of prices
    """
    if price <= 100:
        count = 3
    elif price <= 1000:
        count = 10
    elif price <= 5000:
        count = 50
    elif price <= 10000:
        count = 100

    # Round up to the nearest multiple of 10
    base_price = math.ceil(price / 10) * 10

    # Generate the next count prices with step size of 10
    return [base_price + i * 10 for i in range(count)]

def get_goals(all_products, product_prices, if_persona=False):
    return get_existed_goals(all_products, product_prices, if_persona)

def get_existed_goals(all_products, product_prices, if_persona=False):
    goals = []
    cnt_atts = defaultdict(int)
    cnt_1, cnt_2, cnt_3 = 0, 0, 0
    goal_instructions = []
    for item in all_products:
        if 'instructions' not in item:
            cnt_1 += 1
            continue
        asin = item['asin']
        for product in item['instructions']:
            if product['instruction'] in goal_instructions:
                cnt_2 += 1
                #continue
            else:
                goal_instructions.append(product['instruction'])
            attributes = item['instruction_attributes']
            if len(attributes) == 0:
                cnt_3 += 1
                continue

            if product_prices is not None:
                price = product_prices[asin]
                price_range = get_price_range_above(price)
                if len(price_range) >= 2:
                    _, price_upper = sorted(random.sample(price_range, 2))
                else:
                    price_upper = 10000000
            else:
                price_upper = 10000000

            # Process user_persona, place __reasoning__ field in the first position
            if not isinstance(item['user_persona'], dict):
                item['user_persona'] = {}
            user_persona = item['user_persona'].copy()
            reason_key = item['reason_key']
            if user_persona and '__reasoning__' in user_persona:
                reasoning_value = user_persona.pop('__reasoning__')
                # Create a new ordered dictionary with __reasoning__ in the first position
                ordered_persona = {'__reasoning__': reasoning_value}
                ordered_persona.update(user_persona)
                user_persona = ordered_persona

            if if_persona:
                instruction_text = product['instruction_sample']
            else:
                instruction_text = product['instruction']
            goals.append({
                'asin': asin,
                'category': item['category'],
                'query': item['query'],
                'name': item['title'],
                'instruction_text': instruction_text,
                'instruction_simple': product['instruction_simple'],
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': product['instruction_options'],
                'user_persona': user_persona,
                'reason_key': reason_key,
            })
            for att in attributes:
                cnt_atts[att] += 1
            # goals += product_goals
    for goal in goals:
        goal['weight'] = 1
    print('skipped')
    print(cnt_1, cnt_2, cnt_3)
    return goals

def get_type_reward(purchased_product, goal):
    """Determines the type reward - captures whether chosen product is in the same category"""
    query_match = purchased_product['query'] == goal['query']

    # Check number of unique categories that match, ignoring order
    purchased_product_category = [x.strip() for x in purchased_product['category'].split('›')]
    goal_product_category = [x.strip() for x in goal['category'].split('›')]
    category_match = len(set(purchased_product_category) & set(goal_product_category)) >= 2

    # Determine whether types align based on product name similarity
    purchased_type = purchased_product['title']
    desired_type = goal['name']

    purchased_type_parse = nlp(purchased_type)
    desired_type_parse = nlp(desired_type)

    purchased_type_parse = [t.text.lower() for t in purchased_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]
    desired_type_parse = [t.text.lower() for t in desired_type_parse if t.pos_ in ('PNOUN', 'NOUN', 'PROPN')]

    n_intersect_type = len(
        set(purchased_type_parse) & set(desired_type_parse)
    )
    if len(desired_type_parse) == 0:
        title_score = 0.2
    else:
        title_score = n_intersect_type / len(desired_type_parse)

    r_type = 1.0

    # Adjust r_type score based on query, category title matching/scores
    match = query_match or category_match or title_score > 0.2
    if not match:
        r_type = 0.5

    return dict(
        r_type=r_type,
        query_match=query_match,
        category_match=category_match,
        title_score=title_score,
    )

def get_attribute_reward(purchased_product, goal):
    """Determines whether purchased products shares same attributes as goal"""
    purchased_attrs = purchased_product['Attributes']
    goal_attrs = goal['attributes']

    num_attr_matches = 0
    for g_attr in goal_attrs:
        matched = False
        # Check whether goal attribute found in purchased product attribute list
        for p_attr in purchased_attrs:
            score = fuzz.token_set_ratio(p_attr, g_attr)
            if score > 85:
                num_attr_matches += 1
                matched = True
                break
        # If not in purchased attrs, check Title, Bullet Points (Features), Desc
        if (
            not matched and
            (
                g_attr in purchased_product['Title'].lower() or
                g_attr in ' '.join(purchased_product['BulletPoints']).lower() or
                g_attr in purchased_product['Description'].lower()
            )
        ):
            num_attr_matches += 1
            matched = True
    r_attr = num_attr_matches / len(goal_attrs)
    return r_attr, num_attr_matches

def get_option_reward(purchased_options, goal_options):
    """Calculate reward for purchased product's options w.r.t. goal options"""
    def safe_normalize(option):
        if isinstance(option, str):
            return normalize_color(option)
        return str(option)  # Convert non-string options to string

    purchased_options = [safe_normalize(o) for o in purchased_options]
    goal_options = [safe_normalize(o) for o in goal_options]

    # Perform fuzzy matching of each purchased option against each goal option
    num_option_matches = 0
    for g_option in goal_options:
        for p_option in purchased_options:
            score = fuzz.token_set_ratio(p_option, g_option)
            if score > 85:
                num_option_matches += 1
                break
    # Calculate option reward as fraction of goal options hit
    r_option = num_option_matches / len(goal_options) if len(goal_options) > 0 else 1
    return r_option, num_option_matches

def get_reward(purchased_product, goal, price, options, **kwargs):
    """Get cumulative reward score for purchased product and goal"""
    r_type_dict = get_type_reward(purchased_product, goal)
    purchased_product['price'] = price

    r_price = (
        price <= goal['price_upper']
    ) if goal['price_upper'] > 0 else 1

    r_att, num_attr_matches = get_attribute_reward(purchased_product, goal)

    r_option, num_option_matches = get_option_reward(
        list(options.values()),
        goal['goal_options'].items()
        if isinstance(goal['goal_options'], dict)
        else goal['goal_options']
    )

    total_reward = (
        (num_attr_matches + num_option_matches + r_price) \
            / (len(goal['attributes']) + len(goal['goal_options']) + 1)
    )
    total_reward *= r_type_dict['r_type']

    # If verbose flag enabled, store score sub-components into dictionary
    if kwargs.get('verbose', False):
        info =  {
            'query_match': r_type_dict['query_match'],
            'category_match': r_type_dict['category_match'],
            'title_score': r_type_dict['title_score'],
            'num_attr_matches': num_attr_matches,
            'num_option_matches': num_option_matches,
            'r_type': r_type_dict['r_type'],
            'r_att': r_att,
        }
        if r_option is not None:
            info['r_option'] = r_option
        if r_price is not None:
            info['r_price'] = r_price
        return total_reward, info
    return total_reward
