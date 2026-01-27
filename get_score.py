"""
Evaluation results statistics script
Used to calculate and save various metrics for model evaluation
"""

import json
import os
from typing import List, Dict, Any


def get_finished_task(out_path: str) -> List[str]:
    """
    Get all completed JSON task files in the specified directory
    
    Args:
        out_path: Output directory path
        
    Returns:
        List of JSON file paths
    """
    if not os.path.exists(out_path):
        print(f"Warning: Path does not exist: {out_path}")
        return []
    
    json_files = []
    try:
        for filename in os.listdir(out_path):
            if filename.endswith(".json"):
                filepath = os.path.join(out_path, filename)
                json_files.append(filepath)
    except PermissionError:
        print(f"Error: Permission denied to access directory: {out_path}")
        return []
    
    return sorted(json_files)


def calculate_metrics(files: List[str]) -> Dict[str, Any]:
    """
    Calculate evaluation metrics
    
    Args:
        files: List of JSON file paths
        
    Returns:
        Dictionary containing various metrics
    """
    done_rate = []
    r_loose = []
    r_types, r_atts, r_options, r_prices = [], [], [], []
    r_hard = []
    r_success = []
    right = []
    
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            r_loose.append(data.get('reward', 0))
            reward_detail = data.get('reward_detail', {})
            if reward_detail:
                done_rate.append(1)
                r_type = reward_detail.get('r_type', 0)
                r_att = reward_detail.get('r_att', 0)
                r_option = reward_detail.get('r_option', 1)
                r_price = reward_detail.get('r_price', 0)
                
                r_types.append(r_type)
                r_atts.append(r_att)
                r_options.append(r_option)
                r_prices.append(r_price)
                
                # Calculate r_hard: product of four r_details
                r_hard.append(r_type * r_att * r_option * r_price)
                
                # Calculate r_success: 1 if all sub-scores are 1, otherwise 0
                if r_type == 1 and r_att == 1 and r_option == 1 and r_price == 1:
                    r_success.append(1)
                else:
                    r_success.append(0)
                
                # Check if the correct product was selected
                purchase = data.get('purchase', {})
                goal = data.get('goal', {})
                if purchase.get('asin') == goal.get('asin'):
                    right.append(1)
                else:
                    right.append(0)
            else:
                done_rate.append(0)
                r_types.append(0)
                r_atts.append(0)
                r_options.append(0)
                r_prices.append(0)
                r_hard.append(0)
                r_success.append(0)
                right.append(0)
                
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Warning: Failed to read file {filepath}: {e}")
            continue
    
    # Calculate averages, avoid division by zero
    total_samples = len(r_loose)
    if total_samples == 0:
        return {
            "样本总量": 0,
            "完成率": 0.0,
            "平均分_r_loose": 0.0,
            "平均分_r_hard": 0.0,
            "平均分_r_success": 0.0,
            "平均分_r_types": 0.0,
            "平均分_r_attribute": 0.0,
            "平均分_r_option": 0.0,
            "平均分_r_price": 0.0,
            "选对商品率": 0.0
        }
    
    metrics = {
        "样本总量": total_samples,
        "完成率": round(sum(done_rate) / len(done_rate), 4) if done_rate else 0.0,
        "平均分_r_loose": round(sum(r_loose) / len(r_loose), 4) if r_loose else 0.0,
        "平均分_r_hard": round(sum(r_hard) / len(r_hard), 4) if r_hard else 0.0,
        "平均分_r_success": round(sum(r_success) / len(r_success), 4) if r_success else 0.0,
        "平均分_r_types": round(sum(r_types) / len(r_types), 4) if r_types else 0.0,
        "平均分_r_attribute": round(sum(r_atts) / len(r_atts), 4) if r_atts else 0.0,
        "平均分_r_option": round(sum(r_options) / len(r_options), 4) if r_options else 0.0,
        "平均分_r_price": round(sum(r_prices) / len(r_prices), 4) if r_prices else 0.0,
        "选对商品率": round(sum(right) / len(right), 4) if right else 0.0
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Print evaluation metrics
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*50)
    print("Evaluation Results Statistics")
    print("="*50)
    print(f"Total Samples: {metrics['样本总量']}")
    print(f"Completion Rate: {metrics['完成率']}")
    print(f"Average Score--r_loose: {metrics['平均分_r_loose']}")
    print(f"Average Score--r_hard: {metrics['平均分_r_hard']}")
    print(f"Average Score--r_success: {metrics['平均分_r_success']}")
    print(f"Average Score--r_types: {metrics['平均分_r_types']}")
    print(f"Average Score--r_attribute: {metrics['平均分_r_attribute']}")
    print(f"Average Score--r_option: {metrics['平均分_r_option']}")
    print(f"Average Score--r_price: {metrics['平均分_r_price']}")
    print(f"Correct Product Rate: {metrics['选对商品率']}")
    print("="*50 + "\n")


def save_metrics(metrics: Dict[str, Any], save_path: str, model: str, eval_type: str) -> None:
    """
    Save evaluation metrics to JSON file
    
    Args:
        metrics: Metrics dictionary
        save_path: Save path
        model: Model name
        eval_type: Evaluation type (standard/persona)
    """
    os.makedirs(save_path, exist_ok=True)
    
    result_data = {
        "model": model,
        "eval_type": eval_type,
        "metrics": metrics
    }
    
    filename = os.path.join(save_path, f"{model}_metrics.json")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {filename}")
    except IOError as e:
        print(f"Error: Failed to save file {filename}: {e}")


if __name__ == '__main__':
    # Configuration parameters
    model = "qwen3-235b-a22b-instruct-2507"
    turn = "single"  # "single" or "multi"
    eval_type = "standard"  # "standard" or "persona"
    
    # Build correct output path: {turn}_eval/outputs/{eval_type}/{model}/
    out_path = os.path.join(f"{turn}_eval", "outputs", eval_type, model)
    
    print(f"Reading evaluation results from: {out_path}")
    
    # Get all completed task files
    files = get_finished_task(out_path)
    
    if not files:
        print(f"Error: No JSON files found, please check path: {out_path}")
        exit(1)
    
    print(f"Found {len(files)} evaluation result files")
    
    # Calculate metrics
    metrics = calculate_metrics(files)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    # Save path: {turn}_eval/outputs/{eval_type}/metrics/
    save_path = os.path.join(f"{turn}_eval", "outputs", eval_type, "metrics")
    save_metrics(metrics, save_path, model, eval_type)
