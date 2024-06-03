import argparse
import json
import numpy as np
import tqdm
from collections import defaultdict

def validate_model_predictions(model_predictions, test_annotations):
    # Validate model predictions global file structure:
    assert type(model_predictions) == dict
    for key in ["videos", "ego_model_predictions", "exo_model_predictions"]:
        assert key in model_predictions.keys()

    assert model_predictions["videos"] == test_annotations["videos"]
    
    # Sanity check the test annotation file:
    assert type(test_annotations["gt_annotations"]) == list
    assert type(test_annotations["scenario_names"]) == list
    assert len(test_annotations["gt_annotations"]) == len(test_annotations["videos"])
    assert len(test_annotations["scenario_names"]) == len(test_annotations["videos"])

    # Validate ego and exo model predictions structure:
    assert len(model_predictions["ego_model_predictions"]) == len(test_annotations["videos"])
    for prediction in model_predictions["ego_model_predictions"]:
        assert np.array(prediction).shape == (4,)

    assert len(model_predictions["exo_model_predictions"]) == len(test_annotations["videos"])
    for prediction in model_predictions["exo_model_predictions"]:
        # Validate predictions present for each Exo view:
        assert len(prediction) == 4 
        for tensor in prediction:
            assert np.array(tensor).shape == (4,)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def evaluate_performance(gt_annotations, model_predictions, model_type):
    correct_predictions = 0
    total_predictions = len(gt_annotations)

    for gt_label, ego_pred, exo_preds in zip(
        gt_annotations,
        model_predictions["ego_model_predictions"],
        model_predictions["exo_model_predictions"]
    ):
        if model_type == 'ego':
            pred_logits = ego_pred
        elif model_type == 'exo':
            pred_logits = np.mean([softmax(p) for p in exo_preds], axis=0)
        elif model_type == 'combined':
            combined_preds = [softmax(ego_pred)] + [softmax(p) for p in exo_preds]
            pred_logits = np.mean(combined_preds, axis=0)
        else:
            raise ValueError("Invalid model type specified")

        predicted_class = np.argmax(softmax(pred_logits))
        if predicted_class == gt_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def evaluate_performance_by_scenario(gt_annotations, model_predictions, scenarios, model_type):
    scenario_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})

    for gt_label, ego_pred, exo_preds, scenario in zip(
        gt_annotations["gt_annotations"],
        model_predictions["ego_model_predictions"],
        model_predictions["exo_model_predictions"],
        scenarios
    ):
        if model_type == 'ego':
            pred_logits = ego_pred
        elif model_type == 'exo':
            pred_logits = np.mean([softmax(p) for p in exo_preds], axis=0)
        elif model_type == 'combined':
            combined_preds = [softmax(ego_pred)] + [softmax(p) for p in exo_preds]
            pred_logits = np.mean(combined_preds, axis=0)
        else:
            raise ValueError("Invalid model type specified")

        predicted_class = np.argmax(softmax(pred_logits))
        if predicted_class == gt_label:
            scenario_accuracies[scenario]['correct'] += 1
        scenario_accuracies[scenario]['total'] += 1

    return {scenario: (info['correct'] / info['total']) if info['total'] else 0 for scenario, info in scenario_accuracies.items()}

def evaluate(gt_file, pred_file):
    print("Starting Evaluation.....")

    with open(gt_file, "r") as fp:
        gt_annotations = json.load(fp)
    with open(pred_file, "r") as fp:
        model_predictions = json.load(fp)

    # Validate model predictions:
    validate_model_predictions(model_predictions, gt_annotations)

    ego_accuracy = evaluate_performance(gt_annotations["gt_annotations"], model_predictions, 'ego')
    exo_accuracy = evaluate_performance(gt_annotations["gt_annotations"], model_predictions, 'exo')
    combined_accuracy = evaluate_performance(gt_annotations["gt_annotations"], model_predictions, 'combined')

    ego_scenario_accuracies = evaluate_performance_by_scenario(gt_annotations, model_predictions, gt_annotations["scenario_names"], 'ego')
    exo_scenario_accuracies = evaluate_performance_by_scenario(gt_annotations, model_predictions, gt_annotations["scenario_names"], 'exo')
    combined_scenario_accuracies = evaluate_performance_by_scenario(gt_annotations, model_predictions, gt_annotations["scenario_names"], 'combined')

    print("Evaluation Completed:")
    print(f"Full results:\nEgo Model Accuracy: {ego_accuracy:.3f}\nExo Model Accuracy: {exo_accuracy:.3f}\nCombined Model Accuracy: {combined_accuracy:.3f}")
    print("Per-Scenario Results:")
    for scenario in ego_scenario_accuracies:
        print(f"Scenario: {scenario}")
        print(f"  Ego Model Accuracy: {ego_scenario_accuracies[scenario]:.3f}")
        print(f"  Exo Model Accuracy: {exo_scenario_accuracies[scenario]:.3f}")
        print(f"  Combined Model Accuracy: {combined_scenario_accuracies[scenario]:.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True, type=str)
    parser.add_argument("--pred-file", required=True, type=str)

    args = parser.parse_args()

    evaluate(args.gt_file, args.pred_file)