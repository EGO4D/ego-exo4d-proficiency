**Demonstrator Proficiency Benchmark - Evaluation Server**

**Overview**

The Demonstrator Proficiency Benchmark is a classification challenge focused on evaluating the proficiency level of demonstrators based on short, multi-view video clips of them performing various actions. Participants must submit model predictions assessing proficiency levels for each video clip.

**Submission Instructions**

**1. File Format**

Participants must submit a JSON file containing their model predictions. The structure of the JSON file should match the following format:

```json
{
  "videos": ["video_1", "video_2", ...],
  "ego_model_predictions": [[p1, p2, p3, p4], ...],
  "exo_model_predictions": [
    [[p1, p2, p3, p4], [p1, p2, p3, p4], [p1, p2, p3, p4], [p1, p2, p3, p4]],
    ...
  ]
}
```

**2. Model Prediction Details**

**videos:** A list of video identifiers corresponding to test set videos.
**ego_model_predictions:** A list of 4-dimensional prediction logits for each video from the ego model.
**exo_model_predictions:** A nested list where each video has 4 sets of 4-dimensional prediction logits, corresponding to the four exo views.

**3. Submission Process**

**Prepare your submission:**
Generate your model predictions and save them in the required JSON format.
Ensure that the predictions align with the test dataset structure.

**Validate your submission:**
Use the provided evaluation script to check the validity of your JSON file before submitting.

Run:
```
python evaluate.py --gt-file path/to/ground_truth.json --pred-file path/to/your_predictions.json
```
