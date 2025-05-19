**Demonstrator Proficiency Benchmark - Evaluation Server**

**Overview**

The Demonstrator Proficiency Benchmark is a classification challenge focused on evaluating the proficiency level of demonstrators based on short, multi-view video clips of them performing various actions. Participants must submit model predictions assessing proficiency levels for each video clip.

**Submission Instructions**

**1. File Format**

Participants must submit a JSON file containing their model predictions. The structure of the JSON file should match the following format:

```json
{
  "videos": ["video_1", "video_2", ...],
  "predictions": [p_1, p_2, ...]
}
```

**2. Model Prediction Details**

**videos:** A list of video identifiers corresponding to test set videos.
**predictions:** A list of the predicted proficiency class labels (Novice, Early Expert, Intermediate Expert, Late Expert)

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

**Submit to EvalAI Server:**
Follow the instructions on the [EvalAI challenge page](https://eval.ai/web/challenges/challenge-page/2291/overview) to submit your predictions to the challenge.
