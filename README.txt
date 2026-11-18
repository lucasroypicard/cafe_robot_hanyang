Cafe Cup Picking — Start Here

RUN
1. Extract the outer student.zip and open baseline.ipynb in Jupyter or Colab.
   For Colab, upload student/cafe_project to Drive with dataset.zip still compressed.
   Set PROJECT_DIR in section 0 to that code folder, then run the notebook from the top.
2. The notebook installs packages, prepares data, and evaluates the supplied weights.
   Keep the training call commented for baseline analysis. W&B is optional and off
   by default. Results are saved under SAVE_PATH (runs/ in Drive when using Colab).
3. Copy the notebook to improved.ipynb, set MODEL_NAME="improved", and develop your
   method using all four views. Keep the supplied baseline code and weights unchanged.

DATASET_GUIDE.txt explains the data, annotations, coordinate conventions, and model
interface. You can upload it together with your notebook when asking an LLM for help.
Detailed execution and training instructions are in the notebook.

SUBMIT
Submit report.pdf separately and one code ZIP containing this top-level folder:

submission/
  baseline.ipynb
  improved.ipynb
  weights/
    baseline.pth
    improved.pth        # Example name; use the actual path in submission.json.
  submission.json
  requirements.txt

Copy submission.example.json to submission.json and set:
- max_level: 1, 2, or 3. All lower levels are also evaluated.
- baseline_checkpoint: relative path to the supplied baseline weights.
- improved_checkpoint: relative path to your main checkpoint, or null if none.

Put any additional model/tokenizer files in weights/ and load them relative to the
checkpoint location. Exclude the dataset, common/, robot/, robot_world/, caches,
and personal keys. Each notebook must contain its own inference definitions tagged
cafe.inference, independent of setup/training cells. Inference must run without
downloads, login, or personal Drive access.

Follow the course slides and announcements for report format, visual comparisons,
assessment, citations, and the required LLM Usage Statement.
