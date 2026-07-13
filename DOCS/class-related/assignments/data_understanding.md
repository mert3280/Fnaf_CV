Instructions
Submit a professional data understanding report. This is a living document — you will reference it in future weeks when justifying modeling decisions.

1. Data Source & Ingestion Confirm your dataset is live and accessible. Document:

The source, access method (API, scheduled download, scrape), and update frequency
How you have automated ingestion using Airflow — include a brief description of your DAG structure
Any surprises or changes from what you proposed in Week 1
2. Data Profile Describe the shape and composition of your dataset:

Row and column counts, data types, and time range covered
Missing values, duplicates, and anomalies found — and how you handled them
Key statistical summaries for the most important features (mean, range, distribution shape)
Are there corrupted files (e.g., unreadable JPEGs, truncated text files)? What percentage of your data is usable out of the box?
Report the final counts. (e.g., Total rows for tabular; total tokens/vocabulary size for text; total images and spatial dimensions for 2D/3D vision)
3. For Classical ML or Tabular Data:

Preprocessing & Imputation Document your strategy for missing values, outliers, and categorical encoding. Explain why this strategy fits your data (e.g., why median imputation vs. dropping rows).
Exploratory Analysis Document your most meaningful findings from EDA:
At minimum 3 visualizations with written interpretation (not just captions)
Any relationships, trends, or patterns that inform your ML approach
Any findings that caused you to revise your original problem framing or approach
Feature Candidates List the features you plan to use in your model. For each, briefly justify its inclusion and note any transformation needed (encoding, scaling, binning, etc.).
4. For Deep Learning or Unstructured Data:

The Ingestion Pipeline Architecture Detail how data goes from storage to your GPU. If you are using PyTorch Dataset/DataLoader or TensorFlow tf.data, show the code snippet of your pipeline pipeline. How are you handling batching and shuffling?

Data Transformation & Standardization

Text: Explain your tokenization, padding, and truncation strategy. What is your max sequence length, and what percentage of data gets truncated?

2D/3D Vision: Detail your spatial resizing, normalization, or volumetric resampling (e.g., isotropic voxel spacing for 3D).

The "Overfit a Single Batch" Test Before training a real model, you must prove your pipeline works. Take a tiny subset of your data (e.g., 2–5 samples), pass it through your deep learning network, and train it until loss hits ~0. Provide the training curve graph proving your network can successfully overfit this mini-batch.

5. Revised Core Requirements & Schedule Now that your data is validated, revisit and finalize the Core Requirements and Proposed Schedule from your proposal. Update any requirements that changed based on what you found. Requirements must still meet the granular / specific / measurable standard from Week 1.

implementation-plan.md — Your living 5-week implementation plan. Contains a detailed plan of execution to achieve the end goal of the project. Update this file whenever your plan changes — do not rewrite history, add to it.
schedule.md — Maps your finalized Core Requirements to specific weeks with clear milestones.
6. Context Files By end of Week 2, three AI documentation files must be initialized or updated in your repository. These files will be maintained and updated every week for the remainder of the project.

claude.md — Your AI context file. Defines the goal of your project, instructions for how Claude should assist you (tone, scope, constraints), and any best practices you have established for working with it effectively. Think of this as the system prompt for your project's AI collaborator.
ai-usage-log.md — A running log of your human-AI interaction, updated each week. Each entry should cover:
What tasks you used AI assistance for this week
Specific prompts or context definitions that worked well
Any cases where AI output needed correction or required specific guidance
Learning Outcomes
Architect and automate a resilient data and training pipeline
Manage an independent project from proposal to delivery
Deliverables
Submit via GitHub (branch → merge to main, inside a week2/ documentation folder):

data-understanding-report.md — full report covering sections 1–5 above
eda-notebook.ipynb — EDA notebook with all visualizations and analysis
The following files live in the root docs/ folder and are updated in place each week (not duplicated into weekly folders):

claude.md — initialized or updated this week
implementation-plan.md — finalized this week with revised requirements and schedule
ai-usage-log.md — Week 2 entry added
schedule.md — Schedule with core requirements and clear timeline