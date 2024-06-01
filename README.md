# detection_ai
Project which help people to recognize generated text.

Code steps:

1) Launch the parser on kaggle and get the files that need to be read. The "examples" folder contains examples to run.
2) Before starting, you need to clear all received files using "processing_files.py". We will delete those that are known to be broken. 
3) Using "processing_text.py", we read all PDFs into one large json using modules from "paper_parse.py" and perform filtering.
4) Using "dim_script.py" we calculate the dimensions of PHdim and MLE. However, generated texts have already been added here, which you need to do yourself using the openAI API. There is an example file in the "examples folder". We save it in separate json files for quick pre-processing.
5) Using "for_diploma_graph.py" and "mle_stats.py", we draw graphs and train models for both methods of calculating the internal dimension.
