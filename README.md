# Scene-Segmentation-Using-the-MovieScenes-Dataset

Steps:

1. Use the "Train.py" to train the model with movies data files present in the "file" folder. Make sure the files are present in the same directory of the script.

2. Once thr model is trained run the "Test_results.py" to get the testing results. Make sure to place the testing files in the same directory.

3. "evaluate_sceneseg.py" is utilized to obtained the metric fuctions Mean Average Precision (mAP) and Mean Maximum IoU (mean Miou) along with other metrics.


RESULTS OBTAINED

"model" file contains the trained model with the following specifications,

      epochs = 150, learning_rate = 1e-2, batch_size = 64

The results of the above model with the same set of training data are as shown below,

      Scores: {
          "AP": 0.9002494082490906,
          "mAP": 0.9251553371150955,
          "Precision": 0.9293243804713711,
          "Recall": 0.920260111684911,
          "F1": 0.9217258403263986
      }

