# Image Classification with Convolutional Autoencoder
  
<p style="text-align: center;">
    <img src="./doc/images/di_uoa.png" alt="UOA">
    <h1>University of Athens</h1>
    <h2>Department of Informatics and Telecomunications</h2>
</p>

<h3>Dionysis Taxiarchis Balaskas - 1115201700094</h3>
<h3>Andreas Giannoutsos - 1115201700021</h3>
<br><br>


<h3>Introduction to our project (info, goals, complexity, speed, results, simplicity, abstractiveness)</h3>
<h4>
Written digits images classification with Convolutional Autoencoders in Keras
</h4>

<h3>How we run the executables</h3>
<h4>
  To run it with google Colab:
</h4>

   [![Click here to open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AGiannoutsos/Image_Classification_with_Convolutional_Autoencoder/blob/main/experiments.ipynb)

<h3>How we tested that it works</h3>
 <h4>
    We tested our (.py) scripts with every possible combination, based on the project requests. For better performance we used google colab which provides powerfull hardware with memory and GPU. Google Colab minimized the time that each experiment took. We tested many hyperparameters' combinations and found some very good models with good performance and small loss. These two were the creterions which we used to consider which experiment was the best.
 </h4> 

<h3>Project Directories and Files organization</h3><br>
<h4>
  
  Main Directory:
    configuration/<br>        # Directory with hyperparameters configurations, saved on JSON form.<br><br>
    data/<br>                 # Directory with data files.<br><br>
    models/<br>               # Directory where models are saved.<br><br>
    autoencoder.py<br>        # Autoencoder script.<br><br>
    classification.py<br>     # Classifier script.<br><br>
    model.py<br>              # Contains functions that are used for the Neural Network creation, train and test.<br><br>
    test.py<br>               # Creates the configuration files.<br><br>
    visualization.py<br>      # Contains functions that are used on the visualization of the Neural Network results and predictions.<br><br>
    experiments.ipynb<br>     # The python notebook that we run on colab.<br><br>
    
</h4>

<h3>Experiments Details</h3>
<h4>
  
  The experiments have all been tested on the notebook as they have their relevant report and their results there with the graphs.
  For the experiments we have followed the following series:

  First we try different models for the autoencoder
  3 different architectures are tested,

  Small 32, 64
  
  Medium 32, 64, 128
  
  Large 32, 64, 128, 256
  

  These architectures differ in the size of the filters but also in the number of their layers.

  For each architecture we try different hyperparameters to reduce overfitting and increase accuracy.
  Then for each architecture all the different models are printed with the different hyperparameters and the best is stored in the list of the best autoencoders.
  Finally, we print compared to the letters of loss for all the best models.

  Then we do research for the best classifier model.
  The classifier consists of an encoder and a dense neural network. To search for the best classifier we use the best pretrained encoders from the previous research for autoencoders and on them we make different combinations of dense model architectures.

  The models tested are:

  Small encoder + Extra Small dense 16, 16
  
  Small encoder + Small dense 64, 32
  
  Small encoder + Large dense 512, 128
  
  Large encoder + Small dense 64, 32
  
  Large encoder + Medium dense 128
  

  And in these the methodology we follow is similar to that in autoencoders.

  Initially for each architecture we try different hyperparameters in order to reduce the overfitting and increase the accuracy and in the end we print comparatively all the graphs in order to choose the best model for each architecture.

  Finally, for the best models, we print graphs again to compare them and for the best models, we print the classification report as well as random images and the category assigned to them by the best classifier.
</h4>

<h3>Modules Details</h3>
For the models we have models.py

Here are the functions 
```
get_Autoencoder (), get_Classifier (), train_Autoencoder (), train_Classifier ()
```
In the get methods we create a dictionary with the values of the hyperparameters of the format:

for autoencoder
```
small_model_info = {"encoder_layers": [["conv", 32, (3,3)],
                                        ["pool", (2,2)],
                                        ["conv", 64, (3,3)],
                                        ["pool", (2,2)]]
                    ,
                    "decoder_layers": [["upSample", (2,2)],
                                        ["conv", 64, (3,3)],
                                        ["upSample", (2,2)],
                                        ["conv", 32, (3,3)]]
                    ,
                    "optimizer": ["adam", 0.001]
                    ,
                    "batch_size": 32
                    ,
                    "epochs": 30
                    }
```

and for the classifier
```
small_model_classifier_info = {"dense_layers": [["dense", 64],
                                                ["dense", 32]
                                ,
                                "encoder_layers": "models / small_model.h5"
                                ,
                                "optimizer": ["adam", 0.001]
                                ,
                                "dense_only_train_epochs": 30
                                ,
                                "full_train_epochs": 10
                                ,
                                "batch_size": 32
                                }
```

Then this model is trained by the train methods respectively.

In the module autencoder.py classifier.py from the input of data from the user this dictionary is created.


For visualizations.py
In this module we have 3 types of functions for visualization:

```autoencoder_visualization()```
Here are the graphs for the autoencoders with the loss for each experiment but also images along with their prediction through the autoencoder.

```classifier_loss_visualization()```
Here the loss, the accuracy, recall, precision, f1, classification_report are printed for each experiment and for each model that has been given in the list. Finally, all errors are printed comparatively

```classifier_prediction_visualization()```
Finally, the classification report for the model we have chosen is printed here, as well as the confusion matrix, as well as images together with their provided label from the model.
  


