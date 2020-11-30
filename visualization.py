import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# print autencoder summary layers
def print_model_info_autoencoder(model_info):

    num_of_layers_encoder = len(model_info["encoder_layers"])
    num_of_layers_decoder = len(model_info["decoder_layers"])
    encoder_layers = model_info["encoder_layers"]
    decoder_layers = model_info["decoder_layers"]

    encoder = "Encoder("
    decoder = "Decoder("

    # print encoder layers
    for layer in range(num_of_layers_encoder):
        encoder+="->"
        # choose the model layer
        if (encoder_layers[layer][0] == "conv"):
            encoder+="conv"+str(encoder_layers[layer][1])
        elif (encoder_layers[layer][0] == "pool"):
            encoder+="pool"
        elif (encoder_layers[layer][0] == "batchNorm"):
            encoder+="bNorm"
        elif (encoder_layers[layer][0] == "drop"):
            encoder+="drop"
    encoder+=")"

    # print decoder layers
    for layer in range(num_of_layers_decoder):
        decoder+="->"
        # choose the model layer
        if (decoder_layers[layer][0] == "conv"):
            decoder+="conv"+str(decoder_layers[layer][1])
        elif (decoder_layers[layer][0] == "upSample"):
            decoder+="upSa"
        elif (decoder_layers[layer][0] == "batchNorm"):
            decoder+="bNorm"
        elif (decoder_layers[layer][0] == "drop"):
            decoder+="drop"
    decoder+=")"
    
    hyper_parameters = "Optim: [%s] lr: %s batch_s: [%d] epochs: [%d]"%(model_info["optimizer"][0], model_info["optimizer"][1:], model_info["batch_size"], model_info["epochs"])

    return encoder+"\n"+decoder+"\n"+hyper_parameters

# print classifier summary
def print_model_info_classifier(model_info):

    num_of_layers_classifier = len(model_info["dense_layers"])
    classifier_layers = model_info["dense_layers"]
    encoder = model_info["encoder_layers"]
    
    classifier = "Classifier("+encoder

    # print layers
    for layer in range(num_of_layers_classifier):
        classifier+="->"
        # choose the model layer
        if (classifier_layers[layer][0] == "dense"):
            classifier+="dense"+str(classifier_layers[layer][1])
        elif (classifier_layers[layer][0] == "batchNorm"):
            classifier+="bNorm"
        elif (classifier_layers[layer][0] == "drop"):
            classifier+="drop"
    classifier+=")"

    
    hyper_parameters = "Optim: [%s] lr: %s batch_s: [%d] epochs_dense: [%d] epochs_all: [%d]"%(model_info["optimizer"][0], str(model_info["optimizer"][1:]), model_info["batch_size"], model_info["dense_only_train_epochs"], model_info["full_train_epochs"])

    return classifier+"\n"+hyper_parameters

# plot the loss for multiple models + some predicted data
def autoencoder_visualization(histories, train_data, num_of_test_images=4, fontsize=15):

    # set plot surface
    num_of_test_images+=1
    num_of_histories = len(histories)
    # if num_of_histories > 1:
    num_of_histories += 1
    num_of_train_data = train_data.shape[0]
    x_dim = train_data.shape[1]
    y_dim = train_data.shape[2]
    fig = plt.figure(figsize=(20,11*num_of_histories))
    # fig = plt.figure(figsize=(10,4*num_of_histories))
    fig.suptitle("Visualization of Loss with random True and their Predicted images for Every Experiment", fontsize=fontsize+5)

    gs  = gridspec.GridSpec(num_of_test_images*num_of_histories, 3, width_ratios=[0.66, 0.165, 0.165], height_ratios=np.ones(num_of_test_images*num_of_histories))
    ax0 = [plt.subplot(gs[i*num_of_test_images:i*num_of_test_images+num_of_test_images-1, 0]) for i in range(num_of_histories-1)]
    ax_true = [plt.subplot(gs[i,1]) for i in range(num_of_test_images*(num_of_histories-1))]
    ax_pred = [plt.subplot(gs[i,2]) for i in range(num_of_test_images*(num_of_histories-1))]


    # plot loss and validation loss
    for history in range(num_of_histories-1):
        ax0[history].plot(histories[history].history['loss'], label ='loss')
        ax0[history].plot(histories[history].history['val_loss'], label='val loss')
        ax0[history].set_xlabel('Epoch', fontsize=fontsize)
        ax0[history].set_ylabel('Loss', fontsize=fontsize)
        ax0[history].legend(fontsize=fontsize)
        ax0[history].grid(True)
        for val in (histories[history].history['loss'], histories[history].history['val_loss'])[1:]:
            ax0[history].annotate('%0.5f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
        # get model info
        model_info = print_model_info_autoencoder(histories[history].history['model_info'])
        ax0[history].set_title("Experiment: "+str(history+1)+"\n"+model_info+"\nModel loss", fontsize=fontsize)

        # show true random images
        random_test_indexes = np.random.randint(0, num_of_train_data-1, num_of_test_images)
        for ax in range(num_of_test_images-1):
            ax_true[ax+history*num_of_test_images].imshow(train_data[random_test_indexes[ax]].reshape(x_dim,y_dim))
            ax_true[ax+history*num_of_test_images].set_title("True image", fontsize=fontsize)
            ax_true[ax+history*num_of_test_images].axis("off")
        ax_true[num_of_test_images-1+history*num_of_test_images].axis("off")

        # get the predictions
        prediction = histories[history].model.predict(train_data[random_test_indexes,:,:,:])
        for ax in range(num_of_test_images-1):
            ax_pred[ax+history*num_of_test_images].imshow(prediction[ax].reshape(x_dim,y_dim))
            ax_pred[ax+history*num_of_test_images].set_title("Predicted image", fontsize=fontsize)
            ax_pred[ax+history*num_of_test_images].axis("off")
        ax_pred[num_of_test_images-1+history*num_of_test_images].axis("off")

    ### plot all together ###
    if num_of_histories > 2:
        ax = plt.subplot(gs[(num_of_histories-1)*num_of_test_images:, 0])

        for history in range(num_of_histories-1):
            ax.plot(histories[history].history['val_loss'], label="experiment "+str(history+1))
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Epoch', fontsize=fontsize)
        ax.set_ylabel('Loss', fontsize=fontsize)
        ax.set_title("Models' losses", fontsize=fontsize)

        # scatter experimets
        ax = plt.subplot(gs[(num_of_histories-1)*num_of_test_images:, 1:])

        for history in range(num_of_histories-1):
            ax.scatter(history, histories[history].history['loss'][-1], label="exp "+str(history+1)+" loss", marker="o")
            ax.scatter(history, histories[history].history['val_loss'][-1], label="exp "+str(history+1)+" val loss", marker="X")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Experiments', fontsize=fontsize)
        ax.set_ylabel('Loss', fontsize=fontsize)
        ax.set_xticks(range(num_of_histories-1))
        ax.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories-1)])
        ax.set_title("Models' losses per hyperparameters", fontsize=fontsize)


    # plt.close()
    # _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show(block=True)
    return fig

def autoencoder_visualization_window(histories, train_data, num_of_test_images=4, fontsize=6):

    # set plot surface
    num_of_test_images+=1
    num_of_histories = len(histories)
    # if num_of_histories > 1:
    # num_of_histories += 1
    num_of_train_data = train_data.shape[0]
    x_dim = train_data.shape[1]
    y_dim = train_data.shape[2]
    fig = plt.figure(figsize=(8,4*num_of_histories))
    # fig = plt.figure(figsize=(10,4*num_of_histories))
    # fig.suptitle("Visualization of Loss with random True and their Predicted images for Every Experiment", fontsize=fontsize+2)

    gs  = gridspec.GridSpec(num_of_test_images*num_of_histories, 3, width_ratios=[0.66, 0.165, 0.165], height_ratios=np.ones(num_of_test_images*num_of_histories))
    ax0 = [plt.subplot(gs[(i)*(num_of_test_images)+1:i*num_of_test_images+num_of_test_images, 0]) for i in range(num_of_histories)]
    ax_true = [plt.subplot(gs[i,1]) for i in range(num_of_test_images*(num_of_histories))]
    ax_pred = [plt.subplot(gs[i,2]) for i in range(num_of_test_images*(num_of_histories))]
    ax_title = [plt.subplot(gs[i*num_of_test_images, :]) for i in range(num_of_histories)]


    # plot loss and validation loss
    for history in range(num_of_histories):
        ax0[history].plot(histories[history].history['loss'], label ='loss')
        ax0[history].plot(histories[history].history['val_loss'], label='val loss')
        ax0[history].grid(True)
        ax0[history].set_xlabel('Epoch', fontsize=fontsize)
        ax0[history].set_ylabel('Loss', fontsize=fontsize)
        ax0[history].legend(fontsize=fontsize)
        ax0[history].tick_params(axis='both', which='major', labelsize=fontsize)
        ax0[history].tick_params(axis='both', which='minor', labelsize=fontsize)
        # ax0[history].axis("off")\frame1.axes.get_xaxis().set_visible(False)
        if num_of_histories > 3:
            ax0[history].axes.get_xaxis().set_visible(False)
        for val in (histories[history].history['loss'], histories[history].history['val_loss'])[1:]:
            ax0[history].annotate('%0.5f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'), fontsize=fontsize)
        # get model info
        model_info = print_model_info_autoencoder(histories[history].history['model_info'])
        ax0[history].set_title("Model loss", fontsize=fontsize)
        ax_title[history].text(0.5, 0.5, "Experiment: "+str(history+1)+"\n"+model_info, ha="center", va="center", color="black", fontsize=fontsize)
        ax_title[history].axis("off")

        # show true random images
        random_test_indexes = np.random.randint(0, num_of_train_data-1, num_of_test_images)
        for ax in range(num_of_test_images-1):
            ax_true[(ax+1)+history*num_of_test_images].imshow(train_data[random_test_indexes[ax]].reshape(x_dim,y_dim))
            ax_true[(ax+1)+history*num_of_test_images].set_title("True image", fontsize=fontsize)
            ax_true[(ax+1)+history*num_of_test_images].axis("off")
        # ax_true[num_of_test_images-1+history*num_of_test_images].axis("off")

        # get the predictions
        prediction = histories[history].model.predict(train_data[random_test_indexes,:,:,:])
        for ax in range(num_of_test_images-1):
            ax_pred[(ax+1)+history*num_of_test_images].imshow(prediction[ax].reshape(x_dim,y_dim))
            ax_pred[(ax+1)+history*num_of_test_images].set_title("Predicted image", fontsize=fontsize)
            ax_pred[(ax+1)+history*num_of_test_images].axis("off")
        # ax_pred[num_of_test_images-1+history*num_of_test_images].axis("off")

    plt.rcParams.update({'font.size': fontsize})
    
    ### plot all together ###
    fig2 = plt.figure(2, figsize=(10,4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[0.6, 0.4], height_ratios=[1])
    if num_of_histories > 0:
        ax = plt.subplot(gs[0, 0])

        for history in range(num_of_histories):
            ax.plot(histories[history].history['val_loss'], label="experiment "+str(history+1))
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title("Models' losses", fontsize=15)

        # scatter experimets
        ax = plt.subplot(gs[0, 1])

        for history in range(num_of_histories):
            ax.scatter(history, histories[history].history['loss'][-1], label="exp "+str(history+1)+" loss", marker="o")
            ax.scatter(history, histories[history].history['val_loss'][-1], label="exp "+str(history+1)+" val loss", marker="X")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Experiments', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_xticks(range(num_of_histories))
        ax.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories)])
        ax.set_title("Models' losses per hyperparameters", fontsize=15)


    # plt.close()
    _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(top=0.99)
    _ = fig2.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show(block=True)

# def autoencoder_visualization_window_(histories, train_data, fontsize=7):

#     # set plot surface
#     num_of_test_images = 4
#     num_of_test_images+=1
#     num_of_histories = len(histories)
#     # if num_of_histories > 1:
#     num_of_histories += 1

#     fig = plt.figure(figsize=(10,4*num_of_histories))
#     # fig = plt.figure(figsize=(10,4*num_of_histories))
#     fig.suptitle("Visualization of Loss", fontsize=fontsize+5)
#     height_ratios = np.ones(num_of_histories)
#     for i in range(num_of_histories-1):
#         height_ratios[i] *= 0.6/(num_of_histories-1)
#     height_ratios[num_of_histories-1] *= 0.4
#     print(height_ratios)
#     gs  = gridspec.GridSpec(num_of_histories, 3, width_ratios=[0.66, 0.165, 0.165], height_ratios=height_ratios)
#     ax0 = [plt.subplot(gs[i, :]) for i in range(num_of_histories-1)]

#     # print hyperparametres
#     for history in range(num_of_histories-1):
#         model_info = print_model_info_autoencoder(histories[history].history['model_info'])
#         ax0[history].text(0.5, 0.5, "Experiment: "+str(history+1)+"\n"+model_info+"\nModel loss", ha="center", va="center", color="black", fontsize=fontsize)
#         ax0[history].axis("off")

#     ### plot all together ###
#     if num_of_histories > 1:
#         ax = plt.subplot(gs[num_of_histories-1:, 0])

#         for history in range(num_of_histories-1):
#             ax.plot(histories[history].history['val_loss'], label="experiment "+str(history+1))
#         ax.grid(True)
#         ax.legend(fontsize=fontsize)
#         ax.set_xlabel('Epoch', fontsize=fontsize)
#         ax.set_ylabel('Loss', fontsize=fontsize)
#         ax.set_title("Models' losses", fontsize=fontsize)

#         # scatter experimets
#         ax = plt.subplot(gs[num_of_histories-1:, 1:])

#         for history in range(num_of_histories-1):
#             ax.scatter(history, histories[history].history['loss'][-1], label="exp "+str(history+1)+" loss", marker="o")
#             ax.scatter(history, histories[history].history['val_loss'][-1], label="exp "+str(history+1)+" val loss", marker="X")
#         ax.grid(True)
#         ax.legend(fontsize=fontsize)
#         ax.set_xlabel('Experiments', fontsize=fontsize)
#         ax.set_ylabel('Loss', fontsize=fontsize)
#         ax.set_xticks(range(num_of_histories-1))
#         ax.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories-1)] , fontsize=fontsize) 
#         ax.set_title("Models' losses per hyperparameters", fontsize=fontsize)


#     # plt.close()
#     _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
#     plt.show(block=True)
#     return fig


def get_matrix_classification_report(report):
    matrix = []
    matrix_row = []
    for item in report.items():
        if isinstance(item[1], dict):
            matrix_row = list(item[1].values())[:-1]
            matrix.append(matrix_row)
    # add accuracy
    accuracy = report["accuracy"]
    matrix.append([accuracy]*3)
    matrix = np.array(matrix)

    return matrix


# plot the loss for multiple models + some predicted data
def classifier_prediction_visualization_window(model, test_data, test_labels, num_of_test_images=8, fontsize=9):

    # set plot surface
    num_of_classes = len(test_labels[0])
    classes = np.arange(0,num_of_classes)
    heatmap_report_size = 5
    x_dim = test_data.shape[1]
    y_dim = test_data.shape[2]
    fig = plt.figure(figsize=(8,(2*num_of_test_images)+heatmap_report_size))
    # fig.suptitle("Classification Report\nConfusion Matrix\nVisualization of Classifier predictions", fontsize=fontsize+2)

    gs  = gridspec.GridSpec(num_of_test_images+heatmap_report_size+1, 3, width_ratios=[0.33, 0.33, 0.33], height_ratios=np.ones(num_of_test_images+heatmap_report_size+1))
    ax_heatmap = plt.subplot(gs[0:heatmap_report_size,0:2])
    ax_cmatrix = plt.subplot(gs[0:heatmap_report_size,2:])
    ax_true = [plt.subplot(gs[i+heatmap_report_size+1,0]) for i in range(num_of_test_images)]
    ax_pred = [plt.subplot(gs[i+heatmap_report_size+1,1]) for i in range(num_of_test_images)]
    ax_bar = [plt.subplot(gs[i+heatmap_report_size+1,2]) for i in range(num_of_test_images)]

    # shuffle random
    random_order = np.arange(0,test_data.shape[0])
    np.random.shuffle(random_order)
    test_data = test_data[random_order]
    test_labels = test_labels[random_order]
    num_of_test_images_random = int(num_of_test_images/2)

    # get the predictions
    prediction_hot = model.predict(test_data[:,:,:,:])
    prediction = np.argmax(prediction_hot, axis=1) #history.model.predict_classes(test_data[0:num_of_test_images,:,:,:])

    # get the correct and the incorrect ones
    num_of_correct = np.sum(prediction == np.argmax(test_labels, axis=1))
    num_of_incorrect = np.sum(prediction != np.argmax(test_labels, axis=1))

    # plot confusion matrix
    cm = confusion_matrix(np.argmax(test_labels, axis=1), prediction)
    ax_cmatrix.imshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            text = ax_cmatrix.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=fontsize-5)
    ax_cmatrix.set_xticks(classes)
    ax_cmatrix.set_yticks(classes)
    ax_cmatrix.set_xticklabels(classes, fontsize=fontsize)
    ax_cmatrix.set_yticklabels(classes, fontsize=fontsize)
    ax_cmatrix.set_xlabel("Predicted", fontsize=fontsize+1)
    ax_cmatrix.set_ylabel("True", fontsize=fontsize+1)
    ax_cmatrix.set_title("Confusion Matrix", fontsize=fontsize+2)

    # plot classification report
    report = classification_report(np.argmax(test_labels, axis=1), prediction, output_dict=True)
    matrix = get_matrix_classification_report(report)
    sns.heatmap(matrix, annot=True, ax=ax_heatmap, linewidths=0.5, robust=True, annot_kws={"size":fontsize-4})
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=fontsize-1)
    ax_heatmap.xaxis.tick_top() # x axis on top
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.set_xticklabels(["Precision", "Recall", "F1"], fontsize=fontsize)
    ax_heatmap.set_yticklabels(["Class %d"%i for i in classes]+["macro avg", "weighted avg", "Accuracy"], fontsize=fontsize)
    plt.setp(ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    ax_heatmap.set_title("Classification Report\nTested: %d images\nCorrect: %d Incorrect: %d"%(num_of_correct+num_of_incorrect, num_of_correct, num_of_incorrect), fontsize=fontsize+2)

    # get the wrong predictions
    wrong_predictions = []
    for i, pred in enumerate(prediction):
        if pred != np.argmax(test_labels[i], axis=0): 
            wrong_predictions.append(i)
    print(len(wrong_predictions))

    # title
    title_ax = plt.subplot(gs[heatmap_report_size,:])
    title_ax.axis("off")
    # title_ax.set_title("True and their Predicted images", fontsize=fontsize+2)
    title_ax.text(0.5, 0.5, "True images and their Predicted labels with Propabilities", ha="center", va="center", color="black", fontsize=fontsize+2)
    # title_ax.text(0.5, 0.6, "The first half consists of random images and the second consists of wrong predictions", ha="center", va="center", color="black", fontsize=fontsize)

    # show true random images
    for ax in range(num_of_test_images):
        pred = ax
        # print the wrong ones
        if ax >= num_of_test_images_random:
            pred = wrong_predictions[ax]
        ax_true[ax].imshow(test_data[pred].reshape(x_dim,y_dim))
        ax_true[ax].set_title("True image label: "+str(np.argmax(test_labels[pred], axis=0)), fontsize=fontsize-2)
        ax_true[ax].axis("off")

    for ax in range(num_of_test_images):
        pred = ax
        # print the wrong ones
        if ax >= num_of_test_images_random:
            pred = wrong_predictions[ax]
        ax_pred[ax].text(0.5,0.5, str(prediction[pred]), fontsize=fontsize+10 ,verticalalignment='center', horizontalalignment='center') 
        ax_pred[ax].xaxis.set_visible(False)
        ax_pred[ax].yaxis.set_visible(False)
        ax_pred[ax].set_title("Predicted label", fontsize=fontsize-2)
        # if wrong prediction then font rent
        if prediction[pred] != np.argmax(test_labels[pred], axis=0): 
            ax_pred[ax].set_facecolor("r")

    # plot bar propabilities
    for ax in range(num_of_test_images):
        pred = prediction_hot[ax]
        # print the wrong ones
        if ax >= num_of_test_images_random:
            pred = prediction_hot[wrong_predictions[ax]]
        ax_bar[ax].bar(classes, pred)
        ax_bar[ax].yaxis.set_visible(False)
        ax_bar[ax].set_xticks(classes) 
        ax_bar[ax].set_xticklabels(classes, fontsize=fontsize-5)

    _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show(block=True) 
    # return fig

# plot the loss for multiple models + some predicted data
def classifier_prediction_visualization(model, test_data, test_labels, num_of_test_images=8):

    # set plot surface
    num_of_classes = len(test_labels[0])
    classes = np.arange(0,num_of_classes)
    heatmap_report_size = 5
    confusion_matrix_size = 5
    x_dim = test_data.shape[1]
    y_dim = test_data.shape[2]
    fig = plt.figure(figsize=(8,3*num_of_test_images+heatmap_report_size+confusion_matrix_size+1))
    fig.suptitle("Classification Report\nConfusion Matrix\nVisualization of Classifier predictions", fontsize=20)

    gs  = gridspec.GridSpec(num_of_test_images+heatmap_report_size+confusion_matrix_size+1, 3, width_ratios=[0.33, 0.33, 0.33], height_ratios=np.ones(num_of_test_images+heatmap_report_size+confusion_matrix_size+1))
    ax_heatmap = plt.subplot(gs[0:heatmap_report_size,:])
    ax_cmatrix = plt.subplot(gs[heatmap_report_size:heatmap_report_size+confusion_matrix_size,:])
    ax_true = [plt.subplot(gs[i+heatmap_report_size+confusion_matrix_size+1,0]) for i in range(num_of_test_images)]
    ax_pred = [plt.subplot(gs[i+heatmap_report_size+confusion_matrix_size+1,1]) for i in range(num_of_test_images)]
    ax_bar = [plt.subplot(gs[i+heatmap_report_size+confusion_matrix_size+1,2]) for i in range(num_of_test_images)]

    # shuffle random
    random_order = np.arange(0,test_data.shape[0])
    np.random.shuffle(random_order)
    test_data = test_data[random_order]
    test_labels = test_labels[random_order]
    num_of_test_images_random = int(num_of_test_images/2)

    # get the predictions
    prediction_hot = model.predict(test_data[:,:,:,:])
    prediction = np.argmax(prediction_hot, axis=1) #history.model.predict_classes(test_data[0:num_of_test_images,:,:,:])

    # get the correct and the incorrect ones
    num_of_correct = np.sum(prediction == np.argmax(test_labels, axis=1))
    num_of_incorrect = np.sum(prediction != np.argmax(test_labels, axis=1))

    # plot confusion matrix
    cm = confusion_matrix(np.argmax(test_labels, axis=1), prediction)
    ax_cmatrix.imshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            text = ax_cmatrix.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax_cmatrix.set_xticks(classes)
    ax_cmatrix.set_yticks(classes)
    ax_cmatrix.set_xticklabels(classes, fontsize=12)
    ax_cmatrix.set_yticklabels(classes, fontsize=12)
    ax_cmatrix.set_xlabel("Predicted", fontsize=15)
    ax_cmatrix.set_ylabel("True", fontsize=15)
    ax_cmatrix.set_title("Confusion Matrix", fontsize=20)

    # plot classification report
    report = classification_report(np.argmax(test_labels, axis=1), prediction, output_dict=True)
    matrix = get_matrix_classification_report(report)
    sns.heatmap(matrix, annot=True, ax=ax_heatmap, linewidths=0.5, robust=True)
    ax_heatmap.xaxis.tick_top() # x axis on top
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.set_xticklabels(["Precision", "Recall", "F1"], fontsize=12)
    ax_heatmap.set_yticklabels(["Class %d"%i for i in classes]+["macro avg", "weighted avg", "Accuracy"], fontsize=12)
    plt.setp(ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    ax_heatmap.set_title("Classification Report\nTested: %d images\nCorrect: %d Incorrect: %d\n"%(num_of_correct+num_of_incorrect, num_of_correct, num_of_incorrect), fontsize=20)

    # get the wrong predictions
    wrong_predictions = []
    for i, pred in enumerate(prediction):
        if pred != np.argmax(test_labels[i], axis=0): 
            wrong_predictions.append(i)
    # print(len(wrong_predictions))

    # title
    title_ax = plt.subplot(gs[heatmap_report_size+confusion_matrix_size,:])
    title_ax.axis("off")
    title_ax.set_title("True and their Predicted images", fontsize=20)
    title_ax.text(0.5, 0.5, "The first half consists of random images and the second consists of wrong predictions", ha="center", va="center", color="black", fontsize=12)

    # show true random images
    for ax in range(num_of_test_images):
        pred = ax
        # print the wrong ones
        if ax >= num_of_test_images_random:
            pred = wrong_predictions[ax]
        ax_true[ax].imshow(test_data[pred].reshape(x_dim,y_dim))
        ax_true[ax].set_title("True image label: "+str(np.argmax(test_labels[pred], axis=0)) )
        ax_true[ax].axis("off")

    for ax in range(num_of_test_images):
        pred = ax
        # print the wrong ones
        if ax >= num_of_test_images_random:
            pred = wrong_predictions[ax]
        ax_pred[ax].text(0.5,0.5, str(prediction[pred]), fontsize=60 ,verticalalignment='center', horizontalalignment='center') 
        ax_pred[ax].xaxis.set_visible(False)
        ax_pred[ax].yaxis.set_visible(False)
        ax_pred[ax].set_title("Predicted label")
        # if wrong prediction then font rent
        if prediction[pred] != np.argmax(test_labels[pred], axis=0): 
            ax_pred[ax].set_facecolor("r")

    # plot bar propabilities
    for ax in range(num_of_test_images):
        pred = prediction_hot[ax]
        # print the wrong ones
        if ax >= num_of_test_images_random:
            pred = prediction_hot[wrong_predictions[ax]]
        ax_bar[ax].bar(classes, pred)
        ax_bar[ax].yaxis.set_visible(False)
        ax_bar[ax].set_xticks(classes) 
        ax_bar[ax].set_xticklabels(classes)

    _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
    
    # plt.close()
    plt.show(block=False) 
    return fig

def classifier_loss_visualization_window(histories, fontsize=5):

    # set plot surface
    num_of_offset = 2
    num_of_histories = len(histories)
    # num_of_histories += 1

    fig = plt.figure(figsize=(15,5*num_of_histories))
    # fig.suptitle("Visualization of Metrics for Every Experiment", fontsize=10)

    gs  = gridspec.GridSpec(num_of_offset*num_of_histories, 4, width_ratios=[0.35, 0.15, 0.15, 0.35], height_ratios=np.ones(num_of_offset*num_of_histories))

    ax0 = [plt.subplot(gs[i*num_of_offset:i*num_of_offset+num_of_offset, 0]) for i in range(num_of_histories)]
    ax_heatmap = [plt.subplot(gs[i*num_of_offset:i*num_of_offset+num_of_offset, 3]) for i in range(num_of_histories)]

    # get f1 score
    f1 = []
    f1_val = []
    for history in range(num_of_histories):
        pre = np.array(histories[history].history["val_Precision"])
        rec = np.array(histories[history].history["val_Recall"])
        try:
            c = np.divide((pre * rec), (pre + rec), out=np.zeros_like(pre * rec), where=(pre + rec)!=0)
            f1_val.append(2* c)
        except:
            f1_val.append(pre + rec)

        pre = np.array(histories[history].history["Precision"])
        rec = np.array(histories[history].history["Recall"])
        try:
            c = np.divide((pre * rec), (pre + rec), out=np.zeros_like(pre * rec), where=(pre + rec)!=0)
            f1.append(2* c)
        except:
            f1.append(pre + rec)

    history = 0
    for i in np.arange(0, (num_of_histories)*num_of_offset, num_of_offset):
        # accuracy
        subplot = plt.subplot(gs[i,1]) 
        subplot.plot(histories[history].history['accuracy'], label ='accuracy')
        subplot.plot(histories[history].history['val_accuracy'], label ='test accuracy')
        subplot.set_title("Accuracy", fontsize=fontsize)
        subplot.legend(fontsize=fontsize)
        subplot.tick_params(axis='both', which='major', labelsize=fontsize)
        subplot.tick_params(axis='both', which='minor', labelsize=fontsize)
        for val in (histories[history].history['accuracy'], histories[history].history['val_accuracy'])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'), fontsize=fontsize)
        subplot.grid(True)
        # precision
        subplot = plt.subplot(gs[i+1,1])
        subplot.plot(histories[history].history['Precision'], label ='precision')
        subplot.plot(histories[history].history['val_Precision'], label ='test precision')
        subplot.set_title("Precision", fontsize=fontsize)
        subplot.legend(fontsize=fontsize)
        subplot.tick_params(axis='both', which='major', labelsize=fontsize)
        subplot.tick_params(axis='both', which='minor', labelsize=fontsize)
        for val in (histories[history].history['Precision'], histories[history].history['val_Precision'])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'), fontsize=fontsize)
        subplot.grid(True)
        # recall
        subplot = plt.subplot(gs[i,2]) 
        subplot.plot(histories[history].history['Recall'], label ='recall')
        subplot.plot(histories[history].history['val_Recall'], label ='test recall')
        subplot.set_title("Recall", fontsize=fontsize)
        subplot.legend(fontsize=fontsize)
        subplot.tick_params(axis='both', which='major', labelsize=fontsize)
        subplot.tick_params(axis='both', which='minor', labelsize=fontsize)
        for val in (histories[history].history['Recall'], histories[history].history['val_Recall'])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'), fontsize=fontsize)
        subplot.grid(True)
        # f1
        subplot = plt.subplot(gs[i+1,2]) 
        subplot.plot(f1[history], label ='f1')
        subplot.plot(f1_val[history], label ='test f1')
        subplot.set_title("F1", fontsize=fontsize)
        subplot.legend(fontsize=fontsize)
        subplot.tick_params(axis='both', which='major', labelsize=fontsize)
        subplot.tick_params(axis='both', which='minor', labelsize=fontsize)
        for val in (f1[history], f1_val[history])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'), fontsize=fontsize)
        subplot.grid(True)
        history+=1

    # plot loss and validation loss
    for history in range(num_of_histories):
        ax0[history].plot(histories[history].history['loss'], label ='loss')
        ax0[history].plot(histories[history].history['val_loss'], label='test loss')
        ax0[history].set_xlabel('Epoch', fontsize=fontsize)
        ax0[history].set_ylabel('Loss', fontsize=fontsize)
        ax0[history].legend(fontsize=fontsize)
        ax0[history].tick_params(axis='both', which='major', labelsize=fontsize)
        ax0[history].tick_params(axis='both', which='minor', labelsize=fontsize)
        for val in (histories[history].history['loss'], histories[history].history['val_loss'])[1:]:
            ax0[history].annotate('%0.5f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'), fontsize=fontsize)
        ax0[history].grid(True)
        if num_of_histories > 2:
            ax0[history].axes.get_xaxis().set_visible(False)
        classifier = print_model_info_classifier(histories[history].history['model_info'])
        ax0[history].set_title("Experiment: "+str(history+1)+"\n"+classifier+"\nModel loss", fontsize=fontsize+1)

    # plot classification report on validation set
    for history in range(num_of_histories):
        matrix = get_matrix_classification_report(histories[history].history['classification_report'])
        im = sns.heatmap(matrix, annot=True, ax=ax_heatmap[history], linewidths=0.5, robust=True, annot_kws={"size":fontsize})
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=fontsize)
        ax_heatmap[history].xaxis.tick_top() # x axis on top
        ax_heatmap[history].xaxis.set_label_position('top')
        ax_heatmap[history].set_xticklabels(["Precision", "Recall", "F1"], fontsize=fontsize)
        ax_heatmap[history].set_yticklabels(["Class %d"%i for i in range(10)]+["macro avg", "weighted avg", "Accuracy"], fontsize=fontsize)
        im = plt.setp(ax_heatmap[history].yaxis.get_majorticklabels(), rotation=0)
        num_of_correct = histories[history].history['num_of_correct']
        num_of_incorrect = histories[history].history['num_of_incorrect']
        ax_heatmap[history].set_title("Classification Report on Test Data\nTested: %d images Correct: %d Incorrect: %d"%(num_of_correct+num_of_incorrect, num_of_correct, num_of_incorrect), fontsize=fontsize+1)


    ### plot all together test losses and metrics ###
    # plt.rcParams.update({'font.size': 2})
    # plt.xticks(fontsize=7)
    
    ### plot all together ###
    fig2 = plt.figure(2, figsize=(12,6))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[0.5, 0.5], height_ratios=[1,1])

    if num_of_histories > 0:
        ax0 = plt.subplot(gs[0:, 0])
        ax1 = plt.subplot(gs[0, 1])
        ax2 = plt.subplot(gs[1, 1])

        for history in range(num_of_histories):
            # ax0.plot(histories[history].history['loss'], label ='loss')
            ax0.plot(histories[history].history['val_loss'], label='experiment '+str(history+1))
            ax0.set_xlabel('Epoch', fontsize=7)
            ax0.set_ylabel('Loss', fontsize=7)
            ax0.legend(fontsize=7)
            ax0.set_title("Models' losses on test set", fontsize=12)
            # for val in [ histories[h].history['val_loss'] for h in range(num_of_histories-1) ]:
            val = histories[history].history['val_loss'][-1]
            ax0.annotate('%0.5f exp %d'%(val, history+1), xy=(1, val), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
            ax0.grid(True)

            # scatter experimets
            ax1.scatter(history, histories[history].history['loss'][-1], label="exp "+str(history+1)+" loss", marker="o")
            ax1.scatter(history, histories[history].history['val_loss'][-1], label="exp "+str(history+1)+" test loss", marker="X")
            ax1.grid(True)
            ax1.legend(fontsize=7)
            ax1.set_xlabel('Experiments', fontsize=7)
            ax1.set_ylabel('Accuracy', fontsize=7)
            ax1.set_xticks(range(num_of_histories))
            ax1.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories)])
            # ax1.set_xticks(range(num_of_histories-1), np.arange(1,num_of_histories-1))
            ax1.set_title("Models' losses per hyperparameters", fontsize=12)

            ax2.scatter(history, histories[history].history['accuracy'][-1], label="exp "+str(history+1)+" acc", marker="o")
            ax2.scatter(history, histories[history].history['val_accuracy'][-1], label="exp "+str(history+1)+" test acc", marker="X")
            ax2.grid(True)
            ax2.legend(fontsize=7)
            ax2.set_xlabel('Experiments', fontsize=7)
            ax2.set_ylabel('Loss', fontsize=7)
            ax2.set_xticks(range(num_of_histories))
            ax2.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories)])
            # ax2.set_xticks(range(num_of_histories-1), np.arange(1,num_of_histories-1))
            ax2.set_title("Models' accuracy per hyperparameters", fontsize=12)



    _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(top=0.90)
    _ = fig2.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show(block=True)

def classifier_loss_visualization(histories):

    # set plot surface
    num_of_offset = 2
    num_of_histories = len(histories)
    num_of_histories += 1

    fig = plt.figure(figsize=(25,8*num_of_histories))
    fig.suptitle("Visualization of Metrics for Every Experiment", fontsize=25)

    gs  = gridspec.GridSpec(num_of_offset*num_of_histories, 4, width_ratios=[0.35, 0.15, 0.15, 0.35], height_ratios=np.ones(num_of_offset*num_of_histories))

    ax0 = [plt.subplot(gs[i*num_of_offset:i*num_of_offset+num_of_offset, 0]) for i in range(num_of_histories-1)]
    ax_heatmap = [plt.subplot(gs[i*num_of_offset:i*num_of_offset+num_of_offset, 3]) for i in range(num_of_histories-1)]

    # get f1 score
    f1 = []
    f1_val = []
    for history in range(num_of_histories-1):
        pre = np.array(histories[history].history["val_Precision"])
        rec = np.array(histories[history].history["val_Recall"])
        try:
            c = np.divide((pre * rec), (pre + rec), out=np.zeros_like(pre * rec), where=(pre + rec)!=0)
            f1_val.append(2* c)
        except:
            f1_val.append(pre + rec)

        pre = np.array(histories[history].history["Precision"])
        rec = np.array(histories[history].history["Recall"])
        try:
            c = np.divide((pre * rec), (pre + rec), out=np.zeros_like(pre * rec), where=(pre + rec)!=0)
            f1.append(2* c)
        except:
            f1.append(pre + rec)

    history = 0
    for i in np.arange(0, (num_of_histories-1)*num_of_offset, num_of_offset):
        # accuracy
        subplot = plt.subplot(gs[i,1]) 
        subplot.plot(histories[history].history['accuracy'], label ='accuracy')
        subplot.plot(histories[history].history['val_accuracy'], label ='test accuracy')
        subplot.set_title("Accuracy")
        subplot.legend()
        for val in (histories[history].history['accuracy'], histories[history].history['val_accuracy'])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
        subplot.grid(True)
        # precision
        subplot = plt.subplot(gs[i+1,1])
        subplot.plot(histories[history].history['Precision'], label ='precision')
        subplot.plot(histories[history].history['val_Precision'], label ='test precision')
        subplot.set_title("Precision")
        subplot.legend()
        for val in (histories[history].history['Precision'], histories[history].history['val_Precision'])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
        subplot.grid(True)
        # recall
        subplot = plt.subplot(gs[i,2]) 
        subplot.plot(histories[history].history['Recall'], label ='recall')
        subplot.plot(histories[history].history['val_Recall'], label ='test recall')
        subplot.set_title("Recall")
        subplot.legend()
        for val in (histories[history].history['Recall'], histories[history].history['val_Recall'])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
        subplot.grid(True)
        # f1
        subplot = plt.subplot(gs[i+1,2]) 
        subplot.plot(f1[history], label ='f1')
        subplot.plot(f1_val[history], label ='test f1')
        subplot.set_title("F1")
        subplot.legend()
        for val in (f1[history], f1_val[history])[1:]:
            subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
        subplot.grid(True)
        history+=1

    # plot loss and validation loss
    for history in range(num_of_histories-1):
        ax0[history].plot(histories[history].history['loss'], label ='loss')
        ax0[history].plot(histories[history].history['val_loss'], label='test loss')
        ax0[history].set_xlabel('Epoch', fontsize=15)
        ax0[history].set_ylabel('Loss', fontsize=15)
        ax0[history].legend(fontsize=15)
        for val in (histories[history].history['loss'], histories[history].history['val_loss'])[1:]:
            ax0[history].annotate('%0.5f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
        ax0[history].grid(True)
        classifier = print_model_info_classifier(histories[history].history['model_info'])
        ax0[history].set_title("Experiment: "+str(history+1)+"\n"+classifier+"\nModel loss", fontsize=15)

    # plot classification report on validation set
    for history in range(num_of_histories-1):
        matrix = get_matrix_classification_report(histories[history].history['classification_report'])
        im = sns.heatmap(matrix, annot=True, ax=ax_heatmap[history], linewidths=0.5, robust=True)
        ax_heatmap[history].xaxis.tick_top() # x axis on top
        ax_heatmap[history].xaxis.set_label_position('top')
        ax_heatmap[history].set_xticklabels(["Precision", "Recall", "F1"], fontsize=12)
        ax_heatmap[history].set_yticklabels(["Class %d"%i for i in range(10)]+["macro avg", "weighted avg", "Accuracy"], fontsize=12)
        im = plt.setp(ax_heatmap[history].yaxis.get_majorticklabels(), rotation=0)
        num_of_correct = histories[history].history['num_of_correct']
        num_of_incorrect = histories[history].history['num_of_incorrect']
        ax_heatmap[history].set_title("Classification Report on Test Data\nTested: %d images\nCorrect: %d Incorrect: %d"%(num_of_correct+num_of_incorrect, num_of_correct, num_of_incorrect), fontsize=15)


    ### plot all together test losses and metrics ###
    if num_of_histories > 2:
        ax0 = plt.subplot(gs[(num_of_histories-1)*num_of_offset:(num_of_histories-1)*num_of_offset+num_of_offset, 0])
        ax1 = plt.subplot(gs[(num_of_histories-1)*num_of_offset:(num_of_histories-1)*num_of_offset+num_of_offset-1, 1:])
        ax2 = plt.subplot(gs[(num_of_histories-1)*num_of_offset+1:(num_of_histories-1)*num_of_offset+num_of_offset, 1:])

        # subplot1 = plt.subplot(gs[(num_of_histories-1)*num_of_offset,1]) 
        # subplot1.grid(True)

        # subplot2 = plt.subplot(gs[(num_of_histories-1)*num_of_offset,2]) 
        # subplot2.grid(True)

        # subplot3 = plt.subplot(gs[(num_of_histories-1)*num_of_offset+1,1]) 
        # subplot3.grid(True)

        # subplot4 = plt.subplot(gs[(num_of_histories-1)*num_of_offset+1,2]) 
        # subplot4.grid(True)
        # for history in range(num_of_histories-1):
        #     # accuracy
        #     subplot1.plot(histories[history].history['val_accuracy'], label ='exp '+str(history+1))
        #     subplot1.set_title("Accuracy")
        #     # for val in (histories[history].history['accuracy'], histories[history].history['val_accuracy']):
        #     #     subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))

        #     # precision
        #     subplot2.plot(histories[history].history['val_Precision'], label ='exp '+str(history+1))
        #     subplot2.set_title("Precision")
        #     # for val in (histories[history].history['Precision'], histories[history].history['val_Precision']):
        #     #     subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))

        #     # recall
        #     subplot3.plot(histories[history].history['val_Recall'], label ='exp '+str(history+1))
        #     subplot3.set_title("Recall")
        #     # for val in (histories[history].history['Recall'], histories[history].history['val_Recall']):
        #     #     subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))

        #     # f1
        #     subplot4.plot(f1_val[history], label ='exp '+str(history+1))
        #     subplot4.set_title("F1")
        #     # for val in (f1[history], f1_val[history]):
        #     #     subplot.annotate('%0.3f'%val[-1], xy=(1, val[-1]), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))

        # subplot1.legend()
        # subplot2.legend()
        # subplot3.legend()
        # subplot4.legend()

        for history in range(num_of_histories-1):
            # ax0.plot(histories[history].history['loss'], label ='loss')
            ax0.plot(histories[history].history['val_loss'], label='experiment '+str(history+1))
            ax0.set_xlabel('Epoch', fontsize=15)
            ax0.set_ylabel('Loss', fontsize=15)
            ax0.legend(fontsize=15)
            ax0.set_title("Models' losses on test set", fontsize=15)
            # for val in [ histories[h].history['val_loss'] for h in range(num_of_histories-1) ]:
            val = histories[history].history['val_loss'][-1]
            ax0.annotate('%0.5f exp %d'%(val, history+1), xy=(1, val), xytext=(5, 0), textcoords='offset points', xycoords=('axes fraction', 'data'))
            ax0.grid(True)

            # scatter experimets
            ax1.scatter(history, histories[history].history['loss'][-1], label="exp "+str(history+1)+" loss", marker="o")
            ax1.scatter(history, histories[history].history['val_loss'][-1], label="exp "+str(history+1)+" test loss", marker="X")
            ax1.grid(True)
            ax1.legend()
            ax1.set_xlabel('Experiments', fontsize=15)
            ax1.set_ylabel('Accuracy', fontsize=15)
            ax1.set_xticks(range(num_of_histories-1))
            ax1.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories-1)])
            # ax1.set_xticks(range(num_of_histories-1), np.arange(1,num_of_histories-1))
            ax1.set_title("Models' losses per hyperparameters", fontsize=15)

            ax2.scatter(history, histories[history].history['accuracy'][-1], label="exp "+str(history+1)+" acc", marker="o")
            ax2.scatter(history, histories[history].history['val_accuracy'][-1], label="exp "+str(history+1)+" test acc", marker="X")
            ax2.grid(True)
            ax2.legend()
            ax2.set_xlabel('Experiments', fontsize=15)
            ax2.set_ylabel('Loss', fontsize=15)
            ax2.set_xticks(range(num_of_histories-1))
            ax2.set_xticklabels(["exp "+str(x+1) for x in range(num_of_histories-1)])
            # ax2.set_xticks(range(num_of_histories-1), np.arange(1,num_of_histories-1))
            ax2.set_title("Models' accuracy per hyperparameters", fontsize=15)



    _ = fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show(block=False)
    return None

# if __name__ == '__main__':


    # a = classifier_prediction_visualization(histor, x_train_scal, y_train, num_of_test_images=20)
    # autoencoder_visualization(history, x_train_scal)