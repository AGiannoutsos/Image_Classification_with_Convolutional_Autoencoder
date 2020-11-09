import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# plot the loss for multiple models + some predicted data
# plot the loss for multiple models + some predicted data
def autoencoder_visualization(histories, train_data, num_of_test_images=4):

    # set plot surface
    num_of_test_images+=1
    num_of_histories = len(histories)
    num_of_train_data = train_data.shape[0]
    x_dim = train_data.shape[1]
    y_dim = train_data.shape[2]
    fig = plt.figure(figsize=(15,7*num_of_histories))
    fig.suptitle("Visualization of Loss with random True and their Predicted images", fontsize=25)

    gs  = gridspec.GridSpec(num_of_test_images*num_of_histories, 3, width_ratios=[0.66, 0.165, 0.165], height_ratios=np.ones(num_of_test_images*num_of_histories))
    ax0 = [plt.subplot(gs[i*num_of_test_images:i*num_of_test_images+num_of_test_images-1, 0]) for i in range(num_of_histories)]
    ax_true = [plt.subplot(gs[i,1]) for i in range(num_of_test_images*num_of_histories)]
    ax_pred = [plt.subplot(gs[i,2]) for i in range(num_of_test_images*num_of_histories)]


    # plot loss and validation loss
    for history in range(num_of_histories):
        ax0[history].plot(histories[history].history['loss'], label ='loss')
        ax0[history].plot(histories[history].history['val_loss'], label='val loss')
        ax0[history].set_xlabel('Epoch', fontsize=15)
        ax0[history].set_ylabel('Loss', fontsize=15)
        ax0[history].legend(fontsize=15)
        ax0[history].set_title("Model loss", fontsize=15)
        ax0[history].grid(True)

        # show true random images
        random_test_indexes = np.random.randint(0, num_of_train_data-1, num_of_test_images)
        for ax in range(num_of_test_images-1):
            ax_true[ax+history*num_of_test_images].imshow(train_data[random_test_indexes[ax]].reshape(x_dim,y_dim))
            ax_true[ax+history*num_of_test_images].set_title("True image")
            ax_true[ax+history*num_of_test_images].axis("off")
        ax_true[num_of_test_images-1+history*num_of_test_images].axis("off")

        # get the predictions
        prediction = histories[history].model.predict(train_data[random_test_indexes,:,:,:])
        for ax in range(num_of_test_images-1):
            ax_pred[ax+history*num_of_test_images].imshow(prediction[ax].reshape(x_dim,y_dim))
            ax_pred[ax+history*num_of_test_images].set_title("Predicted image")
            ax_pred[ax+history*num_of_test_images].axis("off")
        ax_pred[num_of_test_images-1+history*num_of_test_images].axis("off")

    # plt.close()
    return fig

# plot the loss for multiple models + some predicted data
def classifier_prediction_visualization(history, test_data, test_labels, num_of_test_images=4):

    # set plot surface
    # num_of_test_images+=1
    # num_of_histories = len(histories)
    # num_of_train_data = test_data.shape[0]
    x_dim = test_data.shape[1]
    y_dim = test_data.shape[2]
    # fig = plt.figure(figsize=(5,2*num_of_test_images))
    fig = plt.figure(figsize=(5,10))
    fig.suptitle("Visualization of Loss with random True and their Predicted images", fontsize=25)

    gs  = gridspec.GridSpec(num_of_test_images, 2, width_ratios=[0.5, 0.5], height_ratios=np.ones(num_of_test_images))
    # ax0 = [plt.subplot(gs[i*num_of_test_images:i*num_of_test_images+num_of_test_images-1, 0]) for i in range(num_of_histories)]
    ax_true = [plt.subplot(gs[i,0]) for i in range(num_of_test_images)]
    ax_pred = [plt.subplot(gs[i,1]) for i in range(num_of_test_images)]


    # plot loss and validation loss
    # for history in range(num_of_histories):
    # ax0[history].plot(histories[history].history['loss'], label ='loss')
    # ax0[history].plot(histories[history].history['val_loss'], label='val loss')
    # ax0[history].set_xlabel('Epoch', fontsize=15)
    # ax0[history].set_ylabel('Loss', fontsize=15)
    # ax0[history].legend(fontsize=15)
    # ax0[history].set_title("Model loss", fontsize=15)
    # ax0[history].grid(True)

    # show true random images
    # random_test_indexes = np.random.randint(0, num_of_train_data-1, num_of_test_images)
    for ax in range(num_of_test_images):
        ax_true[ax].imshow(test_data[ax].reshape(x_dim,y_dim))
        ax_true[ax].set_title("True image")
        ax_true[ax].axis("off")
    # ax_true[num_of_test_images-1+history*num_of_test_images].axis("off")

    # get the predictions
    prediction = np.argmax(history.model.predict(test_data[0:num_of_test_images,:,:,:]), axis=1) #history.model.predict_classes(test_data[0:num_of_test_images,:,:,:])
    for ax in range(num_of_test_images):
        # ax_pred[ax].imshow(prediction[ax].reshape(x_dim,y_dim))
        ax_pred[ax].text(0,0, str(prediction[ax])+"->"+str(np.argmax(test_labels[ax], axis=0) ) )
        ax_pred[ax].set_title("")
        ax_pred[ax].axis("off")
    # ax_pred[num_of_test_images-1+history*num_of_test_images].axis("off")

    # plt.close()
    return fig




if __name__ == '__main__':


    a = classifier_prediction_visualization(histor, x_train_scal, y_train, num_of_test_images=20)
    # autoencoder_visualization(history, x_train_scal)