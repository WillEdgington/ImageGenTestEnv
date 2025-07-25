import matplotlib.pyplot as plt

def plotGANLoss(results):
    genTrainLoss = results["generator_train_loss"]
    genTestLoss = results["generator_test_loss"]
    
    disTrainLoss = results["discriminator_train_loss"]
    disTestLoss = results["discriminator_test_loss"]

    epochs = range(len(genTrainLoss))

    plt.figure(figsize=(15, 7))

    # Plot generator loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, genTrainLoss, label="Train loss")
    plt.plot(epochs, genTestLoss, label="Test loss")
    plt.title("Generator Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot discriminator loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, disTrainLoss, label="Train loss")
    plt.plot(epochs, disTestLoss, label="Test loss")
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

def plotGANTrainLoss(results):
    genTrainLoss = results["generator_train_loss"]
    disTrainLoss = results["discriminator_train_loss"]

    epochs = range(len(genTrainLoss))

    plt.figure(figsize=(15,7))
    plt.plot(epochs, genTrainLoss, label="Generator")
    plt.plot(epochs, disTrainLoss, label="Discriminator")
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()