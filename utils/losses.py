import matplotlib.pyplot as plt

def plotGANLoss(results):
    genTrainLoss = results["generator_train_loss"]
    genTestLoss = results["generator_test_loss"]
    
    disTrainLoss = results["discriminator_train_loss"]
    disTestLoss = results["discriminator_test_loss"]

    epochs = range(len(results["generator_train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot generator loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, genTrainLoss, label="generator_train_loss")
    plt.plot(epochs, genTestLoss, label="generator_test_loss")
    plt.title("Generator Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot discriminator loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, disTrainLoss, label="discriminator_train_loss")
    plt.plot(epochs, disTestLoss, label="discriminator_test_loss")
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()