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

def plotVAELoss(results, title: str=""):
    trainLoss = results["train_loss"]
    testLoss = results["test_loss"]

    DklTrainLoss = results["Dkl_train_loss"]
    DklTestLoss = results["Dkl_test_loss"]

    reconTrainLoss = results["recon_train_loss"]
    reconTestLoss = results["recon_test_loss"]

    epochs = range(len(trainLoss))

    plt.figure(figsize=(15,7))
    
    # Plot total loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, trainLoss, label="Train")
    plt.plot(epochs, testLoss, label="Test")
    plt.title("Total Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot KL divergence loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, DklTrainLoss, label="Train")
    plt.plot(epochs, DklTestLoss, label="Test")
    plt.title("KL Divergence Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot reconstruction loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, reconTrainLoss, label="Train")
    plt.plot(epochs, reconTestLoss, label="Test")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

def plotVAELossAndBeta(results, title: str=""):
    trainLoss = results["train_loss"]
    testLoss = results["test_loss"]

    DklTrainLoss = results["Dkl_train_loss"]
    DklTestLoss = results["Dkl_test_loss"]

    reconTrainLoss = results["recon_train_loss"]
    reconTestLoss = results["recon_test_loss"]

    # Dkl + recon (original VAE loss function)
    sumDKlReconTrainLoss = [k + r for k, r in zip(DklTrainLoss, reconTrainLoss)]
    sumDklReconTestLoss = [k + r for k, r in zip(DklTestLoss, reconTestLoss)]

    # Active latent dimensions
    activeDimsTrain = results["active_latent_dims_train"]
    activeDimsTest = results["active_latent_dims_test"]

    betas = results["beta_value"]

    epochs = range(len(trainLoss))

    plt.figure(figsize=(15,7))
    
    # Plot total loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, trainLoss, label="Train")
    plt.plot(epochs, testLoss, label="Test")
    plt.title("Total Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot sum of Kl loss and reconstruction loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, sumDKlReconTrainLoss, label="Train")
    plt.plot(epochs, sumDklReconTestLoss, label="Test")
    plt.title("Dkl Loss + Recon Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot beta value
    plt.subplot(2, 3, 3)
    plt.plot(epochs, betas)
    plt.title("Beta Value")
    plt.xlabel("Epochs")

    # Plot KL divergence loss
    plt.subplot(2, 3, 4)
    plt.plot(epochs, DklTrainLoss, label="Train")
    plt.plot(epochs, DklTestLoss, label="Test")
    plt.title("KL Divergence Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot reconstruction loss
    plt.subplot(2, 3, 5)
    plt.plot(epochs, reconTrainLoss, label="Train")
    plt.plot(epochs, reconTestLoss, label="Test")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot active latent dims
    plt.subplot(2, 3, 6)
    plt.plot(epochs, activeDimsTrain, label="Train")
    plt.plot(epochs, activeDimsTest, label="Test")
    plt.title("Active Latent Dims")
    plt.xlabel("Epochs")
    plt.legend()

    plt.suptitle(t=title)
    plt.show()