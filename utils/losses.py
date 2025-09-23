import matplotlib.pyplot as plt
import numpy as np

from typing import List

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

def smoothGrads(grads: List[float], alpha: float=0.3) -> List[float]:
    ema = [grads[0]]
    for g in grads[1:]:
        ema.append(alpha * g + (1 - alpha) * ema[-1])
    return ema

def relativeLossGradients(losses: List[float], epsilon: float=1e-8) -> List[float]:
    return [0] + [(losses[i+1] - losses[i]) / (losses[i] + epsilon) for i in range(len(losses) - 1)]

def plotVAELossGradients(results, title: str="", epsilon: float=1e-8, alpha: float=0.3, displayBeta: bool=True):
    DklTrainLoss = results["Dkl_train_loss"]
    DklTestLoss = results["Dkl_test_loss"]

    reconTrainLoss = results["recon_train_loss"]
    reconTestLoss = results["recon_test_loss"]

    epochs = range(len(DklTrainLoss))

    if displayBeta:
        betas = results["beta_value"]

    DklTrainGrad = relativeLossGradients(losses=DklTrainLoss, epsilon=epsilon)
    DklTestGrad = relativeLossGradients(losses=DklTestLoss, epsilon=epsilon)

    reconTrainGrad = relativeLossGradients(losses=reconTrainLoss, epsilon=epsilon)
    reconTestGrad = relativeLossGradients(losses=reconTestLoss, epsilon=epsilon)

    sumDklReconTrainGrad = [r + d for r, d in zip(reconTrainGrad, DklTrainGrad)]
    sumDklReconTestGrad = [r + d for r, d in zip(reconTestGrad, DklTestGrad)]

    DklTrainGrad = smoothGrads(grads=DklTrainGrad, alpha=alpha)
    DklTestGrad = smoothGrads(grads=DklTestGrad, alpha=alpha)
    reconTrainGrad = smoothGrads(grads=reconTrainGrad, alpha=alpha)
    reconTestGrad = smoothGrads(grads=reconTestGrad, alpha=alpha)
    sumDklReconTrainGrad = smoothGrads(grads=sumDklReconTrainGrad, alpha=alpha)
    sumDklReconTestGrad = smoothGrads(grads=sumDklReconTestGrad, alpha=alpha)

    plt.figure(figsize=(15,7))

    plt.subplot(1+int(displayBeta), 3-int(displayBeta), 1)
    plt.plot(epochs, DklTrainGrad, label="Train")
    plt.plot(epochs, DklTestGrad, label="Test")
    plt.title("Dkl loss gradient (scaled, EMA)")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1+int(displayBeta), 3-int(displayBeta), 2)
    plt.plot(epochs, reconTrainGrad, label="Train")
    plt.plot(epochs, reconTestGrad, label="Test")
    plt.title("Reconstruction loss gradient (scaled, EMA)")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1+int(displayBeta), 3-int(displayBeta), 3)
    plt.plot(epochs, sumDklReconTrainGrad, label="Train")
    plt.plot(epochs, sumDklReconTestGrad, label="Test")
    plt.title("Dkl + Recon gradients (scaled, EMA)")
    plt.xlabel("Epochs")
    plt.legend()

    if displayBeta:
        plt.subplot(1+int(displayBeta), 3-int(displayBeta), 4)
        plt.plot(epochs, betas)
        plt.title("Beta Value")
        plt.xlabel("Epochs")
    
    plt.suptitle(t=title)
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

def plotDiffusionLoss(results, log: bool=False, step: int=1, title: str=""):
    difTrainLoss = results["train_loss"]
    difTestLoss = results["test_loss"]

    epochs = range(step, len(difTrainLoss) + 1, step)

    if step > 1:
        difTrainLoss = [difTrainLoss[i] for i in range(step-1, len(difTrainLoss), step)]
        difTestLoss = [difTestLoss[i] for i in range(step-1, len(difTestLoss), step)]

    if log:
        difTrainLoss = np.log(np.array(difTrainLoss))
        difTestLoss = np.log(np.array(difTestLoss))

    plt.figure(figsize=(15,7))
    plt.plot(epochs, difTrainLoss, label="Train")
    plt.plot(epochs, difTestLoss, label="Test")
    plt.title(title + "Loss" + (" (log)" if log else ""))
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()