import torch
import sys

from torchinfo import summary
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data import prepareData
from models.vae import VAE, AdaptiveMomentBetaScheduler
from train.trainVae import train
from utils.save import saveModelAndResultsMap, loadResultsMap, loadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

MANUALSEED = 42
EPOCHS = 100
SAVEPOINT = 50

if __name__=="__main__":
    # Create experiments
    betas = [1.0, 4.0]
    useScheduler = [False, True]
    latentDims = [50, 100]
    batchSizes = [32]
    learningRates = [1e-4]
    extraConvsArr = [0, 5]

    # iterate through experiments
    for beta in betas:
        for scheduler in useScheduler:
            if scheduler and beta == 4.0:
                continue
            for latentDim in latentDims:
                for extraConvs in extraConvsArr:
                    epochsComplete = 0
                    adaptiveStr = "_ADAPTIVE_" if scheduler else ""
                    modelName = f"VAE_CIFAR10_{int(beta)}_BETA{adaptiveStr}_{latentDim}_ZDIMS_{extraConvs}_CONVS_"
                    resultsName = f"VAE_CIFAR10_{int(beta)}_BETA{adaptiveStr}_{latentDim}_ZDIMS_{extraConvs}_CONVS_RESULTS.pth"

                    VAEresults = loadResultsMap(resultsName=resultsName)
                    epochsComplete = 0 if VAEresults is None else len(VAEresults["train_loss"])
                    if epochsComplete == EPOCHS: continue

                    # create CIFAR-10 dataloaders
                    trainDataloader = prepareData(batchSize=batchSizes[0], seed=MANUALSEED)
                    testDataloader = prepareData(train=False, batchSize=batchSizes[0])
                    
                    torch.manual_seed(MANUALSEED)
                    if epochsComplete == 0:
                        vae = VAE(latentDim=latentDim, upAddConv=extraConvs, downAddConv=extraConvs)
                    else:
                        vae = loadModel(model=VAE(latentDim=latentDim, upAddConv=extraConvs, downAddConv=extraConvs),
                                        modelName=modelName + f"{epochsComplete}_EPOCHS_MODEL.pth")
                    vae.to(device)

                    betaScheduler = AdaptiveMomentBetaScheduler(betaInit=beta) if scheduler else None

                    print(f"TRIAL:\nBeta: {beta}, scheduler?: {scheduler}, latent dimensions: {latentDim}, extras convs: {extraConvs}\n\n")

                    # Get summary of model
                    summary(model=vae,
                        input_size=(1, 3, 32, 32),
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])
                    
                    # Define the optimizer
                    optimizer = torch.optim.Adam(vae.parameters(), lr=learningRates[0])
                    while epochsComplete != EPOCHS:
                        # run training
                        torch.manual_seed(MANUALSEED)
                        VAEresults = train(model=vae,
                                        trainDataloader=trainDataloader,
                                        testDataloader=testDataloader,
                                        optimizer=optimizer,
                                        epochs=SAVEPOINT,
                                        beta=beta if not scheduler else betaScheduler.beta,
                                        device=device,
                                        latentDim=latentDim,
                                        decSamplesPerEpoch=5,
                                        countActiveDims=True,
                                        betaScheduler=betaScheduler,
                                        results=VAEresults)
                        
                        epochsComplete = len(VAEresults["train_loss"])

                        # Save the results and model
                        saveModelAndResultsMap(model=vae,
                                            results=VAEresults, 
                                            modelName=modelName + f"{epochsComplete}_EPOCHS_MODEL.pth",
                                            resultsName=resultsName)