import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from models.vae import vaeLoss, VAE, BetaScheduler

def activeLatents(mu: torch.Tensor, threshold: float = 5e-2) -> tuple[int, float]:
    std = mu.std(dim=0)
    activeDims = (std > threshold).sum().item()
    return int(activeDims), float(std.mean().item())

def trainStep(model: VAE, 
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              beta: float=1.0,
              countActiveDims: bool=False,
              device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float, float, float | None, float | None]:
    model.train()
    
    # Setup train loss values
    trainLoss, DklTrainLoss, reconTrainLoss = 0, 0, 0

    if countActiveDims:
        activeDims, latentStd = 0, 0

    for x, _ in dataloader:
        x = x.to(device)

        # Forward pass through model
        xhat, mu, logvar = model(x)

        # Get loss
        loss, reconLoss, DklLoss = vaeLoss(xhat=xhat, x=x, mu=mu, logvar=logvar, beta=beta)
        trainLoss += loss.item()
        DklTrainLoss += DklLoss.item()
        reconTrainLoss += reconLoss.item()

        # Get active dimensions in latent space z
        if countActiveDims:
            dims, std = activeLatents(mu)
            latentStd += std
            activeDims += dims

        # Perform backpropagation and gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainLoss /= len(dataloader)
    DklTrainLoss /= len(dataloader)
    reconTrainLoss /= len(dataloader)

    if countActiveDims:
        activeDims /= len(dataloader)
        latentStd /= len(dataloader)
        return trainLoss, DklTrainLoss, reconTrainLoss, activeDims, latentStd
    
    return trainLoss, DklTrainLoss, reconTrainLoss, None, None

def testStep(model: VAE,
             dataloader: torch.utils.data.DataLoader,
             beta: float=1.0,
             countActiveDims: bool=False,
             device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float, float, float | None, float | None]:
    model.eval()

    # Setup train loss values 
    testLoss, DklTestLoss, reconTestLoss = 0, 0, 0

    if countActiveDims:
        activeDims, latentStd = 0, 0

    with torch.inference_mode():
        for x, _ in dataloader:
            x.to(device)

            # Forward pass through model
            xhat, mu, logvar = model(x)

            # Get loss
            loss, reconLoss, DklLoss = vaeLoss(xhat=xhat, x=x, mu=mu, logvar=logvar, beta=beta)
            testLoss += loss.item()
            DklTestLoss += DklLoss.item()
            reconTestLoss += reconLoss.item()

            # Get active dimensions in latent space z
            if countActiveDims:
                dims, std = activeLatents(mu)
                latentStd += std
                activeDims += dims

    testLoss /= len(dataloader)
    DklTestLoss /= len(dataloader)
    reconTestLoss /= len(dataloader)

    if countActiveDims:
        activeDims /= len(dataloader)
        latentStd /= len(dataloader)
        return testLoss, DklTestLoss, reconTestLoss, activeDims, latentStd

    return testLoss, DklTestLoss, reconTestLoss, None, None

def train(model: VAE,
          trainDataloader: torch.utils.data.DataLoader,
          testDataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          beta: float=1.0,
          device: torch.device="cuda" if torch.cuda.is_available() else "cpu",
          latentDim: int=100,
          decSamplesPerEpoch: int=0,
          countActiveDims: bool=False,
          betaScheduler: BetaScheduler | None=None,
          results: Dict[str, list] | None=None) -> Dict[str, list]:
    assert decSamplesPerEpoch <= 10, f"genSamplesPerEpoch must be less than or equal to 10, currently it is {decSamplesPerEpoch}"
    model.to(device)

    if not results:
        results = {"train_loss": [],
                   "Dkl_train_loss": [],
                   "recon_train_loss": [],
                   "test_loss": [],
                   "Dkl_test_loss": [],
                   "recon_test_loss": [],
                   "beta_value": [],
                   "decoder_samples": [],
                   "active_latent_dims_train": [],
                   "active_latent_dims_test": [],
                   "latent_dims_std_train": [],
                   "latent_dims_std_test": []}
    
    # Create sample noise that will be used to generate images per epoch
    if decSamplesPerEpoch:
        latentVectors = torch.randn(decSamplesPerEpoch, latentDim).to(device)
    
    for epoch in tqdm(range(epochs)):
        # Run a training step
        trainLoss, DklTrainLoss, reconTrainLoss, activeDimsTrain, latentStdTrain = trainStep(model=model,
                                                                                             dataloader=trainDataloader,
                                                                                             optimizer=optimizer,
                                                                                             beta=beta,
                                                                                             countActiveDims=countActiveDims,
                                                                                             device=device)
        results["train_loss"].append(trainLoss)
        results["Dkl_train_loss"].append(DklTrainLoss)
        results["recon_train_loss"].append(reconTrainLoss)

        # Run a testing step
        testLoss, DklTestLoss, reconTestLoss, activeDimsTest, latentStdTest = testStep(model=model,
                                                                                       dataloader=testDataloader,
                                                                                       beta=beta,
                                                                                       countActiveDims=countActiveDims,
                                                                                       device=device)
        results["test_loss"].append(testLoss)
        results["Dkl_test_loss"].append(DklTestLoss)
        results["recon_test_loss"].append(reconTestLoss)

        if countActiveDims:
            results["active_latent_dims_train"].append(activeDimsTrain)            
            results["latent_dims_std_train"].append(latentStdTrain)
            results["active_latent_dims_test"].append(activeDimsTest)
            results["latent_dims_std_test"].append(latentStdTest)

        results["beta_value"].append(beta)
        if betaScheduler:
            beta = betaScheduler.update(Dkl=DklTrainLoss, recon=reconTrainLoss)

        # Generate samples from decoder
        if decSamplesPerEpoch:
            model.eval()
            with torch.inference_mode():
                decSample = model.decode(latentVectors).to("cpu")
        results["decoder_samples"].append(decSample)
        
        # Print epoch and loss values
        print(f"\nEpoch: {epoch+1} | (Train) Total loss: {trainLoss:.4f}, Dkl loss: {DklTrainLoss:.4f}, Reconstruction loss: {reconTrainLoss:.4f} | "
              f"(Test) Total loss: {testLoss:.4f}, Dkl loss: {DklTestLoss:.4f}, Reconstruction loss: {reconTestLoss:.4f}")
        if countActiveDims:
            print(f"(Active latent dimensions) Train: {activeDimsTrain:.1f}, Test: {activeDimsTest:.1f} | (Mean std of latent dimensions) Train: {latentStdTrain:.4f}, Test: {latentStdTest:.4f}")

        if betaScheduler:
            print(f"Updated Beta: {beta:.4f}")
    
    return results