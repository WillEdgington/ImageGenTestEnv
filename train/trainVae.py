import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from models.vae import vaeLoss, VAE

def trainStep(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              beta: float=1.0,
              device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float, float]:
    model.train()
    
    # Setup train loss values
    trainLoss, DklTrainLoss, reconTrainLoss = 0, 0, 0

    for x, _ in dataloader:
        x.to(device)

        # Forward pass through model
        xhat, mu, logvar = model(x)

        # Get loss
        loss, reconLoss, DklLoss = vaeLoss(xhat=xhat, x=x, mu=mu, logvar=logvar, beta=beta)
        trainLoss += loss.item()
        DklTrainLoss += DklLoss.item()
        reconTrainLoss += reconLoss.item()

        # Perform backpropagation and gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainLoss /= len(dataloader)
    DklTrainLoss /= len(dataloader)
    reconTrainLoss /= len(dataloader)

    return trainLoss, DklTrainLoss, reconTrainLoss

def testStep(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             beta: float=1.0,
             device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float, float]:
    model.eval()

    # Setup train loss values 
    testLoss, DklTestLoss, reconTestLoss = 0, 0, 0

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

    testLoss /= len(dataloader)
    DklTestLoss /= len(dataloader)
    reconTestLoss /= len(dataloader)

    return testLoss, DklTestLoss, reconTestLoss

def train(model: VAE,
          trainDataloader: torch.utils.data.DataLoader,
          testDataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          beta: float=1.0,
          device: torch.device="cuda" if torch.cuda.is_available() else "cpu",
          latentDim: int=100,
          decSamplesPerEpoch: int=0) -> Dict[str, list]:
    assert decSamplesPerEpoch <= 10, f"genSamplesPerEpoch must be less than or equal to 10, currently it is {decSamplesPerEpoch}"

    results = {"train_loss": [],
               "Dkl_train_loss": [],
               "recon_train_loss": [],
               "test_loss": [],
               "Dkl_test_loss": [],
               "recon_test_loss": [],
               "decoder_samples": [] if decSamplesPerEpoch else None}
    
    # Create sample noise that will be used to generate images per epoch
    if decSamplesPerEpoch:
        latentVectors = torch.randn(decSamplesPerEpoch, latentDim).to(device)
    
    for epoch in tqdm(range(epochs)):
        # Run a training step
        trainLoss, DklTrainLoss, reconTrainLoss = trainStep(model=model,
                                                            dataloader=trainDataloader,
                                                            optimizer=optimizer,
                                                            beta=beta,
                                                            device=device)
        results["train_loss"].append(trainLoss)
        results["Dkl_train_loss"].append(DklTrainLoss)
        results["recon_train_loss"].append(reconTrainLoss)

        # Run a testing step
        testLoss, DklTestLoss, reconTestLoss = testStep(model=model,
                                                        dataloader=testDataloader,
                                                        beta=beta,
                                                        device=device)
        results["test_loss"].append(testLoss)
        results["Dkl_test_loss"].append(DklTestLoss)
        results["recon_test_loss"].append(reconTestLoss)

        # Generate samples from decoder
        if decSamplesPerEpoch:
            model.eval()
            with torch.inference_mode():
                decSample = model.decode(latentVectors).to("cpu")
        results["decoder_samples"].append(decSample)
        
        # Print epoch and loss values
        print(f"\nEpoch: {epoch+1} | (Train) Total loss: {trainLoss:.4f}, Dkl loss: {DklTrainLoss:.4f}, Reconstruction loss: {reconTrainLoss:.4f} | "
              f"(Test) Total loss: {testLoss:.4f}, Dkl loss: {DklTestLoss:.4f}, Reconstruction loss: {reconTestLoss:.4f}")
    
    return results