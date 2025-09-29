import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from models.vae import vaeLoss, VAE, BetaScheduler
from models.LDM import LDMVAE

def activeLatents(mu: torch.Tensor, threshold: float = 5e-2) -> tuple[int, float]:
    if mu.dim() == 2:
        std = mu.std(dim=0)
    elif mu.dim() == 4:
        std = mu.view(mu.size(0), mu.size(1), -1).mean(dim=2).std(dim=0)
    else:
        raise ValueError(f"Unsupported shape for mu: {mu.shape}")
    activeDims = (std > threshold).sum().item()
    return int(activeDims), float(std.mean().item())

def trainStep(model: VAE|LDMVAE, 
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              beta: float=1.0,
              countActiveDims: bool=False,
              enableAmp: bool=False,
              gradClipping: None|float=None,
              device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float, float, float | None, float | None]:
    model.train()
    
    # Setup train loss values
    trainLoss, DklTrainLoss, reconTrainLoss = 0, 0, 0

    if countActiveDims:
        activeDims, latentStd = 0, 0
    
    scaler = torch.amp.GradScaler(device=device, enabled=(device=="cuda" and enableAmp))

    for x, _ in dataloader:
        x = x.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass through model and get loss
        with torch.amp.autocast(device_type=device, enabled=(device=="cuda" and enableAmp)):
            xhat, mu, logvar = model(x)
            loss, reconLoss, DklLoss = vaeLoss(xhat=xhat, x=x, mu=mu, logvar=logvar, beta=beta)
        
        # track loss
        trainLoss += loss.item()
        DklTrainLoss += DklLoss.item()
        reconTrainLoss += reconLoss.item()

        # Get active dimensions in latent space z
        if countActiveDims:
            dims, std = activeLatents(mu)
            latentStd += std
            activeDims += dims

        # Perform backpropagation and gradient descent
        scaler.scale(loss).backward()
        if gradClipping is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradClipping)
        scaler.step(optimizer)
        scaler.update()

    trainLoss /= len(dataloader)
    DklTrainLoss /= len(dataloader)
    reconTrainLoss /= len(dataloader)

    if countActiveDims:
        activeDims /= len(dataloader)
        latentStd /= len(dataloader)
        return trainLoss, DklTrainLoss, reconTrainLoss, activeDims, latentStd
    
    return trainLoss, DklTrainLoss, reconTrainLoss, None, None

def testStep(model: VAE|LDMVAE,
             dataloader: torch.utils.data.DataLoader,
             beta: float=1.0,
             countActiveDims: bool=False,
             enableAmp: bool=False,
             device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float, float, float | None, float | None]:
    model.eval()

    # Setup train loss values 
    testLoss, DklTestLoss, reconTestLoss = 0, 0, 0

    if countActiveDims:
        activeDims, latentStd = 0, 0

    with torch.inference_mode():
        for x, _ in dataloader:
            x = x.to(device)

            # Forward pass through model and get loss
            with torch.amp.autocast(device_type=device, enabled=(device=="cuda" and enableAmp)):
                xhat, mu, logvar = model(x)
                loss, reconLoss, DklLoss = vaeLoss(xhat=xhat, x=x, mu=mu, logvar=logvar, beta=beta)

            # track loss
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

def train(model: VAE|LDMVAE,
          trainDataloader: torch.utils.data.DataLoader,
          testDataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          beta: float=1.0,
          enableAmp: bool=False,
          seed: int=42,
          device: torch.device="cuda" if torch.cuda.is_available() else "cpu",
          latentDim: int=100,
          decSamplesPerEpoch: int=0,
          countActiveDims: bool=False,
          betaScheduler: BetaScheduler | None=None,
          results: Dict[str, list] | None=None) -> Dict[str, list]:
    assert decSamplesPerEpoch <= 10, f"genSamplesPerEpoch must be less than or equal to 10, currently it is {decSamplesPerEpoch}"
    model.to(device)
    torch.manual_seed(seed)

    initialEpoch = 1
    if results is not None:
        initialEpoch += len(results["train_loss"])

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
        latentShape = latentDim if not model.latentChannels else (model.latentChannels, latentDim, latentDim)
        latentVectors = torch.randn(decSamplesPerEpoch, *latentShape).to(device)
    
    for epoch in tqdm(range(epochs)):
        torch.manual_seed(seed + epoch + initialEpoch)
        # Run a training step
        trainLoss, DklTrainLoss, reconTrainLoss, activeDimsTrain, latentStdTrain = trainStep(model=model,
                                                                                             dataloader=trainDataloader,
                                                                                             optimizer=optimizer,
                                                                                             beta=beta,
                                                                                             countActiveDims=countActiveDims,
                                                                                             enableAmp=enableAmp,
                                                                                             device=device)
        results["train_loss"].append(trainLoss)
        results["Dkl_train_loss"].append(DklTrainLoss)
        results["recon_train_loss"].append(reconTrainLoss)

        # Run a testing step
        testLoss, DklTestLoss, reconTestLoss, activeDimsTest, latentStdTest = testStep(model=model,
                                                                                       dataloader=testDataloader,
                                                                                       beta=beta,
                                                                                       countActiveDims=countActiveDims,
                                                                                       enableAmp=enableAmp,
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
        print(f"\nEpoch: {initialEpoch+epoch} | (Train) Total loss: {trainLoss:.4f}, Dkl loss: {DklTrainLoss:.4f}, Reconstruction loss: {reconTrainLoss:.4f} | "
              f"(Test) Total loss: {testLoss:.4f}, Dkl loss: {DklTestLoss:.4f}, Reconstruction loss: {reconTestLoss:.4f}")
        if countActiveDims:
            print(f"(Active latent dimensions) Train: {activeDimsTrain:.1f}, Test: {activeDimsTest:.1f} | (Mean std of latent dimensions) Train: {latentStdTrain:.4f}, Test: {latentStdTest:.4f}")

        if betaScheduler:
            print(f"Updated Beta: {beta:.4f}")
    
    return results