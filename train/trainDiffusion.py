import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from models.diffusion import NoiseScheduler, sample

def trainStep(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn: torch.nn.Module,
              noiseScheduler: NoiseScheduler,
              device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> float:
    model.train()

    trainLoss = 0
    timesteps = noiseScheduler.timesteps

    for x, _ in dataloader:
        x = x.to(device)
        batchSize = x.size(0)


        # sample random batch of t in [0,T]
        t = torch.randint(0, timesteps, (batchSize,), device=device, dtype=torch.int64)

        # sample noise ~ N(0,1)
        noise = torch.randn_like(x)

        # Compute xt
        alphahatt = noiseScheduler.getNoiseLevel(t).view(batchSize, 1, 1, 1)
        xt = (torch.sqrt(alphahatt) * x) + (torch.sqrt(1 - alphahatt) * noise)

        # Forward pass xt, t through model
        predNoise = model(xt, t)

        # Loss
        loss = lossFn(predNoise, noise)
        trainLoss += loss.item()

        # Backpropagation and gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainLoss /= len(dataloader)

    return trainLoss

def testStep(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             noiseScheduler: NoiseScheduler,
             device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> float:
    model.eval()

    testLoss = 0
    timesteps = noiseScheduler.timesteps

    with torch.inference_mode():
        for x, _ in dataloader:
            x = x.to(device)
            batchSize = x.size(0)


            # sample random batch of t in [0,T]
            t = torch.randint(0, timesteps, (batchSize,), device=device, dtype=torch.int64)

            # sample noise ~ N(0,1)
            noise = torch.randn_like(x)

            # Compute xt
            alphahatt = noiseScheduler.getNoiseLevel(t).view(batchSize, 1, 1, 1)
            xt = (torch.sqrt(alphahatt) * x) + (torch.sqrt(1 - alphahatt) * noise)

            # Forward pass xt, t through model
            predNoise = model(xt, t)

            # Loss
            loss = lossFn(predNoise, noise)
            testLoss += loss.item()

    testLoss /= len(dataloader)

    return testLoss

def train(model: torch.nn.Module,
          trainDataloader: torch.utils.data.DataLoader,
          testDataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          lossFn: torch.nn.Module,
          noiseScheduler: NoiseScheduler,
          epochs: int=5,
          results: Dict[str, list] | None=None,
          numGeneratedSamples: int=0,
          imgShape: Tuple[int, int, int] | None=None, # (C,H,W)
          sampleEta: float=1.0,
          seed: int=42,
          device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, list]:
    model.to(device)
    
    initialEpoch = 1
    if results is not None:
        initialEpoch += len(results["train_loss"])
    else:
        results = {"train_loss": [],
                   "test_loss": [],
                   "generated_samples": []}
        
    torch.manual_seed(seed)
    if numGeneratedSamples > 0:
        assert imgShape is not None, "imgShape for generated samples cannot be None."
        xTbatch = torch.randn(numGeneratedSamples, imgShape[0], imgShape[1], imgShape[2], device=device)

    for epoch in tqdm(range(epochs)):
        torch.manual_seed(seed+epoch)
        trainLoss = trainStep(model=model,
                              dataloader=trainDataloader,
                              optimizer=optimizer,
                              lossFn=lossFn,
                              noiseScheduler=noiseScheduler,
                              device=device)
        results["train_loss"].append(trainLoss)

        testLoss = testStep(model=model,
                            dataloader=testDataloader,
                            lossFn=lossFn,
                            noiseScheduler=noiseScheduler,
                            device=device)
        results["test_loss"].append(testLoss)

        if numGeneratedSamples > 0:
            # torch.manual_seed(seed)
            model.eval()
            x0 = sample(model=model,
                        noiseScheduler=noiseScheduler,
                        xT=xTbatch,
                        skip=1,
                        eta=sampleEta,
                        device=device)
            results["generated_samples"].append(x0)
            

        print(f"\nEpochs: {epoch+initialEpoch} | (Train) Loss: {trainLoss:.4f} | (Test) Loss: {testLoss:.4f}")
    
    return results