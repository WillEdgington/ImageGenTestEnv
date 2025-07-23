import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def trainStep(generator: torch.nn.Module,
              discriminator: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              optimG: torch.optim.Optimizer,
              optimD: torch.optim.Optimizer,
              latentDim: int=100,
              device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float,float]:
    discriminator.train()
    generator.train()

    # Setup train loss values
    disTrainLoss, genTrainLoss = 0, 0

    for realImgs, _ in dataloader:
        realImgs = realImgs.to(device)
        batchSize = realImgs.size(0)

        # Real and fake labels reference batches
        realLabels = torch.ones(batchSize, 1).to(device)
        fakeLabels = torch.zeros(batchSize, 1).to(device)
        
        # Generate fake images
        noise = torch.randn(batchSize, latentDim, 1, 1).to(device)
        fakeImgs = generator(noise)

        # Send fake and real images through discriminator
        realOut = discriminator(realImgs)
        fakeOut = discriminator(fakeImgs.detach())

        # Calculate discriminator loss
        realLossDis = lossFn(realOut, realLabels)
        fakeLossDis = lossFn(fakeOut, fakeLabels)
        disLoss = realLossDis + fakeLossDis
        disTrainLoss += disLoss.item()

        # Perform backpropagation and grad descent on discriminator
        optimD.zero_grad()
        disLoss.backward()
        optimD.step()

        # Calculate generator loss
        fakeOut = discriminator(fakeImgs) # re-run fakeImgs through discriminator
        genLoss = lossFn(fakeOut, realLabels)
        genTrainLoss += genLoss.item()

        # Perform backpropagation and grad descent on generator
        optimG.zero_grad()
        genLoss.backward()
        optimG.step()

    disTrainLoss /= len(dataloader)
    genTrainLoss /= len(dataloader)

    return disTrainLoss, genTrainLoss

def testStep(generator: torch.nn.Module,
             discriminator: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             latentDim: int=100,
             device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float]:
    generator.eval()
    discriminator.eval()

    disTestLoss, genTestLoss = 0, 0

    with torch.inference_mode():
        for realImgs, _ in dataloader:
            realImgs.to(device)
            batchSize = realImgs.size(0)

            # Real and fake labels reference batches
            realLabels = torch.ones(batchSize, 1).to(device)
            fakeLabels = torch.zeros(batchSize, 1).to(device)

            # Generate fake images
            noise = torch.randn(batchSize, latentDim, 1, 1).to(device)
            fakeImgs = generator(noise)

            # Send fake and real images through discriminator
            realOut = discriminator(realImgs)
            fakeOut = discriminator(fakeImgs.detach())

            # Calculate discriminator loss
            realLossDis = lossFn(realOut, realLabels)
            fakeLossDis = lossFn(fakeOut, fakeLabels)
            disLoss = realLossDis + fakeLossDis
            disTestLoss += disLoss.item()

            # Calculate generator loss
            genLoss = lossFn(fakeOut, realLabels)
            genTestLoss += genLoss.item()

    disTestLoss /= len(dataloader)
    genTestLoss /= len(dataloader)

    return disTestLoss, genTestLoss 


def train(generator: torch.nn.Module,
          discriminator: torch.nn.Module,
          trainDataloader: torch.utils.data.DataLoader,
          testDataloader: torch.utils.data.DataLoader,
          optimD: torch.optim.Optimizer,
          optimG: torch.optim.Optimizer,
          lossFn: torch.nn.Module,
          epochs: int,
          latentDim: int=100,
          device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, List]:
    # Create results dictionary
    results = {"generator_train_loss": [],
               "generator_test_loss": [],
               "discriminator_train_loss": [],
               "discriminator_test_loss": []}
    
    generator.to(device)
    discriminator.to(device)



    for epoch in tqdm(range(epochs)):
        # Run a training step
        disTrainLoss, genTrainLoss = trainStep(generator=generator,
                                               discriminator=discriminator,
                                               dataloader=trainDataloader,
                                               lossFn=lossFn,
                                               optimG=optimG,
                                               optimD=optimD,
                                               latentDim=latentDim,
                                               device=device)
        
        # Test the models
        disTestLoss, genTestLoss = testStep(generator=generator,
                                            discriminator=discriminator,
                                            dataloader=testDataloader,
                                            lossFn=lossFn,
                                            latentDim=latentDim,
                                            device=device)
        
        # Print the epoch and loss values
        print(f"/nEpoch: {epoch+1} | (Discriminator) Train loss: {disTrainLoss:.4f}, Test loss: {disTestLoss:.4f} | "
              f"(Generator) Train loss: {genTrainLoss:.4f}, Test loss: {genTestLoss:.4f}")

    return results

