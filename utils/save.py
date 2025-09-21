import torch
import os

from pathlib import Path

def saveTorchObject(obj, targetDir: str, fileName: str):
    targetDirPath = Path(targetDir)
    targetDirPath.mkdir(parents=True, exist_ok=True)
    
    assert fileName.endswith(".pth") or fileName.endswith(".pt"), "fileName must end with '.pt' or '.pth'."
    fileSavePath = targetDirPath / fileName

    print(f"Saving torch object to: {fileSavePath}")
    torch.save(obj=obj,
               f=fileSavePath)
    
def loadTorchObject(targetDir: str, fileName: str, device="cpu"):
    targetDirPath = Path(targetDir)
    fileLoadPath = targetDirPath / fileName

    return torch.load(fileLoadPath, map_location=device)

def saveModelAndResultsMap(model: torch.nn.Module|dict, results: dict, modelName: str, resultsName: str, modelDir: str="saved_models", resultsDir: str="saved_results"):
    # Save model weights
    if not isinstance(model, dict):
        model = model.state_dict()

    saveTorchObject(obj=model,
                    targetDir=modelDir,
                    fileName=modelName)
    
    # Save results dictionary
    saveTorchObject(obj=results,
                    targetDir=resultsDir,
                    fileName=resultsName)
    
def saveGANandResultsMap(generator: torch.nn.Module, discriminator: torch.nn.Module, results: dict, modelName: str, resultsName: str,
                         modelDir: str="saved_models", resultsDir: str="saved_results"):
    # Save generator weights
    saveTorchObject(obj=generator.state_dict(),
                    targetDir=modelDir,
                    fileName="GENERATOR_" + modelName)
    
    # Save discriminator weights
    saveTorchObject(obj=discriminator.state_dict(),
                    targetDir=modelDir,
                    fileName="DISCRIMINATOR_" + modelName)
    
    # Save results dictionary
    saveTorchObject(obj=results,
                    targetDir=resultsDir,
                    fileName=resultsName)
    
def loadModel(model: torch.nn.Module, modelName: str, modelDir: str="saved_models",
              device="cuda" if torch.cuda.is_available() else "cpu"):
    model.load_state_dict(loadTorchObject(targetDir=modelDir, fileName=modelName, device=device))

    return model

def loadResultsMap(resultsName: str, resultsDir: str="saved_results", device="cpu"):
    print(f"Trying to load: {resultsDir}/{resultsName}")
    try:
        result = loadTorchObject(targetDir=resultsDir, fileName=resultsName, device=device)
        return result
    except:
        return None
    
def loadStates(stateName: str, stateDir: str="saved_models", **kwargs):
    try:
        state = loadTorchObject(targetDir=stateDir, fileName=stateName, device="cpu")
    except:
        return None
    
    for k, v in kwargs.items():
        if k not in state.keys():
            continue
        v.load_state_dict(state[k])
    
    return state
