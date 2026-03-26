import os

## globals
SEQLEN = 8
MODELNAME = "RPPgaussianGaussian"#"RPPgaussianGaussianCov" # set the name of the model you want to train  here

############################################### HPE ######################################################################

if MODELNAME in ("RPPgaussianGaussian", "RPPgaussianGaussianCov"):
  TRAINCONFIG = {"learningRate": 0.001, 
                  "weightDecay": 0.0001,
                  "epochs": 24, 
                  "batchSize": 16, 
                  "optimizer": "adam", 
                  "device": "cuda", 
                  "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
                  "lrDecay": 0.99, 
                  "beta": 20, 
                  "gamma": 5,  
                  "nll": True,
                  "evd": False,
                  "nf": False,
                  "testing_epoch": None  # Set to specific epoch number or None for latest
            }

if MODELNAME in ("RPPlaplaceLaplace"):
  TRAINCONFIG = {"learningRate": 0.001, 
                 "weightDecay": 0.0001,
                 "epochs": 24, 
                 "batchSize": 16, 
                 "optimizer": "adam", 
                 "device": "cuda", 
                 "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
                 "lrDecay": 0.99, 
                 "beta": 1, 
                 "gamma": 1, 
                 "nll": True,
                 "evd": False,
                 "nf": False,
                 "testing_epoch": None  # Set to specific epoch number or None for latest
            }
  
if MODELNAME in ("RPPlaplaceGaussian"):
  TRAINCONFIG = {"learningRate": 0.001, 
                 "weightDecay": 0.0001,
                 "epochs": 24, 
                 "batchSize": 16, 
                 "optimizer": "adam", 
                 "device": "cuda", 
                 "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
                 "lrDecay": 0.99, 
                 "beta": 1, 
                 "gamma": 5, 
                 "nll": True,
                 "evd": False,
                 "nf": False,
                 "testing_epoch": None  # Set to specific epoch number or None for latest
            }
  
if MODELNAME in ("RPPgaussianLaplace"):
  TRAINCONFIG = {"learningRate": 0.001, 
                 "weightDecay": 0.0001,
                 "epochs": 24, 
                 "batchSize": 16, 
                 "optimizer": "adam", 
                 "device": "cuda", 
                 "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
                 "lrDecay": 0.99, 
                 "beta": 20, 
                 "gamma": 5, 
                 "nll": True,
                 "evd": False,
                 "nf": False,
                 "testing_epoch": None  # Set to specific epoch number or None for latest
            }

        
if MODELNAME in ("RPPevidential"):
  TRAINCONFIG = {"learningRate": 0.0001, 
            "weightDecay": 0.0001,
            "epochs": 24, 
            "batchSize": 16, 
            "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
            "lambda":0.001, # 0.001
            "nll": False,
            "evd": True,
            "nf": False,
            "testing_epoch": None  # Set to specific epoch number or None for latest
            }


if MODELNAME in ("RPPnormalizingFlow"):
  TRAINCONFIG = {"learningRate": 0.0001, 
            "weightDecay": 0.0001,
            "epochs": 24, 
            "batchSize": 32, 
            "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
            "gamma": 1,
            "beta": 1,
            "nll": False,
            "evd": False, 
            "nf": True,
            "testing_epoch": None  # Set to specific epoch number or None for latest
            }

if MODELNAME in ("HoEtAlBaseline"):
  TRAINCONFIG = {"learningRate": 0.0001, 
            "weightDecay": 0.0001,
            "epochs": 40, 
            "batchSize": 16, #16 #64 
            "optimizer": "adam", 
            "device": "cuda", 
            "betas": (0.9, 0.999), # momentum and scaling for ADAM, 
            "lrDecay": 0.99, 
            "nll": False,
            "evd": False,
            "nf": False, 
            "HR_SINGLE": True,
            "testing_epoch": None  # Set to specific epoch number or None for latest
            }



# paths
PATHORIGIN = None  # Set this to your root path
PATHRAW = None  # Set this to your raw radar data path

# Validate paths before using them
if PATHORIGIN is None:
    raise ValueError("PATHORIGIN must be set in config.py")
if PATHRAW is None:
    raise ValueError("PATHRAW must be set in config.py")

MODELPATH = os.path.join(PATHORIGIN, "models")
HPECKPT = os.path.join(PATHORIGIN, "trainedModels", "humanPoseEstimation")





