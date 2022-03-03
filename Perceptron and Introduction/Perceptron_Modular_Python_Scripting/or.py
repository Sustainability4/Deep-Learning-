from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd
import logging
import os


log_dir = "logs"
os.makedirs(log_dir, exist_ok = True)
# setting up the basic configuration for logging
# various levels to logging : info, debug, warnings 
# filemode has been mentioned as a which means append as otherwise it will overwrite the logs
# once you have set the logging information here, you don't have to set the info in the utils file 
# you can simply call logging there an duse it. 
logging.basicConfig(
    filename = os.path.join(log_dir, "running.log"),
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
    )


def main(data, modelName, plotName, eta, epochs):
    # docstring is nothing but extra information added below your function 
    # When we use python libraries help feature to get information on the functions we get the docstring text back 
    """ This is an example of a docstring : This function is called training the perceptron for OR gate
    
    Args provided : data, modelName, plotName, eta, epoch 
    
    returns : This function returns nothing it saves your plot and model in the desired location. 
    """
    df_OR = pd.DataFrame(data)
    
    logging.info(f"this is the raw dataset {df_OR}")
    X, y = prepare_data(df_OR)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    model.save(filename=modelName, model_dir="model")
    save_plot(df_OR, model, filename=plotName)

    
# Entry Point where your code will start
if __name__ == "__main__":
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,1]
    }
    ETA = 0.3
    EPOCHS = 10
    try:
        logging.info(">>>>>>>starting training>>>>>>>>>")
        main(data=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)
        logging.info(">>>>>>>done training>>>>>>>>")
    except e:
        logging.exception(e)
        raise e
        
    
