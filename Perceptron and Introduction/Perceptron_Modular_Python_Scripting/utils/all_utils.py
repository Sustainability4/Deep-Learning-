import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def prepare_data(df, target_column = "y"):
    X = df.drop(target_column, axis =1)
    y = df[target_column]

    return X,y

def save_plot(df,model, filename = "plot.png", plot_dir = "plots"):
    def _create_base_plot(df):
        df.plot(kind = "scatter", x = "x1", y = "x2", c = "y", s = 100, cmap = "coolwarm")
        plt.axhline(y = 0, color = "black", linestyle = "--", linewidth = 1)
        plt.axvline(x = 0, color = "black", linestyle = "--", linewidth = 1)

        fig = plt.gcf()
        fig.set_size_inches(10,8)


    def _plot_decision_regions(X,y,classifier, resolution = 0.02):
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)

        X = X.values # as an array
        x1 = X[:,0]
        x2 = X[:,1]

        x1_min, x1_max = x1.min()-1, x1.max()+1
        x2_min, x2_max = x2.min()-1, x2.max()+1

        # creating the mashgrid 
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                np.arange(x2_min,x2_max,resolution))

        # predict the y_hat value 
        # ravel will flatten the xx1 and xx2 matrix into an array
        # We took the transpose of it to convert it to column matrix 
        y_hat = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        # alpha is mentioned for transparency factor 
        plt.contourf(xx1,xx2, y_hat, alpha = 0.3, cmpa = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()

        
    X,y = prepare_data(df)

    # creating and internal function
    _create_base_plot(df)
    _plot_decision_regions(X,y,model)

    # create the directory
    os.makedirs(plot_dir, exist_ok = True)

    # create the plot file path
    plot_path = os.path.join(plot_dir,filename)
    plt.savefig(plot_path)
