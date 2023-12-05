import matplotlib.pyplot as plt

class BaseClass:
    def __init__(self,):
        self.fig = None 
        self.ax = None 
        
    def loss_plot(self,epoch:int,gloss:float,dloss:float,title:str,label_loss:str,last_epoch:int,sign:str,gcolor="green",
                  dcolor='red'):
         # Create a new figure if needed
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

        # Add the current loss to the plot
        self.fig.suptitle(title)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel(label_loss)
        if epoch == 1:
            self.ax.plot(epoch, gloss, sign,color=gcolor,label="Generator")
            self.ax.plot(epoch,dloss,sign,color=dcolor,label="Discriminator")
            plt.legend()

        else:
            self.ax.plot(epoch,gloss,sign,color=gcolor)
            self.ax.plot(epoch,dloss,sign,color=dcolor)

        # Update the plot
        # Add a legend

        plt.draw()
        plt.pause(0.01)
        if epoch+1 == last_epoch:
            plt.show()

