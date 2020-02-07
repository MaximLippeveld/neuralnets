import matplotlib.pyplot as plt, seaborn, numpy
from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    fig, ax = plt.subplots()
    ave_grads = []
    max_grads= []
    layers = []
    for n, me, ma in named_parameters:
        layers.append(n)
        ave_grads.append(me)
        max_grads.append(ma)
    ax.bar(numpy.arange(len(max_grads))+0.5, max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(numpy.arange(len(max_grads))+0.5, ave_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.set_xticks(numpy.arange(0,len(ave_grads))+0.5)
    ax.set_xticklabels(layers)
    ax.tick_params(axis="x", labelrotation=-90.0)
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom = -0.001, top=0.05) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(False)
    ax.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.tight_layout()
    return fig