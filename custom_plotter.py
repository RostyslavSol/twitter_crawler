import numpy as np
import matplotlib.pyplot as plt


class CustomPlotter(object):
    @staticmethod
    def plot(sample_count_arr, cluster_names_hash, save_filename=None,  color=None):
        N = len(sample_count_arr)

        ind = np.arange(N)  # the x locations for the groups
        #TODO: calculate width
        width = 0.35       # the width of the bars

        fig, ax = plt.subplots()
        color = color if color is not None else 'green'
        rects = ax.bar(ind, sample_count_arr, width, color=color)

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Sample count')
        title_apendix = 'LSA' if color == 'blue' else 'NB'
        ax.set_title('Ratings for clusters ' + title_apendix)
        ax.set_xticks(ind + width)
        ax.set_xticklabels(['Cluster #' + str(i+1) + ' ' + cluster_names_hash[str(i)] for i in range(N)])
        
        #ax.legend([rects[0], ['Cluster #1', 'Cluster #2'])
        
        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%d' % int(height),
                        ha='center', va='bottom')
        
        autolabel(rects)

        if save_filename is not None:
            save_filename = save_filename if '.png' in save_filename else save_filename + '.png'
            plt.savefig(save_filename)
        plt.show()
