

"""
    Sample Run
    python plot/plot_visibility_uncertainty_as_bar_for_aflw_ours.py 

    Plots visibility uncertainty as bar for aflw_ours dataset.

    Version 2 2019-11-06 Abhinav Kumar (Support for multiple models, root as 2,4)
    Version 1 2019-07-29 Abhinav Kumar 
"""
import matplotlib.pyplot as plt
import numpy as np

def write_text_above_bar(bar1, hide_first= False):
    for i, rect in enumerate(bar1):
        if hide_first and i == 0:
            pass
        else:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' %height, ha='center', va='bottom', size= fs-4)
        
#===============================================================================
# Plotting Options
#===============================================================================
fs = 20
plt.rcParams.update({'font.size': fs})
lw = 2.5
width     = 0.45
bar_width = 0.43

color1 = (0.27,0.8,0.8) #<--- this is my_purple
color2 = "dodgerblue"
color3 = (1, 0.45, 0.45) #<-- this is red!55
color4 = (1, 0.75, 0.  ) #<-- this is yellow
color5 = 'cadetblue'
colord = 'black'

#===============================================================================
# Plotting Options
#===============================================================================
nodes    = np.array([0   , 1   , 2   ])
vis_pred = np.array([0.13, 0.98, 0.98])
nme      = np.array([-0  , 1.59, 3.53])
sqrt_det = np.array([0.32, 4.17, 17.66])
four_det = np.array([0.15, 1.76, 3.73])
xlabels  = ['Self-Occluded', 'Normal', 'Ext Occluded']

# Setting the positions and width for the bars
pos = list(2*np.arange(nodes.shape[0]))
pos_bar1 = [p - 1.5 *width for p in pos]
pos_bar2 = [p - 0.5 *width for p in pos]
pos_bar3 = [p + 0.5 *width for p in pos]
pos_bar4 = [p + 1.5 *width for p in pos]
pos_xtick= [p for p in pos]

fig = plt.figure(figsize=(10, 6), dpi=100)
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(pos_bar1, vis_pred,  width=bar_width, color= color1, label=r'$\widehat{v}$')
ax1.set_ylabel(r'Mean $\widehat{v}$', color= colord)
ax1.set_xlabel(r'Type of landmark point')
ax1.set_ylim([0, 1.05])
for tl in ax1.get_yticklabels():
    tl.set_color(colord)
write_text_above_bar(bar1)


ax2 = ax1.twinx()
bar2 = ax2.bar(pos_bar2, nme     , width=bar_width, color=color4,  label=r'Error')
bar3 = ax2.bar(pos_bar3, four_det, width=bar_width, color=color2,  label=r'$|\Sigma|^{1/4}$')
bar4 = ax2.bar(pos_bar4, sqrt_det, width=bar_width, color=color3,  label=r'$|\Sigma|^{1/2}$')
ax2.set_ylabel(r'Mean Error, $|\Sigma|^{1/4}, |\Sigma|^{1/2}$', color= colord)
for tl in ax2.get_yticklabels():
    tl.set_color(colord)
ax2.set_ylim([0, np.ceil(np.max(sqrt_det))+1])
write_text_above_bar(bar2, hide_first= True)
write_text_above_bar(bar3, hide_first= False)
write_text_above_bar(bar4, hide_first= False)

ax2.set_xticklabels(xlabels) 
plt.xticks(pos_xtick)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg = ax1.legend(h1+h2, l1+l2, loc='center left', bbox_to_anchor=(-0.01,0.7),prop={'size': fs-2})

for lh in leg.get_lines(): 
    lh.set_alpha(0)

plt.grid()
plt.tight_layout()    
plt.savefig('images/aflw_ours_vis_unc_lmark_stats.png')
plt.show()
plt.close()
