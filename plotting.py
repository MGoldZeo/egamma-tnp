import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from hist import Hist
import hist.intervals
import hist.plot
import mplhep as hep

TABLEAU_COLORS = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

hep.style.use(hep.style.CMS)

def plot_1d(title,bkg_hists=None,bkg_label=None,sgl_hists=None,sgl_label=None,xtr_hists=None,xtr_label=None,\
            xlabel=None,xlim=None,ylabel="Events",signal_sf=1,logy=False,density=False,sort="yield",
            title_pos=None,year=None,lumi=None,com=13.6,legend_xoffset=0.02,flow="hint",figx=7,figy=4.7):
    '''
    Make a 1D histogram plot with stacked backgrounds and signals as lines.
    Shows plots as "CMS Preliminary", puts a legend in the top right, and by default
    calls the y axis "Events".

    Inputs:
        title: (str) The title of the plot.
        bkg_hists: (iterable) An iterable of single-axis histograms to be stacked
                    in the plot. May need to sum/integrate over all but one axis
                    to be iterable.
        bkg_label: (iterable) Iterable of strings of the same length as bkg_hists.
                    This goes in the legend. The first label goes with the first
                    histogram in bkg_hists, etc.
        sgl_hists: (iterable) Like bkg_hists, but will be overlayed as lines,
                    instead of stacked.
        sgl_label: (iterable) Like bkg_label, but with the same length as sgl_hists.
        xlabel: (str) The label for the x-axis. If not given, no label put on plot.
        xlim: (iterable of length 2) Zeroth argument is lower x-axis limit,
                    last argument is upper x-axis limit.
        signal_sf: (float/int) Scales the signal by this factor in plots and labels the key as so
        logy: (bool) If True, makes y-axis log-scaled
        density: (bool) If True, make all histograms plotted normed so their integrals
                    are equal to 1.
        sort: (str | None, optional) The sort kwarg to be passed to mplhep's histplot function. If no
                sorting is desired, pass None.
        title_pos: (float) If set, sets the height of the title in matplotlib coordinates (1.0 is top),
        year: (str | int) If set, labels the plot with a year
        lumi: (str | int) If set, labels the plot with a luminosity
        com: (float) The center-of-mass energy to label the plot with, in TeV. Defaults to 13.6
                    TeV.
        flow: (str) How to show under/overflow bins. Passed to mplhep.histplot
    '''

    fig,ax = plt.subplots(1,1,figsize=(figx,figy))

    ax = plot_1d_ax(ax,title,bkg_hists=bkg_hists,bkg_label=bkg_label,sgl_hists=sgl_hists,sgl_label=sgl_label,xtr_hists=xtr_hists,xtr_label=xtr_label,
                    xlabel=xlabel,xlim=xlim,ylabel=ylabel,signal_sf=signal_sf,logy=logy,density=density,sort=sort,
                    title_pos=title_pos,year=year,lumi=lumi,com=com,legend_xoffset=legend_xoffset,flow=flow)

    plt.show()

def plot_1d_ax(ax,title,bkg_hists=None,bkg_label=None,sgl_hists=None,sgl_label=None,xtr_hists=None,xtr_label=None,\
            xlabel=None,xlim=None,ylabel="Events",signal_sf=1,logy=False,density=False,sort="yield",
            title_pos=None,year=None,lumi=None,com=13.6,legend_xoffset=0.02,flow="hint"):
    '''
    Make a 1D histogram plot with stacked backgrounds and signals as lines.
    Shows plots as "CMS Preliminary", puts a legend in the top right, and by default
    calls the y axis "Events".

    Inputs:
        title: (str) The title of the plot.
        bkg_hists: (iterable) An iterable of single-axis histograms to be stacked
                    in the plot. May need to sum/integrate over all but one axis
                    to be iterable.
        bkg_label: (iterable) Iterable of strings of the same length as bkg_hists.
                    This goes in the legend. The first label goes with the first
                    histogram in bkg_hists, etc.
        sgl_hists: (iterable) Like bkg_hists, but will be overlayed as lines,
                    instead of stacked.
        sgl_label: (iterable) Like bkg_label, but with the same length as sgl_hists.
        xlabel: (str) The label for the x-axis. If not given, no label put on plot.
        xlim: (iterable of length 2) Zeroth argument is lower x-axis limit,
                    last argument is upper x-axis limit.
        ylabel: (str) The label for the y-axis. If not given, defaults to "Events".
        signal_sf: (float/int) Scales the signal by this factor in plots and labels the key as so
        logy: (bool) If True, makes y-axis log-scaled
        density: (bool) If True, make all histograms plotted normed so their integrals
                    are equal to 1.
        sort: (str | None, optional) The sort kwarg to be passed to mplhep's histplot function. If no
                sorting is desired, pass None.
        title_pos: (float) If set, sets the height of the title in matplotlib coordinates (1.0 is top),
        year: (str | int) If set, labels the plot with a year
        lumi: (str | int) If set, labels the plot with a luminosity
        com: (float) The center-of-mass energy to label the plot with, in TeV. Defaults to 13.6
                    TeV.
        flow: (str) How to show under/overflow bins. Passed to mplhep.histplot
    '''
    if bkg_hists:
        hep.histplot(bkg_hists,ax=ax,stack=True,histtype='fill',label=bkg_label,density=density,sort=sort,flow=flow)
    if xtr_hists:
        hep.histplot(xtr_hists,ax=ax,stack=True,histtype='fill',label=xtr_label,density=density,sort=sort,flow=flow)
    if sgl_hists:
        if signal_sf != 1:
            scaled_sgl_hists = [signal_sf*hist for hist in sgl_hists]
            scaled_sgl_label = [label+f' (x{signal_sf})' for label in sgl_label]
            hep.histplot(scaled_sgl_hists,ax=ax,label=scaled_sgl_label,density=density,sort=sort,flow=flow)
        else:
            hep.histplot(sgl_hists,ax=ax,label=sgl_label,density=density,sort=sort,flow=flow)

    #hep.cms.text("Preliminary",loc=1,fontsize=14)
    if lumi or year:
        hep.cms.label(lumi=lumi,year=year,loc=1,fontsize=14,lumi_format='{:.1f}',com=com)
    if (lumi or year) and not title_pos:
        ax.set_title(title,y=1.07,pad=2)
    else:
        ax.set_title(title,y=title_pos,pad=2)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0+legend_xoffset,0.999)).shadow=True
    ax.set_ylabel(ylabel,fontsize=10)
    ax.set_xlabel(xlabel,fontsize=10,labelpad=2)
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
    else:
        if ax.get_xlim()[0] < 0:
            ax.set_xlim(ax.get_xlim()[0]*1.2,ax.get_xlim()[1]*1.2)
        else:
            ax.set_xlim(0,ax.get_xlim()[1]*1.2)
    if logy:
        ax.semilogy()
    return ax

def plot_1d_tofile(title,outfile,bkg_hists=None,bkg_label=None,sgl_hists=None,sgl_label=None,\
            xlabel=None,xlim=None,ylabel="Events",signal_sf=1,logy=False,density=False,sort="yield",
            title_pos=None,year=None,lumi=None,com=13.6,legend_xoffset=0.02,flow="hint"):
    '''
    Make a 1D histogram plot with stacked backgrounds and signals as lines.
    Shows plots as "CMS Preliminary", puts a legend in the top right, and by default
    calls the y axis "Events".

    Inputs:
        title: (str) The title of the plot.
        outfile: (str) Where to save the plot, and what to call the file.
        bkg_hists: (iterable) An iterable of single-axis histograms to be stacked
                    in the plot. May need to sum/integrate over all but one axis
                    to be iterable.
        bkg_label: (iterable) Iterable of strings of the same length as bkg_hists.
                    This goes in the legend. The first label goes with the first
                    histogram in bkg_hists, etc.
        sgl_hists: (iterable) Like bkg_hists, but will be overlayed as lines,
                    instead of stacked.
        sgl_label: (iterable) Like bkg_label, but with the same length as sgl_hists.
        xlabel: (str) The label for the x-axis. If not given, no label put on plot.
        xlim: (iterable of length 2) Zeroth argument is lower x-axis limit,
                    last argument is upper x-axis limit.
        signal_sf: (float/int) Scales the signal by this factor in plots and labels the key as so
        logy: (bool) If True, makes y-axis log-scaled
        density: (bool) If True, make all histograms plotted normed so their integrals
                    are equal to 1.
        sort: (str | None, optional) The sort kwarg to be passed to mplhep's histplot function. If no
                sorting is desired, pass None.
        title_pos: (float) If set, sets the height of the title in matplotlib coordinates (1.0 is top),
        year: (str | int) If set, labels the plot with a year
        lumi: (str | int) If set, labels the plot with a luminosity
        com: (float) The center-of-mass energy to label the plot with, in TeV. Defaults to 13.6
                    TeV.
        flow: (str) How to show under/overflow bins. Passed to mplhep.histplot
    '''

    fig,ax = plt.subplots(1,1,figsize=(7,4.7))

    ax = plot_1d_ax(ax,title,bkg_hists=bkg_hists,bkg_label=bkg_label,sgl_hists=sgl_hists,sgl_label=sgl_label,
                    xlabel=xlabel,xlim=xlim,ylabel=ylabel,signal_sf=signal_sf,logy=logy,density=density,sort=sort,
                    title_pos=title_pos,year=year,lumi=lumi,com=com,legend_xoffset=legend_xoffset,flow=flow)
    
    plt.savefig(outfile)
    print('Saved output to {}'.format(outfile))

def plot_profile(title,prof_axis,hists,label=None,xlabel=None,xlim=None,ylabel="Events",
                 ylim=None,title_pos=None,year=None,lumi=None,com=13.6,legend_xoffset=0.02,flow="hint"):
    '''
    Plot 1D histograms made from profiling 2D histograms.
    Shows plots as "CMS Preliminary", puts a legend in the top right, and calls
    the y axis "Events".

    Inputs:
        title: (str) The title of the plot.
        prof_axis: (int) The index of the axis over which to profile (this axis is removed from the
                    plotted histogram).
        hists: (iterable) The 2D histograms to profile.
        label: (iterable) The labels for hists to go in the legend, in the same order as in hists.
        xlabel: (str) The label for the x-axis. If not given, no label put on plot.
        xlim: (iterable of length 2) Zeroth argument is lower x-axis limit,
                    last argument is upper x-axis limit.
        ylabel: (str) The label for the y-axis. If not given, no label put on plot.
        ylim: (iterable of length 2) Zeroth argument is lower y-axis limit,
                    last argument is upper y-axis limit.
        title_pos: (float) If set, sets the height of the title in matplotlib coordinates (1.0 is top),
        year: (str | int) If set, labels the plot with a year
        lumi: (str | int) If set, labels the plot with a luminosity
        com: (float) The center-of-mass energy to label the plot with, in TeV. Defaults to 13.6
                    TeV.
        flow: (str) How to show under/overflow bins. Passed to mplhep.histplot
    '''
    fig,ax = plt.subplots(1,1,figsize=(7,4.7))

    hep.histplot([h.profile(prof_axis) for h in hists],
                 yerr=[np.sqrt(h.profile(prof_axis).view().variance) for h in hists],
                 ax=ax,
                 label=label,
                 flow=flow
                )

    if lumi or year:
        hep.cms.label(lumi=lumi,year=year,loc=1,fontsize=14,lumi_format='{:.1f}',com=com)
    if (lumi or year) and not title_pos:
        ax.set_title(title,y=1.07,pad=2)
    else:
        ax.set_title(title,y=title_pos,pad=2)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0+legend_xoffset,0.999)).shadow=True
    ax.set_ylabel(ylabel,fontsize=10)
    ax.set_xlabel(xlabel,fontsize=10,labelpad=2)
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
    else:
        if ax.get_xlim()[0] < 0:
            ax.set_xlim(ax.get_xlim()[0]*1.2,ax.get_xlim()[1]*1.2)
        else:
            ax.set_xlim(0,ax.get_xlim()[1]*1.2)
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        if ax.get_ylim()[0] < 0:
            ax.set_ylim(ax.get_ylim()[0]*1.2,ax.get_ylim()[1]*1.2)
        else:
            ax.set_ylim(0,ax.get_ylim()[1]*1.2)
    
    plt.show()

def plot_2d(ax,hist2d,title=None,xlabel=None,xlim=None,logx=False,ylabel=None,ylim=None,\
            logy=False,xbar_low=None,xbar_high=None,ybar_low=None,ybar_high=None,title_pos=None,\
            year=None,lumi=None,com=13.6,norm=None,flow="hint"):
    '''
    Plot a 2D histogram with various customizations, including writing "CMS Preliminary", writing
    the COM energy, potentially the luminosity, year, and title.

    Inputs:
        title: (str) The title of the plot.
        bkg_hists: (iterable) An iterable of single-axis histograms to be stacked
                    in the plot. May need to sum/integrate over all but one axis
                    to be iterable.
        bkg_label: (iterable) Iterable of strings of the same length as bkg_hists.
                    This goes in the legend. The first label goes with the first
                    histogram in bkg_hists, etc.
        sgl_hists: (iterable) Like bkg_hists, but will be overlayed as lines,
                    instead of stacked.
        sgl_label: (iterable) Like bkg_label, but with the same length as sgl_hists.
        xlabel: (str) The label for the x-axis. If not given, no label put on plot.
        xlim: (iterable of length 2) Zeroth argument is lower x-axis limit,
                    last argument is upper x-axis limit.
        signal_sf: (float/int) Scales the signal by this factor in plots and labels the key as so
        logy: (bool) If True, makes y-axis log-scaled
        xbar_low: (float) Draw a vertical line at this x-value
        xbar_high: (float) Draw a vertical line at this x-value
        ybar_low: (float) Draw a horizontal line at this y-value
        ybar_high: (float) Draw a horizontal line at this y-value
        density: (bool) If True, make all histograms plotted normed so their integrals
                    are equal to 1.
        title_pos: (float) If set, sets the height of the title in matplotlib coordinates (1.0 is top),
        year: (str | int) If set, labels the plot with a year
        lumi: (str | int) If set, labels the plot with a luminosity
        com: (float) The center of mass energy, in GeV
        norm: (str, optional) If given, sets the normalization of the z-axis (color) eg: log
        flow: (bool) If True, show underflow and overflow bins
    '''
    hep.hist2dplot(hist2d,ax=ax,norm=norm,flow=flow)
    if lumi or year:
        hep.cms.label(lumi=lumi,year=year,loc=1,fontsize=14,lumi_format='{:.1f}',com=com)
    if (lumi or year) and not title_pos:
        ax.set_title(title,y=1.07,pad=2)
    else:
        ax.set_title(title,y=title_pos,pad=2)
    ax.set_ylabel(ylabel,fontsize=10)
    ax.set_xlabel(xlabel,fontsize=10,labelpad=2)
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
    if logx:
        ax.semilogx()
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    if logy:
        ax.semilogy()
    if xbar_low:
        ax.plot((xbar_low,xbar_low),(ax.get_ylim()[0],ax.get_ylim()[1]),color='red')
    if xbar_high:
        ax.plot((xbar_high,xbar_high),(ax.get_ylim()[0],ax.get_ylim()[1]),color='red')
    if ybar_low:
        ax.plot((ax.get_xlim()[0],ax.get_xlim()[1]),(ybar_low,ybar_low),color='red')
    if ybar_high:
        ax.plot((ax.get_xlim()[0],ax.get_xlim()[1]),(ybar_high,ybar_high),color='red')
    return ax

def plot_wRatio(hMCs,hData,MC_labels,title,sgl_hists=None,sgl_label=None,signal_sf=1,lumi = 100.0,year = 2023,
                com = 13.6,title_pos = 1.07,legend_xoffset = 0.02,xlabel = None,xlim = None,logy = False):
    '''
    Docstring here. Note that hMCs have to use hist.storage.Weight() as storage, otherwise
    we cannot add them together and get variances.
    '''
    #If more than 6 things plotted, use 10-color palette
    if len(hMCs) > 6:
        colors = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
        hep.styles.CMS["axes.prop_cycle"] = cycler("color", colors)
        hep.style.use(hep.style.CMS)
    
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.07)
    
    hep.histplot(hMCs,ax=ax,stack=True,histtype='fill',label=MC_labels,sort="yield")
    if sgl_hists:
        if signal_sf != 1:
            scaled_sgl_hists = [signal_sf*hist for hist in sgl_hists]
            scaled_sgl_label = [label+f' (x{signal_sf})' for label in sgl_label]
            hep.histplot(scaled_sgl_hists,ax=ax,label=scaled_sgl_label,sort="yield")
        else:
            hep.histplot(sgl_hists,ax=ax,label=sgl_label,sort="yield")
    ax.set_xlabel(None)
    
    mc_sum = sum(hMCs)
    
    mcStatUp = np.append(mc_sum.values() + np.sqrt(mc_sum.variances()),[0])
    mcStatDo = np.append(mc_sum.values() - np.sqrt(mc_sum.variances()),[0])
    
    uncertainty_band = ax.fill_between(
        hData.axes[0].edges,
        mcStatUp,
        mcStatDo,
        step='post',
        hatch='///',
        facecolor='none',
        edgecolor='gray',
        linewidth=0,
    )
    
    ax.errorbar(x=hData.axes[0].centers,
                y=hData.values(),
                yerr=np.sqrt(hData.values()),
                color='black',
                marker='.',
                markersize=10,
                linewidth=0,
                elinewidth=1,
                label="Data",
    )
    
    #Ratio plot
    ratio_mcStatUp = np.append(1 + np.sqrt(mc_sum.variances())/mc_sum.values(),[0])
    ratio_mcStatDo = np.append(1 - np.sqrt(mc_sum.variances())/mc_sum.values(),[0])
    
        
    ratio_uncertainty_band = rax.fill_between(
        hData.axes[0].edges,
        ratio_mcStatUp,
        ratio_mcStatDo,
        step='post',
        color='lightgray',
    )
    
    hist_1_values, hist_2_values = hData.values(), mc_sum.values()
    
    ratios = hist_1_values / hist_2_values
    ratio_uncert = hist.intervals.ratio_uncertainty(
        num=hist_1_values,
        denom=hist_2_values,
        uncertainty_type="poisson",
        
    )
    # ratio: plot the ratios using Matplotlib errorbar or bar
    hist.plot.plot_ratio_array(
        hData, ratios, ratio_uncert, ax=rax, uncert_draw_type='line',
    );
    #hData is just used for its bins in the above line
    
    if (not lumi is None) or (not year is None):
        hep.cms.label(ax=ax,lumi=lumi,year=year,loc=1,fontsize=12,lumi_format='{:.1f}',com=com)
    if ((not lumi is None) or (not year is None)) and (title_pos is None):
        ax.set_title(title,y=1.07,pad=2,fontsize=14)
    else:
        ax.set_title(title,y=title_pos,pad=2,fontsize=14)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0+legend_xoffset,0.999)).shadow=True
    ax.set_ylabel('Events',fontsize=10)
    if not xlabel is None:
        rax.set_xlabel(xlabel,fontsize=10,labelpad=2)
    if not xlim is None:
        ax.set_xlim(xlim[0],xlim[1])
    else:
        ax.set_xlim(0,ax.get_xlim()[1]*1.2)
    if logy:
        ax.semilogy()
    
    plt.show()

    hep.styles.CMS["axes.prop_cycle"] = cycler("color", hep.style.cms.cmap_petroff)
    hep.style.use(hep.style.CMS)