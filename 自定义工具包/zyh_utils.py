import pandas as pd

def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(consdataed_layout=True, figsize=(20, 8), dpi=100)
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot=ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax=ax3)

def res_plot(y_true, y_pred, title=None, xlabel=None, 
             new_x=None, ylabel='Residual', way='stem', markersize=10, x_rotation=0):
    import matplotlib.pyplot as plt

    ## 画残差图
    res = y_true - y_pred
    plt.figure(figsize=(20, 8), dpi=100)
    if way=='scatter':
        plt.scatter(range(len(res)), res, linewidths=markersize)
    else:
        pre_markersize = plt.rcParams['lines.markersize']
        plt.rcParams['lines.markersize'] = markersize
        plt.stem(res)
        plt.rcParams['lines.markersize'] = pre_markersize

    if new_x==None:
        new_x = range(len(res))
    plt.xticks(range(len(res)), new_x, rotation=x_rotation)
    plt.title(title, fontsize=30)
    plt.tick_params(labelsize=15)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)

    return res

def corr_plot(data, mask=True, method_id=0):
    # {‘pearson’, ‘spearman’, ‘kendall’}
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    method_list = ['pearson', 'spearman', 'kendall']
    method = method_list[method_id]
    if mask==True:
        mask = np.zeros_like(data.corr(), dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = np.ones_like(data.corr(), dtype=np.bool)

    plt.figure(figsize=(20, 8), dpi=100)
    sns.heatmap(data.corr(method=method), 
            cmap=sns.diverging_palette(20, 220, n=200), 
            mask = mask, 
            annot=True, 
            center = 0, 
           )
    ## Give title. 
    plt.title("Heatmap of all the Features", fontsize = 30)
