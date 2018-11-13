# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# set the path of where the input files are
mywd = "C:/Users/Nick Morris/Downloads/ABP/Budget-Uncertainty-Data/All-Data"

# machine learning with h2o

# ---- modeling options ----
# ?h2o.deepfeatures
# ?h2o.deeplearning
# ?h2o.xgboost
# ?h2o.randomForest
# ?h2o.glm
# ?h2o.stackedEnsemble
# ?h2o.transform
# ?h2o.word2vec
# ?h2o.exportHDFS

# ---- prediction results ----
# ?h2o.make_metrics
# ?h2o.error
# ?h2o.coef
# ?h2o.coef_norm
# ?h2o.predict_leaf_node_assignment

# ---- useful plots ----
# ?h2o.partialPlot
# ?h2o.varimp_plot


# -----------------------------------------------------------------------------------
# ---- Packages ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# data handling
require(data.table)
require(tm)
require(gtools)
require(stringdist)
require(zoo)
require(missRanger)
require(TTR)

# plotting
require(ggplot2)
require(gridExtra)
require(GGally)
require(scales)
require(scatterplot3d)
require(gridGraphics)
require(corrplot)
require(VIM)

# modeling
require(fpc)
require(caret)
require(ranger)
require(cluster)
require(car)
require(nortest)
require(neuralnet)
require(h2o)

# parallel computing
require(foreach)
require(parallel)
require(doSNOW)

}

# -----------------------------------------------------------------------------------
# ---- Functions --------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# these are functions i like to use

# ---- prints the data types of each column in a data frame -------------------------

types = function(dat)
{
  # make dat into a data.frame
  dat = data.frame(dat)
  
  # get the column names
  column = colnames(dat)
  
  # get the class of the columns
  data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
  
  # compute the number of levels for each column
  levels = sapply(1:ncol(dat), function(i) ifelse(data.type[i] == "factor", length(levels(droplevels(dat[,i]))), 0))
  
  return(data.frame(column, data.type, levels))
}

# ---- converts all columns to a character data type --------------------------------

tochar = function(dat)
{
  # make dat into a data.frame
  dat = data.frame(dat)
  
  # get the column names
  column = colnames(dat)
  
  # get the values in the columns and convert them to character data types
  values = lapply(1:ncol(dat), function(i) as.character(dat[,i]))
  
  # combine the values back into a data.frame
  dat = data.frame(do.call("cbind", values), stringsAsFactors = FALSE)
  
  # give dat its column names
  colnames(dat) = column
  
  return(dat)
}

# ---- a qualitative color scheme ---------------------------------------------------

mycolors = function(n)
{
  require(grDevices)
  return(colorRampPalette(c("#e41a1c", "#0099ff", "#4daf4a", "#984ea3", "#ff7f00", "#ff96ca", "#a65628"))(n))
}

# ---- emulates the default ggplot2 color scheme ------------------------------------

ggcolor = function(n, alpha = 1)
{
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

# ---- plots various cluster plots --------------------------------------------------

plot.clusters = function(dat, cluster.column.name = NULL, distance.matrix = NULL, DC.title = "Discriminant Coordinate Cluster Plot", pairs.title = "Cluster Pairs Plot", silhouette.title = "Silhouette Width", font.size = 20, pairs.plot.font.size = 12, rotate = 0, cor.size = 3)
{
  # load packages we need
  require(data.table)
  require(ggplot2)
  require(GGally)
  require(scatterplot3d)
  require(gridGraphics)
  require(grid)
  require(scales)
  require(cluster)
  require(fpc)
  
  # this function emulates the default ggplot2 color scheme
  ggcolor = function(n)
  {
    hues = seq(15, 375, length = n + 1)
    hcl(h = hues, l = 65, c = 100)[1:n]
  }
  
  # error check
  if(is.null(cluster.column.name))
  {
    print("you must specify a value for the parameter: cluster.column.name")
    
  } else
  {
    # ---- computing discriminant coordiantes -------------------------------
    
    # make dat into a data table
    dat = data.table(dat)
    
    # extract the cluster column from dat
    clusters = as.numeric(unname(unlist(dat[, cluster.column.name, with = FALSE])))
    
    # remove the cluster column from dat
    dat = dat[, !cluster.column.name, with = FALSE]
    
    # compute the number of columns in dat
    numcol = ncol(dat)
    
    # compute the discriminant coordinates, and extract the first 3
    dat.dc = data.table(discrproj(x = dat, 
                                  clvecd = clusters,
                                  method = "dc")$proj[,1:min(c(3, numcol))])
    
    # rename the columns appropriately
    setnames(dat.dc, paste0("DC", 1:min(c(3, numcol))))
    
    # give dat.dc the cluster column, and make it into a factor for plotting purposes
    dat.dc[, Cluster := factor(clusters, levels = sort(unique(clusters)))]
    
    # create an output list to store results
    output = list()
    
    # ---- plotting 2D discriminant coordiantes -----------------------------
    
    if(numcol >= 2)
    {
      # create a 2D cluster plot across the first 2 discriminant coordinates
      plot.2D = ggplot(dat.dc, aes(x = DC1, y = DC2, fill = Cluster)) +
        stat_density_2d(geom = "polygon", color = NA, alpha = 1/3) +
        theme_bw(font.size) +
        ggtitle(DC.title) +
        theme(legend.position = "top", 
              legend.key.size = unit(.25, "in"), 
              plot.title = element_text(hjust = 0.5),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank()) +
        guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1))
      
      # store this plot in output
      output$plot.2D = plot.2D
    }
    
    # ---- plotting 3D discriminant coordiantes -----------------------------
    
    if(numcol >= 3)
    {
      # create a table indicating which cluster should get which ggplot color
      color.dat.dc = data.table(Cluster = levels(dat.dc$Cluster),
                                Color = ggcolor(length(levels(dat.dc$Cluster))))
      
      # set Cluster as the key column in color.dat.dc and dat.dc
      setkey(dat.dc, Cluster)
      setkey(color.dat.dc, Cluster)
      
      # join color.dat.dc onto dat.dc
      dat.dc = color.dat.dc[dat.dc]
      
      # convert Cluster back into a factor data type
      dat.dc[, Cluster := factor(Cluster, levels = sort(unique(Cluster)))]
      
      # here is my default font size for base R plotting
      font.default = 20
      
      # compute the desired adjustment according to the specified value of font.size
      font.adjust = font.size / font.default
      
      # adjust the font of the title, axis titles, axis labels, and legend
      font.title = 2 * font.adjust
      font.axis.title = 1.125 * font.adjust
      font.axis.label = 1.125 * font.adjust
      font.legend = 1.75 * font.adjust
      
      # here are my 4 default angles for viewing a 3D scatterplot
      angles = c(45, 135, 225, 315)
      
      # apply the specified rotation
      angles = angles + rotate
      
      # set up 4 plot ID numbers so each plot angle has a position in the plot window
      plot.id = 2:5
      
      # set up a legend ID number so the legend has a postion across the top of the plot window
      legend.id = c(1, 1)
      
      # set up a matrix that defines the plot layout
      plot.layout = matrix(c(legend.id, plot.id), nrow = 3, ncol = 2, byrow = TRUE)
      
      # create a new plot window
      windows()
      plot.new()
      
      # define plot margins
      par(mar = c(0, 0, 3.5, 0))
      
      # apply the layout to the plot window
      layout(mat = plot.layout, heights = c(1, 1.5, 1.5))
      
      # produce a dummy plot as a place holder for the legend and title
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      
      # produce the title
      title(main = DC.title, cex.main = font.title)
      
      # produce the legend
      legend("top", inset = 0, bty = "n", cex = font.legend, horiz = TRUE,
             title = "Clusters", legend = levels(dat.dc$Cluster), fill = ggcolor(length(levels(dat.dc$Cluster))))
      
      # create 3D cluster plots across the first 3 discriminant coordinates
      for(angle in angles)
      {
        # build the scatterplot
        scatterplot3d(x = dat.dc$DC1, y = dat.dc$DC2, z = dat.dc$DC3, color = alpha(dat.dc$Color, 1/3), 
                      xlab = "DC1", ylab = "DC2", zlab = "DC3", mar = c(3, 3, 0, 3), cex = 1.5,
                      pch = 16, cex.lab = font.axis.title, cex.axis = font.axis.label, angle = angle)
      }
      
      # save this plot as a grid object
      grid.echo()
      plot.3D = grid.grab()
      
      # add this plot to our output list
      output$plot.3D = plot.3D
      
      # close the graphics window
      graphics.off()
    }
    
    # ---- plotting clusters across variable pairs ---------------------------
    
    # give dat the cluster column, and make it into a factor for plotting purposes
    dat[, Cluster := factor(clusters, levels = sort(unique(clusters)))]
    
    # plot the clusters across all variable pairs
    plot.pairs = ggpairs(dat,
                         mapping = aes(color = Cluster, fill = Cluster),
                         columns = which(names(dat) != "Cluster"),
                         lower = list(continuous = wrap(ggally_points, size = 1.5, alpha = 1/3)), 
                         upper = list(continuous = wrap(ggally_cor, size = cor.size)),
                         diag = list(continuous = wrap(ggally_densityDiag, alpha = 1/3)),
                         title = pairs.title,
                         legend = grab_legend(plot.2D + theme_classic(base_size = pairs.plot.font.size) + theme(legend.position = "top"))) + 
      theme_classic(base_size = pairs.plot.font.size) +
      theme(legend.position = "top", plot.title = element_text(hjust = 0.5))
    
    # remove the cluster column from dat
    dat[, Cluster := NULL]
    
    # add this plot to our output list
    output$plot.pairs = plot.pairs
    
    # ---- plotting silhouette widths ----------------------------------------
    
    # if the user gave a distance matrix then lets compute silhouette widths
    if(!is.null(distance.matrix))
    {
      # compute the silhouette widths
      dat.sil = silhouette(x = clusters,
                           dist = distance.matrix)
      
      # compute the summary of dat.sil
      dat.sil.sum = summary(dat.sil)
      
      # extract the avg widths from dat.sil.sum
      dat.sil.avg = data.table(Cluster = as.numeric(names(dat.sil.sum$clus.avg.widths)),
                               Average_Width = round(unname(dat.sil.sum$clus.avg.widths), 2))
      
      # order dat.sil.avg by Cluster
      dat.sil.avg = dat.sil.avg[order(Cluster)]
      
      # extract the cluster sizes from dat.sil.sum
      dat.sil.size = data.table(Cluster = as.numeric(names(dat.sil.sum$clus.sizes)),
                                Size = as.numeric(unname(dat.sil.sum$clus.sizes)))
      
      # order dat.sil.size by Cluster
      dat.sil.size = dat.sil.size[order(Cluster)]
      
      # combine dat.sil.avg and dat.sil.size into a table that will go on our plot
      dat.sil.tab = cbind(dat.sil.avg, dat.sil.size[,!"Cluster"])
      
      # convert dat.sil into a data table
      dat.sil = data.table(dat.sil[1:nrow(dat.sil), 1:ncol(dat.sil)])
      
      # sort dat.sil by cluster and sil_width
      dat.sil = dat.sil[order(cluster, -sil_width)]
      
      # give dat.sil an ID column for plotting purposes
      dat.sil[, ID := 1:nrow(dat.sil)]
      
      # convert cluster to a factor for plotting purposes
      dat.sil[, cluster := factor(cluster, levels = sort(unique(cluster)))]
      
      # aggregate sil_width by cluster in dat.sil to determine where to place dat.sil.tab in the plot 
      dat.agg = dat.sil[, .(sil_width.min = min(sil_width),
                            sil_width.max = max(sil_width)),
                        by = cluster]
      
      # build the four corners of the dat.sil.tab to place it in the plot
      # find the cluster with the smallest peak and set the peak's sil_width as the ymin
      ymin = as.numeric(min(dat.agg$sil_width.max))
      
      # find the sil_width of the max peak and set it as the ymax
      ymax = as.numeric(max(dat.sil$sil_width))
      
      # extract the cluster with the smallest peak from dat.agg
      small.peak = dat.agg[which.min(sil_width.max), cluster]
      
      # find the first ID number in dat.sil for the cluster with the smallest peak, and set that as xmin
      xmin = min(dat.sil[cluster == small.peak, ID])
      
      # find the last ID number in dat.sil for the cluster with the smallest peak, and set that as xmax
      xmax = max(dat.sil[cluster == small.peak, ID])
      
      # plot the silhouette width and add the dat.sil.tab to it
      plot.sil.width = ggplot(dat.sil, aes(x = ID, y = sil_width, fill = cluster, color = cluster)) + 
        geom_bar(stat = "identity", position = "dodge") +
        # annotation_custom(tableGrob(as.matrix(dat.sil.tab), rows = NULL, 
        #                             theme = ttheme_default(base_size = font.size,
        #                                                    colhead = list(fg_params = list(col = "black"), bg_params = list(fill = "lightgray", col = "black")),
        #                                                    core = list(fg_params = list(hjust = 0.5), bg_params = list(fill = c("white"), col = "black")))),
        #                   xmin = xmin, 
        #                   xmax = xmax, 
        #                   ymin = ymin, 
        #                   ymax = ymax) +
        ggtitle(silhouette.title) +
        labs(x = "Observation", y = "Silhouette Width", fill = "Cluster", color = "Cluster") +
        theme_bw(base_size = font.size) +
        theme(legend.position = "top", plot.title = element_text(hjust = 0.5)) + 
        guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1),
               color = guide_legend(nrow = 1))
      
      # add this plot to our output list
      output$plot.sil.width = plot.sil.width
    }
    
    # add the DC data to the output list
    output$dat.dc = dat.dc
    
    # helpful message
    print("use grid.draw() to see the 3D cluster plot, for example: grid.draw(my.plot.clusters$plot.3D)")
    
    return(output)
  }
}

# ---- prints out a dat file object in ampl syntax ----------------------------------

ampl = function(dat, object = "param", name = "c")
{
  # converts all columns to a character data type 
  tochar = function(dat)
  {
    # make dat into a data.frame
    dat = data.frame(dat)
    
    # get the column names
    column = colnames(dat)
    
    # get the values in the columns and convert them to character data types
    values = lapply(1:ncol(dat), function(i) as.character(dat[,i]))
    
    # combine the values back into a data.frame
    dat = data.frame(do.call("cbind", values), stringsAsFactors = FALSE)
    
    # give dat its column names
    colnames(dat) = column
    
    return(dat)
  }
  
  # make sure the data is a data frame object
  dat = tochar(dat)
  
  # every parameter/set object in an ampl dat file must end with a semicolon
  # so set up 1 semicolon to give to dat
  semicolon = c(";", rep(" ", ncol(dat) - 1))
  
  # add this semicolon as the last row of the data frame
  result = data.frame(rbind(dat, semicolon))
  
  # every parameter/set object in an ample dat file must begin with the name of the object and what it equals
  # for example: param c := 
  # so set up a header to give to dat
  header = c(paste(object, name, ":="), rep(" ", ncol(dat) - 1))
  
  # update the column names of dat to be the header we created
  colnames(result) = header
  
  # print out the result without any row names
  # print out the result left adjusted
  # print(result, right = FALSE, row.names = FALSE)
  
  return(result)	
}

# ---- compares the quantiles of emprical data against the quantiles of any statistical distribution 

ggqq = function(x, distribution = "norm", ..., conf = 0.95, probs = c(0.25, 0.75), alpha = 0.33, basefont = 20, main = "", xlab = "\nTheoretical Quantiles", ylab = "Empirical Quantiles\n")
{
  require(ggplot2)
  
  # compute the sample quantiles and theoretical quantiles
  q.function = eval(parse(text = paste0("q", distribution)))
  d.function = eval(parse(text = paste0("d", distribution)))
  x = na.omit(x)
  ord = order(x)
  n = length(x)
  P = ppoints(length(x))
  df = data.frame(ord.x = x[ord], z = q.function(P, ...))
  
  # compute the quantile line
  Q.x = quantile(df$ord.x, c(probs[1], probs[2]))
  Q.z = q.function(c(probs[1], probs[2]), ...)
  b = diff(Q.x) / diff(Q.z)
  coef = c(Q.x[1] - (b * Q.z[1]), b)
  
  # compute the confidence interval band
  zz = qnorm(1 - (1 - conf) / 2)
  SE = (coef[2] / d.function(df$z, ...)) * sqrt(P * (1 - P) / n)
  fit.value = coef[1] + (coef[2] * df$z)
  df$upper = fit.value + (zz * SE)
  df$lower = fit.value - (zz * SE)
  
  # plot the qqplot
  p = ggplot(df, aes(x = z, y = ord.x)) + 
    geom_point(color = "blue", alpha = alpha) +
    geom_abline(intercept = coef[1], slope = coef[2], size = 1) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1) +
    coord_cartesian(ylim = c(min(df$ord.x), max(df$ord.x))) + 
    labs(x = xlab, y = ylab) +
    theme_bw(base_size = basefont) +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  # conditional additions
  if(main != "")(p = p + ggtitle(main))
  
  return(p)
}

# ---- plots 6 residual plots -------------------------------------------------------

residplots = function(actual, fit, binwidth = NULL, from = NULL, to = NULL, by = NULL, histlabel.y = -10, n = NULL, basefont = 20)
{
  require(ggplot2)
  
  residual = actual - fit 
  DF = data.frame("actual" = actual, "fit" = fit, "residual" = residual)
  
  rvfPlot = ggplot(DF, aes(x = fit, y = residual)) + 
    geom_point(na.rm = TRUE) +
    stat_smooth(method = "loess", se = FALSE, na.rm = TRUE, color = "blue") +
    geom_hline(yintercept = 0, col = "red", linetype = "dashed") +
    xlab("Fitted values") +
    ylab("Residuals") +
    ggtitle("Residual vs Fitted Plot") + 
    theme_bw(base_size = basefont) +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  ggqq = function(x, distribution = "norm", ..., conf = 0.95, probs = c(0.25, 0.75), note = TRUE, alpha = 0.33, main = "", xlab = "\nTheoretical Quantiles", ylab = "Empirical Quantiles\n")
  {
    # compute the sample quantiles and theoretical quantiles
    q.function = eval(parse(text = paste0("q", distribution)))
    d.function = eval(parse(text = paste0("d", distribution)))
    x = na.omit(x)
    ord = order(x)
    n = length(x)
    P = ppoints(length(x))
    DF = data.frame(ord.x = x[ord], z = q.function(P, ...))
    
    # compute the quantile line
    Q.x = quantile(DF$ord.x, c(probs[1], probs[2]))
    Q.z = q.function(c(probs[1], probs[2]), ...)
    b = diff(Q.x) / diff(Q.z)
    coef = c(Q.x[1] - (b * Q.z[1]), b)
    
    # compute the confidence interval band
    zz = qnorm(1 - (1 - conf) / 2)
    SE = (coef[2] / d.function(DF$z, ...)) * sqrt(P * (1 - P) / n)
    fit.value = coef[1] + (coef[2] * DF$z)
    DF$upper = fit.value + (zz * SE)
    DF$lower = fit.value - (zz * SE)
    
    # plot the qqplot
    p = ggplot(DF, aes(x = z, y = ord.x)) + 
      geom_point(color = "black", alpha = alpha) +
      geom_abline(intercept = coef[1], slope = coef[2], size = 1, color = "blue") +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15) +
      coord_cartesian(ylim = c(min(DF$ord.x), max(DF$ord.x))) + 
      labs(x = xlab, y = ylab) +
      theme_bw(base_size = basefont) +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank())
    
    # conditional additions
    if(main != "")(p = p + ggtitle(main))
    
    return(p)
  }
  
  qqPlot = ggqq(residual, 
                alpha = 1,				  
                main = "Normal Q-Q Plot", 
                xlab = "Theoretical Quantiles", 
                ylab = "Residuals")
  
  rvtPlot = ggplot(data.frame("x" = 1:length(DF$residual), "y" = DF$residual), aes(x = x, y = y)) + 
    geom_line(na.rm = TRUE) +
    geom_hline(yintercept = 0, col = "red", linetype = "dashed") +
    xlab("Obs. Number") +
    ylab("Residuals") +
    ggtitle("Residual Time Series") + 
    theme_bw(base_size = basefont) +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  variogramDF = function(x)
  {
    n = length(x) - 2
    
    num = sapply(1:n, function(k)
      sapply(1:(length(x) - k), function(i)
        x[i + k] - x[i]))
    
    num = sapply(1:length(num), function(j)
      var(num[[j]]))
    
    den = var(sapply(1:(length(x) - 1), function(i)
      x[i + 1] - x[i]))
    
    val = num / den
    
    DF = data.frame("Lag" = 1:n, "Variogram" = val)
    
    return(DF)
  }
  
  DFv = variogramDF(x = DF$residual)
  
  varioPlot = ggplot(DFv, aes(x = Lag, y = Variogram)) + 
    geom_point() +
    geom_line(color = "blue") +
    xlab("Lag") +
    ylab("Variogram") +
    ggtitle("Variogram of Residuals") + 
    theme_bw(base_size = basefont) +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  test = t.test(DF$residual)
  
  CI = data.frame("x" = test$estimate, 
                  "LCB" = test$conf.int[1], 
                  "UCB" = test$conf.int[2], 
                  row.names = 1)
  
  histPlot = ggplot(DF, aes(x = residual)) +
    geom_histogram(color = "white", fill = "black", binwidth = binwidth) +
    geom_segment(data = CI, aes(x = LCB, xend = LCB, y = 0, yend = Inf), color = "blue") +
    geom_segment(data = CI, aes(x = UCB, xend = UCB, y = 0, yend = Inf), color = "blue") +
    annotate("text", x = CI$x, y = histlabel.y, 
             label = "T-Test C.I.", size = 5, 
             color = "blue", fontface = 2) + 
    ggtitle("Residual Histogram") +
    labs(x = "Residuals", y = "Frequency") +
    theme_bw(base_size = basefont) +
    theme(legend.key.size = unit(.25, "in"),
          legend.position = "bottom",
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  if(class(from) != "NULL" & class(to) != "NULL" & class(by) != "NULL") (histPlot = histPlot + scale_x_continuous(breaks = seq(from = from, to = to, by = by)))
  
  ggacf = function(x, n = NULL, conf.level = 0.95, main = "ACF Plot", xlab = "Lag", ylab = "Autocorrelation", basefont = 20) 
  {
    if(class(n) == "NULL")
    {
      n = length(x) - 2
    }
    
    ciline = qnorm((1 - conf.level) / 2) / sqrt(length(x))
    bacf = acf(x, lag.max = n, plot = FALSE)
    bacfdf = with(bacf, data.frame(lag, acf))
    bacfdf = bacfdf[-1,]
    
    p = ggplot(bacfdf, aes(x = lag, y = acf)) + 
      geom_bar(stat = "identity", position = "dodge", fill = "black") +
      geom_hline(yintercept = -ciline, color = "blue", size = 1) +
      geom_hline(yintercept = ciline, color = "blue", size = 1) +
      geom_hline(yintercept = 0, color = "red", size = 1) +
      labs(x = xlab, y = ylab) +
      ggtitle(main) +
      theme_bw(base_size = basefont) +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank())
    
    return(p)
  }
  
  acfPlot = ggacf(x = DF$residual, main = "ACF Plot of Residuals", basefont = basefont, n = n)
  
  return(list("rvfPlot" = rvfPlot, 
              "qqPlot" = qqPlot, 
              "rvtPlot" = rvtPlot, 
              "varioPlot" = varioPlot, 
              "histPlot" = histPlot, 
              "acfPlot" = acfPlot))
}

}

# -----------------------------------------------------------------------------------
# ---- Prepare the Data -------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# set the work directory
setwd(mywd)

# do we need to join the data together or has it already been done?
join.data = FALSE

if(join.data)
{
  # ---- import all of the data ----
  bud.imm = data.table(read.csv("budget-for-immunization.csv", stringsAsFactors = FALSE))
  bud.sup = data.table(read.csv("budget-for-supplies.csv", stringsAsFactors = FALSE))
  imm.exp.all = data.table(tochar(read.csv("immunization-expenditure-all.csv", stringsAsFactors = FALSE)))
  imm.exp.gov = data.table(tochar(read.csv("immunization-expenditure-gov.csv", stringsAsFactors = FALSE)))
  vac.exp.all = data.table(tochar(read.csv("vaccine-expenditure-all.csv", stringsAsFactors = FALSE)))
  vac.exp.gov = data.table(tochar(read.csv("vaccine-expenditure-gov.csv", stringsAsFactors = FALSE)))
  gci = data.table(read.csv("GCI.csv", stringsAsFactors = FALSE))
  hnp = data.table(read.csv("HNP-Main.csv", stringsAsFactors = FALSE))
  imf = data.table(tochar(read.csv("IMF.csv", stringsAsFactors = FALSE)))
  edu = data.table(tochar(read.csv("Education.csv", stringsAsFactors = FALSE)))
  gdp = data.table(tochar(read.csv("GDP.csv", stringsAsFactors = FALSE)))
  gni = data.table(tochar(read.csv("GNI.csv", stringsAsFactors = FALSE)))
  inflat = data.table(tochar(read.csv("Inflation.csv", stringsAsFactors = FALSE)))
  unempl = data.table(tochar(read.csv("Unemployment.csv", stringsAsFactors = FALSE)))
  gdppc = data.table(tochar(read.csv("GDPpc.csv", stringsAsFactors = FALSE)))
  gnipc = data.table(tochar(read.csv("GNIpc.csv", stringsAsFactors = FALSE)))
  gini = data.table(read.csv("GINI.csv", stringsAsFactors = FALSE))
  wdi = data.table(read.csv("WDI.csv", stringsAsFactors = FALSE))
  int = data.table(read.csv("Interest Rate.csv", stringsAsFactors = FALSE))
  hdi = data.table(read.csv("HDI.csv", stringsAsFactors = FALSE))
  abp = data.table(read.csv("ABP-countries.csv", stringsAsFactors = FALSE))
  
  # lets extract some indicators of interest from various tables
  prim = data.table(edu[Indicator == "School enrollment, primary (% net)"])
  sec = data.table(edu[Indicator == "School enrollment, secondary (% net)"])
  life = data.table(wdi[Indicator == "Life expectancy at birth, total (years)"])
  water = data.table(wdi[Indicator == "People using basic drinking water services (% of population)"])
  san = data.table(wdi[Indicator == "People using basic sanitation services (% of population)"])
  pop = data.table(wdi[Indicator == "Population ages 15-64, total"])
  elec = data.table(wdi[Indicator == "Access to electricity (% of population)"])
  
  # ---- prepare all of the data ----
  
  # remove unneeded columns from gci
  gci[, c("Placement", "Dataset", "GLOBAL_ID", "Code_GCR_2014.2015", "Series") := NULL]
  
  # collapse all country columns into a Country and value column
  gci = melt(gci, id.vars = c("Edition", "Series_unindented", "Attribute"), variable.name = "Country")
  
  # expand the Attribute column and fill it in with the value column
  gci = dcast(gci, Edition + Series_unindented + Country ~ Attribute, value.var = "value")
  
  # remove unneeded columns from gci
  gci[, c("Note", "Period", "Source", "Source date", "Rank") := NULL]
  
  # clean up the Series_unindented column
  gci[, Series_unindented := gsub(" ", ".", removeNumbers(removePunctuation(Series_unindented)))]
  gci[, Series_unindented := ifelse(substring(Series_unindented, 1, 1) == ".", substring(Series_unindented, 2, nchar(Series_unindented)), Series_unindented)]
  gci[, Series_unindented := gsub("th.pillar.", "", gsub("rd.pillar.", "", gsub("st.pillar.", "", gsub("nd.pillar.", "", gsub(".best", "", Series_unindented)))))]
  
  # replace the Edition column with a year column
  gci[, Year := as.numeric(substr(Edition, 1, 4))]
  gci[, Edition := NULL]
  
  # update the data types of gci
  gci[, Country := as.character(Country)]
  gci[, Value := as.numeric(Value)]
  
  # expand the Series_unindented column and fill it in with the Value column
  gci = dcast(gci, Year + Country ~ Series_unindented, value.var = "Value")
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  gci[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # remove the Indicator column in bud.imm
  bud.imm[, Indicator := NULL]
  
  # collapse all year columns into a Year and value column
  bud.imm = melt(bud.imm, id.vars = "Country", variable.name = "Year", value.name = "Line.in.Budget.for.Immunization")
  
  # update the year column in bud.imm
  bud.imm[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  bud.imm[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # remove the Indicator column in bud.sup
  bud.sup[, Indicator := NULL]
  
  # collapse all year columns into a Year and value column
  bud.sup = melt(bud.sup, id.vars = "Country", variable.name = "Year", value.name = "Line.in.Budget.for.Vaccine.Supplies")
  
  # update the year column in bud.sup
  bud.sup[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  bud.sup[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # remove the Indicator column in imm.exp.all
  imm.exp.all[, Indicator := NULL]
  
  # collapse all year columns into a Year and value column
  imm.exp.all = melt(imm.exp.all, id.vars = "Country", variable.name = "Year", value.name = "All.Expenditure.on.Immunization")
  
  # update the year column in imm.exp.all
  imm.exp.all[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  imm.exp.all[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # convert All.Expenditure.on.Immunization to a numeric data type
  imm.exp.all[, All.Expenditure.on.Immunization := as.numeric(All.Expenditure.on.Immunization)]
  
  # remove the Indicator column in imm.exp.gov
  imm.exp.gov[, Indicator := NULL]
  
  # collapse all year columns into a Year and value column
  imm.exp.gov = melt(imm.exp.gov, id.vars = "Country", variable.name = "Year", value.name = "Gov.Expenditure.on.Immunization")
  
  # update the year column in imm.exp.gov
  imm.exp.gov[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  imm.exp.gov[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # convert All.Expenditure.on.Immunization to a numeric data type
  imm.exp.gov[, Gov.Expenditure.on.Immunization := as.numeric(Gov.Expenditure.on.Immunization)]
  
  # remove the Indicator column in vac.exp.all
  vac.exp.all[, Indicator := NULL]
  
  # collapse all year columns into a Year and value column
  vac.exp.all = melt(vac.exp.all, id.vars = "Country", variable.name = "Year", value.name = "All.Expenditure.on.Vaccines")
  
  # update the year column in vac.exp.all
  vac.exp.all[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  vac.exp.all[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # convert All.Expenditure.on.Vaccines to a numeric data type
  vac.exp.all[, All.Expenditure.on.Vaccines := as.numeric(All.Expenditure.on.Vaccines)]
  
  # remove the Indicator column in vac.exp.gov
  vac.exp.gov[, Indicator := NULL]
  
  # collapse all year columns into a Year and value column
  vac.exp.gov = melt(vac.exp.gov, id.vars = "Country", variable.name = "Year", value.name = "Gov.Expenditure.on.Vaccines")
  
  # update the year column in vac.exp.gov
  vac.exp.gov[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  vac.exp.gov[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # convert All.Expenditure.on.Vaccines to a numeric data type
  vac.exp.gov[, Gov.Expenditure.on.Vaccines := as.numeric(Gov.Expenditure.on.Vaccines)]
  
  # remove the unneeded columns in hnp
  hnp[, c("Country_Code", "Indicator_Code") := NULL]
  
  # clean up the Indicator_Name column
  hnp[, Indicator_Name := gsub(" ", ".", removePunctuation(gsub("%", "Percent", gsub("\\+", "plus", gsub("-", "to", Indicator_Name)))))]
  
  # collapse all year columns into a Year and value column
  hnp = melt(hnp, id.vars = c("Country_Name", "Indicator_Name"), variable.name = "Year", value.name = "Value")
  
  # update the year column in hnp
  hnp[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  hnp[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country_Name)), fixed = FALSE))]
  hnp[, Country_Name := NULL]
  
  # expand the Indicator_Name column and fill it in with the Value column
  hnp = dcast(hnp, Country + Year ~ Indicator_Name, value.var = "Value")
  
  # create an indicator column
  imf[, Indicator := gsub(" ", ".", removePunctuation(paste0(Subject_Descriptor, Units, Scale)))]
  
  # remove the unneeded columns in imf
  imf[, c("Estimates.Start.After", "Subject_Descriptor", "Units", "Scale") := NULL]
  
  # collapse all year columns into a Year and value column
  imf = melt(imf, id.vars = c("Country", "Indicator"), variable.name = "Year", value.name = "Value")
  
  # update the year column in imf
  imf[, Year := as.numeric(substring(Year, 6, 9))]
  
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  imf[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  # update the Value column in imf
  imf[, Value := as.numeric(Value)]
  
  # expand the Indicator column and fill it in with the Value column
  imf = dcast(imf, Country + Year ~ Indicator, value.var = "Value")
  
  # lets put all of our tables into long format
  gdp = melt(gdp[,!"Indicator"], id.vars = "Country", 
             variable.name = "Year", value.name = "GDP")
  
  gni = melt(gni[,!"Indicator"], id.vars = "Country", 
             variable.name = "Year", value.name = "GNI")
  
  inflat = melt(inflat[,!"Indicator"], id.vars = "Country", 
                variable.name = "Year", value.name = "Inflation")
  
  unempl = melt(unempl[,!"Indicator"], id.vars = "Country", 
                variable.name = "Year", value.name = "Unemployment")
  
  gdppc = melt(gdppc[,!"Indicator"], id.vars = "Country", 
               variable.name = "Year", value.name = "GDPpc")
  
  gnipc = melt(gnipc[,!"Indicator"], id.vars = "Country", 
               variable.name = "Year", value.name = "GNIpc")
  
  int = melt(int[,!"Indicator"], id.vars = "Country", 
             variable.name = "Year", value.name = "InterestRate")
  
  gini = melt(gini[,!"Indicator"], id.vars = "Country", 
              variable.name = "Year", value.name = "GINI")
  
  life = melt(life[,!"Indicator"], id.vars = "Country", 
              variable.name = "Year", value.name = "LifeExpectancy")
  
  water = melt(water[,!"Indicator"], id.vars = "Country", 
               variable.name = "Year", value.name = "AccessToWater")
  
  san = melt(san[,!"Indicator"], id.vars = "Country", 
             variable.name = "Year", value.name = "AccessToSanitation")
  
  pop = melt(pop[,!"Indicator"], id.vars = "Country", 
             variable.name = "Year", value.name = "WorkingPopulation")
  
  elec = melt(elec[,!"Indicator"], id.vars = "Country", 
              variable.name = "Year", value.name = "AccessToElectricity")
  
  prim = melt(prim[,!"Indicator"], id.vars = "Country", 
              variable.name = "Year", value.name = "PrimaryEnrollment")
  
  sec = melt(sec[,!"Indicator"], id.vars = "Country", 
             variable.name = "Year", value.name = "SecondaryEnrollment")
  
  hdi = melt(hdi, id.vars = "Country", 
             variable.name = "Year", value.name = "HDI")
  
  # remove all letters from the Year and value columns in each table
  # update the Country column to have no spacing, punctuation, numbers, and all letters as uppercase
  gdp[, Year := as.numeric(gsub("[A-z]", "", Year))]
  gdp[, GDP := as.numeric(GDP)]
  gdp[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  gni[, Year := as.numeric(gsub("[A-z]", "", Year))]
  gni[, GNI := as.numeric(GNI)]
  gni[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  inflat[, Year := as.numeric(gsub("[A-z]", "", Year))]
  inflat[, Inflation := as.numeric(Inflation)]
  inflat[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  unempl[, Year := as.numeric(gsub("[A-z]", "", Year))]
  unempl[, Unemployment := as.numeric(Unemployment)]
  unempl[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  gdppc[, Year := as.numeric(gsub("[A-z]", "", Year))]
  gdppc[, GDPpc := as.numeric(GDPpc)]
  gdppc[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  gnipc[, Year := as.numeric(gsub("[A-z]", "", Year))]
  gnipc[, GNIpc := as.numeric(GNIpc)]
  gnipc[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  int[, Year := as.numeric(gsub("[A-z]", "", Year))]
  int[, InterestRate := as.numeric(InterestRate)]
  int[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  gini[, Year := as.numeric(gsub("[A-z]", "", Year))]
  gini[, GINI := as.numeric(GINI)]
  gini[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  life[, Year := as.numeric(gsub("[A-z]", "", Year))]
  life[, LifeExpectancy := as.numeric(LifeExpectancy)]
  life[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  water[, Year := as.numeric(gsub("[A-z]", "", Year))]
  water[, AccessToWater := as.numeric(AccessToWater)]
  water[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  san[, Year := as.numeric(gsub("[A-z]", "", Year))]
  san[, AccessToSanitation := as.numeric(AccessToSanitation)]
  san[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  pop[, Year := as.numeric(gsub("[A-z]", "", Year))]
  pop[, WorkingPopulation := as.numeric(WorkingPopulation)]
  pop[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  elec[, Year := as.numeric(gsub("[A-z]", "", Year))]
  elec[, AccessToElectricity := as.numeric(AccessToElectricity)]
  elec[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  prim[, Year := as.numeric(gsub("[A-z]", "", Year))]
  prim[, PrimaryEnrollment := as.numeric(PrimaryEnrollment)]
  prim[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  sec[, Year := as.numeric(gsub("[A-z]", "", Year))]
  sec[, SecondaryEnrollment := as.numeric(SecondaryEnrollment)]
  sec[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  hdi[, Year := as.numeric(gsub("[A-z]", "", Year))]
  hdi[, HDI := as.numeric(HDI)]
  hdi[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  abp[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
  
  
  # ---- define countries of interest and their labels ----
  
  # get all of the countries of interest from abp
  countries.of.interest = sort(unique(abp$Country))
  
  # get all unique country labels from out data
  all.country.labels = sort(unique(unname(unlist(list(countries.of.interest,
                                                      sort(unique(imf$Country)),
                                                      sort(unique(hnp$Country)),
                                                      sort(unique(vac.exp.gov$Country)),
                                                      sort(unique(vac.exp.all$Country)),
                                                      sort(unique(imm.exp.gov$Country)),
                                                      sort(unique(imm.exp.all$Country)),
                                                      sort(unique(bud.sup$Country)),
                                                      sort(unique(bud.imm$Country)),
                                                      sort(unique(gci$Country)),
                                                      sort(unique(sec$Country)),
                                                      sort(unique(prim$Country)),
                                                      sort(unique(elec$Country)),
                                                      sort(unique(pop$Country)),
                                                      sort(unique(san$Country)),
                                                      sort(unique(water$Country)),
                                                      sort(unique(life$Country)),
                                                      sort(unique(gini$Country)),
                                                      sort(unique(int$Country)),
                                                      sort(unique(gnipc$Country)),
                                                      sort(unique(gdppc$Country)),
                                                      sort(unique(unempl$Country)),
                                                      sort(unique(inflat$Country)),
                                                      sort(unique(gni$Country)),
                                                      sort(unique(gdp$Country)),
                                                      sort(unique(hdi$Country)))))))
  
  # define country labels to remove
  remove.lables = toupper(gsub(" ", "", 
                               removeNumbers(removePunctuation(c("SMALLSTATES",
                                                                 "LOWINCOME",
                                                                 "ISLEOFMAN",
                                                                 "IDAONLY",
                                                                 "IDAIBRDTOTAL",
                                                                 "IDABLEND",
                                                                 "IDATOTAL",
                                                                 "IBRDONLY",
                                                                 "SINTMAARTENDUTCHPART",
                                                                 "STMARTINFRENCHPART",
                                                                 "FRENCHPOLYNESIA",
                                                                 "CENTRALEUROPEANDTHEBALTICS",
                                                                 "BRITISHVIRGINISLANDS",
                                                                 "AVERAGEGCR",
                                                                 "AMERICANSAMOA",
                                                                 "CARIBBEANSMALLSTATES",
                                                                 "COMMONWEALTHOFINDEPENDENTSTATES",
                                                                 "EASTASIAPACIFICIDAIBRDCOUNTRIES",
                                                                 "EMERGINGANDDEVELOPINGASIA",                       
                                                                 "EMERGINGANDDEVELOPINGEUROPE",                     
                                                                 "FRAGILEANDCONFLICTAFFECTEDSITUATIONS",
                                                                 "HEAVILYINDEBTEDPOORCOUNTRIESHIPC",                
                                                                 "HIGHINCOMENONOECD",                               
                                                                 "HIGHINCOMEOECD",
                                                                 "HONGKONGSAR",                                     
                                                                 "HONGKONGSARCHINA",
                                                                 "LATEDEMOGRAPHICDIVIDEND",
                                                                 "LATINAMERICAANDTHECARIBBEAN",
                                                                 "LEASTDEVELOPEDCOUNTRIESUNCLASSIFICATION",
                                                                 "MIDDLEEASTNORTHAFRICAANDPAKISTAN",                
                                                                 "MIDDLEEASTNORTHAFRICAIDAIBRDCOUNTRIES",
                                                                 "NOTCLASSIFIED",                                           
                                                                 "OTHERSMALLSTATES",                                
                                                                 "PACIFICISLANDSMALLSTATES",
                                                                 "POSTDEMOGRAPHICDIVIDEND",                         
                                                                 "PREDEMOGRAPHICDIVIDEND",
                                                                 "STAGE",
                                                                 "SUBSAHARANAFRICA",                      
                                                                 "SUBSAHARANAFRICAEXCLUDINGHIGHINCOME",            
                                                                 "SUBSAHARANAFRICAIDAIBRDCOUNTRIES",
                                                                 "ADVANCEDECONOMIES",
                                                                 "Early-demographic dividend",
                                                                 "Middle income",
                                                                 "Low & middle income",
                                                                 "High income",
                                                                 "European Union",
                                                                 "Europe & Central Asia (IDA & IBRD countries)",
                                                                 "Europe & Central Asia (excluding high income)",
                                                                 "Euro area",
                                                                 "Europe & Central Asia",
                                                                 "Latin America & Caribbean",
                                                                 "Latin America & Caribbean (excluding high income)",
                                                                 "Latin America & the Caribbean (IDA & IBRD countries)",
                                                                 "Lower middle income",
                                                                 "South Asia",
                                                                 "South Asia (IDA & IBRD)",
                                                                 "North America",
                                                                 "OECD members",
                                                                 "World",
                                                                 "Arab World",
                                                                 "Middle East & North Africa",
                                                                 "Middle East & North Africa (excluding high income)",
                                                                 "East Asia & Pacific",
                                                                 "East Asia & Pacific (excluding high income)",
                                                                 "Upper middle income"))), 
                               fixed = FALSE))
  
  # update all.country.labels
  all.country.labels = all.country.labels[-which(all.country.labels %in% remove.lables)]
  
  # lets build a string distance matrix of our country names to find countries with multiple labels
  country.matrix = stringdistmatrix(a = all.country.labels, 
                                    b = all.country.labels, 
                                    useNames = "strings")
  
  # lets find the top 5 most common matches for each Country to see if there are any Countries with multiple labels
  country.matches = lapply(1:nrow(country.matrix), function(i)
  {
    # get the name of country i
    country = names(sort(country.matrix[i,])[1])
    
    # get the top 5 matches for country i
    matches = sort(country.matrix[i,])[2:6]
    
    # create a table of the top 5 matches for country i
    output = data.table(Id = i, Country = country, Match = as.numeric(matches), Distance = names(matches))
    
    return(output)
  })
  
  # combine the list of tables into 1 table
  country.matches = rbindlist(country.matches)
  
  # lets look at 20 countries at a time to see if there are any true matches
  country.matches[Id %in% 1:20]
  
  country.matches[Id %in% 261:280]
  
  country.matches[Id %in% 281:285]
  
  look.up = "YEMEN" 
  country.matches[grepl(look.up, Country)]
  abp[grepl(look.up, Country)]
  
  gci[grepl(look.up, Country)]
  imf[grepl(look.up, Country)]
  gdp[grepl(look.up, Country)]
  unempl[grepl(look.up, Country)]
  gini[grepl(look.up, Country)]
  
  # lets create a table of old and new country labels to make sure each country gets 1 unique label
  country.labels = data.table(Country = c("BAHAMASTHE", "THEBAHAMAS", "BOLIVIAPLURINATIONALSTATEOF", "BUKINAFASO", "CABOVERDE",
                                          "CÔTEDIVOIRE", "CDIVOIRE", "COTEDIVOIRE", 
                                          "CENTRALAFRICANREPUBLICTHE", "COMOROSTHE",
                                          "CONGODEMOCRATICREPUBLICOFTHE", "CONGODEMREP", "DEMOCRATICREPUBLICOFTHECONGOTHE", "CONGOREP", "REPUBLICOFCONGO", "CONGOTHE",
                                          "CZECHREPUBLICTHE",
                                          "KOREADEMPEOPLE'SREP", "DEMOCRATICPEOPLESREPUBLICOFKOREATHE", 
                                          "KOREAREP", "KOREA", "REPUBLICOFKOREATHE", "KOREAREPUBLICOF",
                                          "DOMINICANREPUBLICTHE",
                                          "EGYPTARABREP", 
                                          "GAMBIATHE", "THEGAMBIA", 
                                          "PAPANEWGUINEA",
                                          "IRANISLAMICREP", "IRANISLAMICREPUBLICOF", "ISLAMICREPUBLICOFIRAN", 
                                          "KYRGYZREPUBLIC", 
                                          "LAOPDR", "LAOPEOPLESDEMOCRATICREPUBLICTHE", 
                                          "MARSHALLISLANDSTHE",
                                          "MACEDONIAFYR", "FYRMACEDONIA", 
                                          "MICRONESIAFEDERATEDSTATESOF", "MICRONESIAFEDSTS", 
                                          "NETHERLANDSTHE", "THENETHERLANDS", 
                                          "NIGERTHE",
                                          "PHILIPPINESTHE", 
                                          "MOLDOVA", "REPUBLICOFMOLDOVATHE", 
                                          "RUSSIA", "RUSSIANFEDERATIONTHE", 
                                          "STKITTSANDNEVIS", "STLUCIA", "STVINCENTANDTHEGRENADINES",
                                          "STOMNDPRIPE", 
                                          "SLOVAKREPUBLIC", 
                                          "SUDANTHE", 
                                          "SYRIA", "SYRIANARABREPUBLICTHE",
                                          "UNITEDARABEMIRATESTHE", 
                                          "UNITEDKINGDOM", "UNITEDKINGDOMOFGREATBRITAINANDNORTHERNIRELANDTHE", 
                                          "TANZANIA", "TANZANIAUNITEDREPUBLICOF", 
                                          "UNITEDSTATES", "UNITEDSTATESOFAMERICATHE", 
                                          "VANATU", 
                                          "VENEZUELARB", "VENEZUELABOLIVARIANREPUBLICOF", 
                                          "YEMENREP"),
                              
                              new = c("BAHAMAS", "BAHAMAS", "BOLIVIA", "BURKINAFASO", "CAPEVERDE", 
                                      "CTEDIVOIRE", "CTEDIVOIRE", "CTEDIVOIRE",
                                      "CENTRALAFRICANREPUBLIC", "COMOROS",
                                      "DEMOCRATICREPUBLICOFTHECONGO", "DEMOCRATICREPUBLICOFTHECONGO", "DEMOCRATICREPUBLICOFTHECONGO", "CONGO", "CONGO", "CONGO",
                                      "CZECHREPUBLIC", 
                                      "DEMOCRATICPEOPLESREPUBLICOFKOREA", "DEMOCRATICPEOPLESREPUBLICOFKOREA",
                                      "REPUBLICOFKOREA", "REPUBLICOFKOREA", "REPUBLICOFKOREA", "REPUBLICOFKOREA",
                                      "DOMINICANREPUBLIC",
                                      "EGYPT",
                                      "GAMBIA", "GAMBIA",
                                      "PAPUANEWGUINEA",
                                      "IRAN", "IRAN", "IRAN", 
                                      "KYRGYZSTAN", 
                                      "LAOPEOPLESDEMOCRATICREPUBLIC", "LAOPEOPLESDEMOCRATICREPUBLIC", 
                                      "MARSHALLISLANDS",
                                      "THEFORMERYUGOSLAVREPUBLICOFMACEDONIA", "THEFORMERYUGOSLAVREPUBLICOFMACEDONIA", 
                                      "MICRONESIA", "MICRONESIA", 
                                      "NETHERLANDS", "NETHERLANDS", 
                                      "NIGER", 
                                      "PHILIPPINES", 
                                      "REPUBLICOFMOLDOVA", "REPUBLICOFMOLDOVA", 
                                      "RUSSIANFEDERATION", "RUSSIANFEDERATION", 
                                      "SAINTKITTSANDNEVIS", "SAINTLUCIA", "SAINTVINCENTANDTHEGRENADINES", 
                                      "SAOTOMEANDPRINCIPE", 
                                      "SLOVAKIA", 
                                      "SUDAN", 
                                      "SYRIANARABREPUBLIC", "SYRIANARABREPUBLIC",
                                      "UNITEDARABEMIRATES", 
                                      "UNITEDKINGDOMOFGREATBRITAINANDNORTHERNIRELAND", "UNITEDKINGDOMOFGREATBRITAINANDNORTHERNIRELAND",
                                      "UNITEDREPUBLICOFTANZANIA", "UNITEDREPUBLICOFTANZANIA",
                                      "UNITEDSTATESOFAMERICA", "UNITEDSTATESOFAMERICA", 
                                      "VANUATU", 
                                      "VENEZUELA", "VENEZUELA", 
                                      "YEMEN"))
  
  # ---- combine all of the data ------
  
  # lets only look at data between 2006 and 2016
  # lets create a single data table to combine all tables together
  dat = data.table(expand.grid(Country = all.country.labels, Year = 2006:2016))
  
  # give dat an id column
  dat[, id := 1:nrow(dat)]
  
  # join all of our data onto dat
  setkey(dat, Country, Year)
  setkey(bud.imm, Country, Year)
  dat = bud.imm[dat]
  
  setkey(dat, Country, Year)
  setkey(bud.sup, Country, Year)
  dat = bud.sup[dat]
  
  setkey(dat, Country, Year)
  setkey(imm.exp.all, Country, Year)
  dat = imm.exp.all[dat]
  
  setkey(dat, Country, Year)
  setkey(imm.exp.gov, Country, Year)
  dat = imm.exp.gov[dat]
  
  setkey(dat, Country, Year)
  setkey(vac.exp.all, Country, Year)
  dat = vac.exp.all[dat]
  
  setkey(dat, Country, Year)
  setkey(vac.exp.gov, Country, Year)
  dat = vac.exp.gov[dat]
  
  setkey(dat, Country, Year)
  setkey(gci, Country, Year)
  dat = gci[dat]
  
  setkey(dat, Country, Year)
  setkey(hnp, Country, Year)
  dat = hnp[dat]
  
  setkey(dat, Country, Year)
  setkey(imf, Country, Year)
  dat = imf[dat]
  
  setkey(dat, Country, Year)
  setkey(gdp, Country, Year)
  dat = gdp[dat]
  
  setkey(dat, Country, Year)
  setkey(gni, Country, Year)
  dat = gni[dat]
  
  setkey(dat, Country, Year)
  setkey(inflat, Country, Year)
  dat = inflat[dat]
  
  setkey(dat, Country, Year)
  setkey(unempl, Country, Year)
  dat = unempl[dat]
  
  setkey(dat, Country, Year)
  setkey(gdppc, Country, Year)
  dat = gdppc[dat]
  
  setkey(dat, Country, Year)
  setkey(gnipc, Country, Year)
  dat = gnipc[dat]
  
  setkey(dat, Country, Year)
  setkey(int, Country, Year)
  dat = int[dat]
  
  setkey(dat, Country, Year)
  setkey(gini, Country, Year)
  dat = gini[dat]
  
  setkey(dat, Country, Year)
  setkey(life, Country, Year)
  dat = life[dat]
  
  setkey(dat, Country, Year)
  setkey(water, Country, Year)
  dat = water[dat]
  
  setkey(dat, Country, Year)
  setkey(san, Country, Year)
  dat = san[dat]
  
  setkey(dat, Country, Year)
  setkey(pop, Country, Year)
  dat = pop[dat]
  
  setkey(dat, Country, Year)
  setkey(elec, Country, Year)
  dat = elec[dat]
  
  setkey(dat, Country, Year)
  setkey(prim, Country, Year)
  dat = prim[dat]
  
  setkey(dat, Country, Year)
  setkey(sec, Country, Year)
  dat = sec[dat]
  
  setkey(dat, Country, Year)
  setkey(hdi, Country, Year)
  dat = hdi[dat]
  
  # join country.labels onto dat to consolidate country names
  setkey(country.labels, Country)
  setkey(dat, Country)
  dat = country.labels[dat]
  
  # update the Country column to take on any avalble values in the new column of dat
  dat[, Country := ifelse(is.na(new), Country, new)]
  
  # remove new from dat
  dat[, new := NULL]
  
  # update Line.in.Budget variables to be binary
  dat[, Line.in.Budget.for.Vaccine.Supplies := ifelse(Line.in.Budget.for.Vaccine.Supplies == "Yes", 1, ifelse(Line.in.Budget.for.Vaccine.Supplies == "No", 0, NA))]
  dat[, Line.in.Budget.for.Immunization := ifelse(Line.in.Budget.for.Immunization == "Yes", 1, ifelse(Line.in.Budget.for.Immunization == "No", 0, NA))]
  
  # update dat to average the variable values for Countrys that had multiple labels
  dat = data.table(dat[, lapply(.SD, mean, na.rm = TRUE), by = .(Country, Year)])
  
  # replace all NaN with NA in dat
  dat = as.matrix(dat)
  dat[is.nan(dat)] = NA
  dat = data.table(dat)
  
  # update dat to be a numeric data table
  dat.update = as.matrix(dat[, !"Country"])
  class(dat.update) = "numeric"
  dat = cbind(dat[,.(Country)], dat.update)
  
  # lets remove duplicate variables that weren't reported on as much
  names(dat)[which(grepl("GDP", names(dat)))]
  nrow(na.omit(dat[,.(GDP)]))/nrow(dat)
  
  dat = dat[,!c("GDP.PPP.as.share..of.world.total",                     
                "GDP.PPP.billions",                                     
                "GDP.US.billions",                                      
                "GDP.per.capita.US",
                "GNI.per.capita.Atlas.method.current.US",
                "Unemployment",                                     
                "Unemployment.ratePercent.of.total.labor.force",    
                "Unemployment.female.Percent.of.female.labor.force",
                "Unemployment.male.Percent.of.male.labor.force",
                "Inflation.annual..change"), 
            with = FALSE]
  
  # check to see that all ABP countries are in dat
  all(unique(abp$Country) %in% unique(dat$Country))
  
  # update dat to only contain ABP countries
  dat = dat[Country %in% unique(abp$Country)]
  
  # get the 4 market system assignment for each country from abp
  four.markets = data.table(abp[Markets == 4, .(Country, Market = MarketID)])
  four.markets[, Market := ifelse(Market == 1, "HIC", 
                                  ifelse(Market == 2, "HMIC", 
                                         ifelse(Market == 3, "LMIC", "LIC")))]
  
  # join Market from four.markets onto dat
  setkey(dat, Country)
  setkey(four.markets, Country)
  dat = four.markets[dat]
  
  # order dat by id
  dat = dat[order(id)]
  
  # remove id
  dat[, id := NULL]
  
  # write out dat
  write.csv(dat, "ABP-Country-Data-2006-2016.csv", row.names = FALSE)
  
  # create a copy of dat
  dat.full = data.table(dat)
  
  # remove unneeded objects
  rm(bud.imm, bud.sup, imm.exp.all, imm.exp.gov, vac.exp.all, 
     vac.exp.gov, gci, hnp, imf, dat.update, edu, elec, four.markets,       
     gdp, gdppc, all.country.labels, countries.of.interest, country.matches, 
     country.matrix, gni, gnipc, inflat, int, life, look.up,
     pop, prim, remove.lables, san, sec, unempl, water, wdi, country.labels,
     gini, hdi)
  
  # clean up workspace
  gc()
  
} else
{
  # import all of the data
  dat = data.table(read.csv("ABP-Country-Data-2006-2016.csv", stringsAsFactors = FALSE))
  dat.full = data.table(dat)
  abp = data.table(read.csv("ABP-countries.csv", stringsAsFactors = FALSE))
  abp[, Country := toupper(gsub(" ", "", removeNumbers(removePunctuation(Country)), fixed = FALSE))]
}

# do we need to clean the data or has it already been done?
clean.data = FALSE

if(clean.data)
{
  # ---- remove variables that wouldn't be indicative of risk ----
  
  # lets look at the variables in dat and remove any variables that appear irrelevant to a country's vaccine budget
  names(dat)
  
  # build a vector of variables to remove
  remove.vars = c("Adolescent.fertility", "Age.at.first.marriage", "HIV", 
                  "AIDS", "ARI.treatment", "Age.dependency.", "Age.population.",
                  "Antiretroviral.therapy", "Cause.of.death.by.nontocommunicable",
                  "Births.attended.by.", "Cause.of.death.by.injury", "HIVAIDS",
                  "Completeness.of.birth.registration", "Completeness.of.death.registration",
                  "Condom.use", "Consumption.of.iodized.salt", "Contraceptive.prevalence",
                  "Demand.for.family.planning", "Diabetes.prevalence", "Exclusive.breastfeeding",
                  "Female.headed.households", "Female.population", "Hospital.beds.per.1000.people",
                  "Improved.sanitation", "Improved.water.source", "Infant.and.young.child", 
                  "Labor.force.", "Lifetime.risk.of.maternal", "Lowtobirthweight", 
                  "Male.population", "Malnutrition", "Maternal",
                  "Net.migration", "Number.of.", "People.practicing", "People.using",
                  "Physicians.per.", "Population.ages.", "Population.female", "Population.male",
                  "Postnatal.", "Poverty.headcount.", "Pregnant.", "anemia", "Prevalence.of.",
                  "Primary.completion.", "Ratio.of.school.attendance.", "Ratio.of.young.",
                  "Public.spending.on.education", "Risk.of.", "Rural.population", 
                  "Rural.poverty.", "School.enrollment.", "Sex.ratio.", "nonagricultural",
                  "Smoking.prevalence", "Specialist.surgical.", "Suicide.mortality", 
                  "Teenage.mothers.", "Total.alcohol.consumption", "Unmet.need.for.contraception",
                  "Urban.", "insecticidetotreated", "SPFansidar", "Wanted.fertility.",
                  "A.Quantity.of.education", "A.Public.institutions", "Agricultural.policy.costs.",
                  "Available.airline.seat.kmweek.millions", "B.Primary.education",
                  "B.Private.institutions", "B.Quality.of.demand.conditions", "B.Quality.of.education",
                  "Burden.of.", "FDI.", "C.Onthejob.training", "Cooperation.in.laboremployer.relations.",
                  "Domestic.market.size.index..", "Effect.of.taxation.", "Efficacy.of.",
                  "Efficiency.of.", "Extent.of.")
  
  # lets build a vector of the variables we want to keep
  keep.vars = names(dat)
  
  # lets remove some variables
  for(i in remove.vars)
  {
    keep.vars = keep.vars[!grepl(i, keep.vars)]
  }
  
  # build a vector of more variables to remove
  keep.vars
  remove.vars = c("Current.account.balancePercent.of.GDP", 
                  "General.government.revenuePercent.of.GDP", "General.government.total.expenditurePercent.of.GDP",
                  "Community.health.workers.per.1000.people", "Diarrhea.treatment.", "Fertility.rate.",
                  "Health.expenditure.total.Percent.", "Life.expectancy.at.birth.male.years", "Life.expectancy.at.birth.female.years",
                  "Literacy.rate.youth", "Literacy.rate.adult.male", "Literacy.rate.adult.female",
                  "Malaria.cases.reported", "Mortality.caused.by.road.", "Mortality.from.CVD.cancer.", 
                  "Nurses.and.midwives.", "Outtooftopocket.health.expenditure.Percent.of.private.expenditure.on.health",
                  "Tuberculosis.case.", "Tuberculosis.death.", "Tuberculosis.treatment.",
                  "Unemployment.male.", "Unemployment.female.", "A.Competition", "A.Domestic.market.size",
                  "A.Efficiency", "A.Flexibility", "Availability.of.research.and.training.services.", 
                  "B.Efficient.use.of.talent", "B.Electricity.and.telephony.infrastructure", "B.Foreign.market.size",
                  "B.ICT.use.", "Basic.requirements", "Business.costs.of.crime.and.violence.", "Business.costs.of.terrorism.",
                  "Business.sophistication.", "Buyer.sophistication.", "Capacity.for.innovation.", "Company.spending.on.RD.", 
                  "Control.of.international.distribution.","Country.capacity.to.attract", "Domestic.competition.",
                  "Ease.of.access.to.loans.", "Efficiency.enhancers", "Exports.as.a.percentage.of.GDP",
                  "Financial.market.development", "Financing.through.local.equity.market.", "Firmlevel.technology.absorption.",
                  "Fixed.broadband.Internet.subscriptions.pop", "Fixed.telephone.lines.pop", "Flexibility.of.wage.determination.",
                  "Foreign.competition.", "Foreign.market.size.index..", "GDP.PPP.as.share..of.world.total", "GDP.PPP.billions",
                  "General.government.debt..GDP", "Goods.market.efficiency", "Gross.national.savings..GDP", "Health.and.primary.education",
                  "Higher.education.and.training", "Hiring.and.firing.practices.", "Imports.as.a.percentage.of.GDP",
                  "Individuals.using.Internet.", "Innovation", "Innovation.and.sophistication.factors", "Institutions", "Intellectual.property.protection.",
                  "Intensity.of.local.competition.", "Internet.access.in.schools.", "Int.l.Internet.bandwidth.kbs.per.user",
                  "Judicial.independence.", "Labor.market.efficiency", "Legal.rights.index..", "Life.expectancy.years", 
                  "Local.supplier.quality.", "Local.supplier.quantity.", "Market.size", 
                  "Mobile.broadband.subscriptions.pop", "Mobile.telephone.subscriptions.pop", "Nature.of.competitive.advantage.",
                  "No.days.to.start.a.business", "No.procedures.to.start.a.business", "Organized.crime.",
                  "PCT.patents.applicationsmillion.pop", "Pay.and.productivity.", "Population.millions", "Primary.education.enrollment.net.", 
                  "Production.process.sophistication.", "Property.rights", "Protection.of.minority.shareholders..interests.",
                  "Quality.of.electricity.supply.", "Quality.of.management.schools.", "Quality.of.math.and.science.education.",
                  "Quality.of.primary.education.", "Quality.of.railroad.infrastructure.", "Quality.of.roads.", "Quality.of.scientific.research.institutions.", 
                  "Quality.of.the.education.system.", "Redundancy.costs.weeks.of.salary", "Regulation.of.securities.exchanges.", "Reliability.of.police.services.",
                  "Reliance.on.professional.management.", "Secondary.education.enrollment.gross.", "Security", 
                  "State.of.cluster.development.", "Strength.of.investor.protection..",
                  "Tertiary.education.enrollment.gross.", "Total.tax.rate..profits", "Trade.tariffs..duty",
                  "Undue.influence", "Universityindustry.collaboration.in.RD.", "Value.chain.breadth.", 
                  "Venture.capital.availability.", "Willingness.to.delegate.authority.", "Women.in.labor.force.ratio.to.men")
  
  # lets remove some variables
  for(i in remove.vars)
  {
    keep.vars = keep.vars[!grepl(i, keep.vars)]
  }
  
  # build a vector of more variables to remove
  keep.vars
  remove.vars = c("Current.account.balanceUS.dollarsBillions",
                  "Unemployment.ratePercent.of.total.labor.force", ".of.financial.services.", "Country.capacity.to.retain.talent.",
                  "Quality.of.air.transport.infrastructure.", "Quality.of.port.infrastructure.", "Availability.of.scientists.and.engineers.",
                  "Cause.of.death.by.communicable.diseases.and.maternal.prenatal.and.nutrition.conditions.Percent.of.total",
                  "Children.with.fever.receiving.antimalarial.drugs.Percent.of.children.under.age.5.with.fever",
                  "Accountability", "Population.growth.annual.Percent",
                  "Gov.t.procurement.of.advanced.tech.products.", "Availability.of.latest.technologies.",
                  "Quality.of.overall.infrastructure.", "Technological.readiness",
                  "Soundness.of.banks.", "Corporate.ethics", "Ethical.behavior.of.firms.")
  
  # lets remove some variables
  for(i in remove.vars)
  {
    keep.vars = keep.vars[!grepl(i, keep.vars)]
  }
  
  # build a vector of more variables to remove
  keep.vars
  remove.vars = c("Health.expenditure.private.Percent.of.GDP", "Health.expenditure.private.Percent.of.total.health.expenditure",
                  "Health.expenditure.total.current.US", "Life.expectancy.at.birth.total.years",
                  "Literacy.rate.adult.total.Percent.of.people.ages.15.and.above", "Unemployment.total.Percent.of.total.labor.force",
                  "Vitamin.A.supplementation.coverage.rate.Percent.of.children.ages.6to59.months", 
                  "A.Technological.adoption", "A.Transport.infrastructure",
                  "Infrastructure", "Global.Competitiveness.Index", "Gross.national.savingsPercent.of.GDP", 
                  "Health.expenditure.public.Percent.of.GDP", "Incidence.of.", "Newborns.protected.",
                  "Strength.of.auditing.and.reporting.standards", "Degree.of.customer.orientation.",
                  "Effectiveness.of.antimonopoly.policy.", "Public.trust.in.politicians.",
                  "Government.efficiency", "Infant.mortality.deaths.live.births", 
                  "Survival.to.age.65.", "Inflation.annual..change", "Health.expenditure.per.capita.current.US",
                  "Health.expenditure.per.capita.PPP", "Favoritism.in.decisions.of.government.officials.",
                  "General.government.net.lendingborrowingPercent.of.GDP", "SecondaryEnrollment",                                                      
                  "PrimaryEnrollment", "AccessToElectricity", "WorkingPopulation", "AccessToSanitation",                                                       
                  "AccessToWater", "External.resources.for.health.Percent.of.total.expenditure.on.health", "Business.impact.of.tuberculosis.")
  
  # lets remove some variables
  for(i in remove.vars)
  {
    keep.vars = keep.vars[!grepl(i, keep.vars)]
  }
  
  # ---- check how well remaining indicators were reported on ----
  
  # our remaining variables are meant to indicate how risky a government is with their spending such that they may defect on their budget for preventative care
  # lets update dat to only contain the varaibles in keep.vars
  dat = dat[, keep.vars, with = FALSE]
  
  # lets look at the countries and years with values in Ethics.and.corruption
  dat.check = na.omit(data.table(dat[,.(Country, Year, Ethics.and.corruption)]))
  dat.check = dat.check[order(Year, Country)]
  
  # lets only keep observations for these countries and years
  dat.check[, CountryYear := paste0(Country, Year)]
  dat[, CountryYear := paste0(Country, Year)]
  dat = dat[CountryYear %in% dat.check$CountryYear]
  dat[, CountryYear := NULL]
  
  # lets look at how many missing values there are in each column of dat
  missing.values = sapply(1:ncol(dat), function(j)
  {
    return(setNames(nrow(na.omit(dat[, j, with = FALSE], invert = TRUE)), names(dat)[j]))
  })
  
  # lets see the top half of variables withpout missing values
  missing.values = sort(missing.values)
  
  # lets only keep the variables with less missing values than Line.in.Budget.for.Immunization
  keep.vars = c("Country", "Market", "Year", "Ethics.and.corruption", "LifeExpectancy",
                "B.Trustworthiness.and.confidence", "Birth.rate.crude.per.1000.people",
                "Health.expenditure.public.Percent.of.total.health.expenditure",
                "Health.expenditure.public.Percent.of.government.expenditure",
                "Transparency.of.government.policymaking.",
                "Wastefulness.of.government.spending.",
                "Macroeconomic.environment",
                "Mortality.rate.infant.per.1000.live.births", "Immunization.DPT.Percent.of.children.ages.12to23.months",
                "Outtooftopocket.health.expenditure.Percent.of.total.expenditure.on.health", 
                "Population.total", "HDI")
  
  # determine which varaibles are the final variables
  final.var = names(missing.values) %in% keep.vars
  
  # lets plot the missing.values to see how much data is missing in each column
  # windows()
  # par(oma = c(16, 1, 1, 1), mar = c(5, 8 , 4, 2))
  # barplot(100 * (missing.values / nrow(dat)), col = "black", main = "Missing Data", xlab = "", ylab = "Percentage\n", las = 2, cex = 2, cex.main = 2, cex.lab = 2, cex.axis = 2, cex.names = 1.5)
  
  # remove missing values from dat
  dat = dat[, keep.vars, with = FALSE]
  dat = na.omit(dat)
  dat = dat[order(Year, Country)]
  
  # compute BirthCohort
  dat[, BirthCohort := Birth.rate.crude.per.1000.people * (Population.total / 1000)]
  
  # compute CohortMortality
  dat[, CohortMortality := Mortality.rate.infant.per.1000.live.births * (BirthCohort / 1000)]
  
  # compute BirthCohortMortality
  dat[, BirthCohortMortality := 100 * (CohortMortality / BirthCohort)]
  
  # remove BirthCohort, CohortMortality, Population.total, Birth.rate.crude.per.1000.people, and Mortality.rate.infant.per.1000.live.births
  dat = dat[, !c("BirthCohort", "CohortMortality", "Population.total", "Birth.rate.crude.per.1000.people", "Mortality.rate.infant.per.1000.live.births"), with = FALSE]
  
  # write out dat
  write.csv(dat, "budget-risk-data.csv", row.names = FALSE)
  
  # remove unneeded objects
  rm(dat.check, i, keep.vars, missing.values, remove.vars, final.var)
  
} else
{
  # import the data
  dat = data.table(read.csv("budget-risk-data.csv", stringsAsFactors = FALSE))
}

}

# -----------------------------------------------------------------------------------
# ---- Compute Risk Index with Principal Components ---------------------------------
# -----------------------------------------------------------------------------------

{

# do we need to do PCA or did we already do it?
do.pca = FALSE

if(do.pca)
{
  # ---- compute risk with PCA ----
  
  # create a copy of dat
  dat.copy = data.table(dat)
  # dat = data.table(dat.copy)
  
  # get a set of indicators with alot of multi-collinearity
  dat = dat[, !c("Country", "Market", "Year",
                 "Immunization.DPT.Percent.of.children.ages.12to23.months",
                 "Health.expenditure.public.Percent.of.total.health.expenditure",
                 "Transparency.of.government.policymaking.",
                 # "BirthCohortMortality", 
                 "LifeExpectancy", 
                 # "B.Trustworthiness.and.confidence",
                 "Health.expenditure.public.Percent.of.government.expenditure",
                 # "Macroeconomic.environment",
                 "Wastefulness.of.government.spending.",
                 "Outtooftopocket.health.expenditure.Percent.of.total.expenditure.on.health"), 
            with = FALSE]
  
  # the rules for keeping a PC: if eignevalue >= 1
  # so, only use original variables that can be combined into a single PC1 that explains the vast majority of variance, so you can use PC1 as your index
  # look at the loadings of PC1 (ie. the correlation between PC1 and the original variables) to see if PC1 makes sense as an index
  # run a regression model to see if PC1 as an index makes sense: mod = lm(PC1 ~ orginal.variables)
  
  # compute correlations and plot them
  windows()
  corrplot(cor(dat), 
           diag = FALSE, tl.offset = 0.5, tl.srt = 5, tl.cex = 1, # tl.pos = "n",
           # p.mat = cors.sig$p, insig = "blank",
           type = "lower", method = "square", order = "FPC", # addgrid.col = "transparent", 
           col = c("red", "blue"), bg = "white", cl.cex = 1.25)
  
  # compute covariances
  covs = cov(dat)
  
  # reorder the position of the variables according to the ordering used in corrplot
  # covs = covs[rownames(cor.plot), rownames(cor.plot)]
  
  # only keep the upper triangle of covs
  covs[lower.tri(covs)] = NA
  
  # convert matrix into long format
  covs = data.table(melt(covs, na.rm = TRUE))
  
  # make Var1 and Var2 back into factor variables
  covs[, Var1 := factor(Var1, levels = unique(covs$Var1))]
  covs[, Var2 := factor(Var2, levels = unique(covs$Var2))]
  
  # plot the covariances
  cov.plot = ggplot(covs, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "black") +
    scale_fill_gradient2(low = "red", high = "blue", mid = "white", midpoint = 0) +
    ggtitle("Covariance Matrix Plot") + 
    labs(fill = "Covariance") + 
    theme_bw(10) + 
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"), 
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          panel.border = element_blank(),
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
    guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1))
  
  cov.plot
  
  # should we keep highly correlated variables?
  keep.cor = TRUE
  
  if(!keep.cor)
  {
    # find out which variables are highly correlated (by magnitude) and remove them
    find.vars = findCorrelation(cors, cutoff = 0.9, names = TRUE, exact = TRUE)
    
    # remove columns from dat according to find.vars
    if(length(find.vars) > 0) dat = dat[, !find.vars, with = FALSE]
  }
  
  # run PCA on dat
  pca.mod = summary(prcomp(dat, scale. = TRUE))
  
  # extract the pricipal components data set
  pca.dat = data.table(pca.mod$x)
  
  # extract the proportion of variance explained by each PC
  pca.var = data.table(PC = colnames(pca.mod$importance), t(pca.mod$importance))
  
  # give column names without spaces to pca.var
  setnames(pca.var, gsub(" ", "", names(pca.var)))
  
  # compute the eigenvalues of each PC
  pca.var[, eigen := Standarddeviation^2]
  
  # extract the loadings matrix
  pca.load = pca.mod$rotation
  
  # look at the loadings and eigenvalues
  pca.load
  pca.var
  
  # keep enough PCs with an eigenvalue >= 1
  keep.pc = unname(unlist(pca.var[eigen >= 1, PC]))
  
  # update pca.dat to only have PCs in keep.pc
  pca.dat = pca.dat[, keep.pc, with = FALSE]
  
  # reset dat
  dat = data.table(dat.copy)
  
  # attach pca.dat to dat
  dat = cbind(dat, pca.dat)
  
  # check the relationship of PC1 with our variables
  corrplot(cor(dat[,!c("Country", "Market", "Year")]), 
           diag = FALSE, tl.offset = 0.5, tl.srt = 5, tl.cex = 1, # tl.pos = "n",
           # p.mat = cors.sig$p, insig = "blank",
           type = "lower", method = "square", order = "FPC", # addgrid.col = "transparent", 
           col = c("red", "blue"), bg = "white", cl.cex = 1.25)
  
  # put -PC1 on a 0 to 100 scale
  # higher values indicate higher risk
  dat[, risk := rescale(-PC1, to = c(0, 100))]
  
  # lets pick a span to exponentially average risk (this gives more recent years more weight)
  exp.span = 2
  
  # exponentially average the risk together by country to create our index
  index = data.table(dat[, .(risk = tail(EMA(risk, n = 1,  ratio = 2 / (exp.span + 1)), 1)), 
                         by = .(Country)])
  
  # ---- impute risk ----
  
  # lets look at which indicators have been most reported on
  reports = data.table(Indicator = names(dat.full)[-(1:3)],
                       Completion = sapply(4:ncol(dat.full), function(j) nrow(na.omit(dat.full[, j, with = FALSE])) / nrow(dat.full)))
  
  # order reports by Completion
  reports = reports[order(Completion, decreasing = TRUE)]
  
  # look at the top reports
  head(reports[Completion >= 0.94], 100)
  
  # get indicators that may correlate with risk
  ind1 = na.omit(data.table(dat.full[,.(Country, Year, GNIpc = GNI / Population.total)]))
  ind2 = na.omit(data.table(dat.full[,.(Country, Year, Account.balance.Percent.of.GDP = Current.account.balancePercent.of.GDP)]))
  ind3 = na.omit(data.table(dat.full[,.(Country, Year, Immunization.measles = Immunization.measles.Percent.of.children.ages.12to23.months)]))
  ind4 = na.omit(data.table(dat.full[,.(Country, Year, Immunization.DPT = Immunization.DPT.Percent.of.children.ages.12to23.months)]))
  
  # look at how many years there are in indicators
  data.table(Years = sort(unique(ind1$Year)))
  data.table(Years = sort(unique(ind2$Year)))
  data.table(Years = sort(unique(ind3$Year)))
  data.table(Years = sort(unique(ind4$Year)))
  
  # lets pick a span to exponentially avergae our data (this gives more recent data more weight)
  exp.span = 3
  
  # exponentially average indicators that correlate with risk
  ind1 = data.table(ind1[, .(GNIpc = tail(EMA(GNIpc, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)])
  ind2 = data.table(ind2[, .(Account.balance.Percent.of.GDP = tail(EMA(Account.balance.Percent.of.GDP, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)])
  ind3 = data.table(ind3[, .(Immunization.measles = tail(EMA(Immunization.measles, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)])
  ind4 = data.table(ind4[, .(Immunization.DPT = tail(EMA(Immunization.DPT, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)])
  
  # get the four market system from abp
  four.markets = data.table(abp[Markets == 4, .(Country, 
                                                Market = MarketID)])
  
  # update the Market column in four.markets
  four.markets[, Market := factor(ifelse(Market == 1, "HIC", 
                                  ifelse(Market == 2, "HMIC", 
                                         ifelse(Market == 3, "LMIC", "LIC"))), 
                                  levels = c("LIC", "LMIC", "HMIC", "HIC"))]
  
  # join index and indicators onto four.markets
  setkey(four.markets, Country)
  setkey(index, Country)
  setkey(ind1, Country)
  setkey(ind2, Country)
  setkey(ind3, Country)
  setkey(ind4, Country)
  
  four.markets = index[four.markets]
  four.markets = ind1[four.markets]
  four.markets = ind2[four.markets]
  four.markets = ind3[four.markets]
  four.markets = ind4[four.markets]
  
  # split up the part of four.markets that is missing all data
  four.markets.missing = data.table(four.markets[is.na(GNIpc) & 
                                                 is.na(Immunization.measles) & 
                                                 is.na(Account.balance.Percent.of.GDP) & 
                                                 is.na(Immunization.DPT) &
                                                 is.na(risk)])
  
  # remove the missing part from four.markets
  four.markets = data.table(four.markets[!(is.na(GNIpc) & 
                                           is.na(Immunization.measles) & 
                                           is.na(Account.balance.Percent.of.GDP) & 
                                           is.na(Immunization.DPT) &
                                           is.na(risk))])
  
  # check correlation of indicators and risk in four.markets
  corrplot(cor(na.omit(four.markets[,!c("Country", "Market")])), 
           diag = FALSE, tl.offset = 0.5, tl.srt = 5, tl.cex = 1, # tl.pos = "n",
           # p.mat = cors.sig$p, insig = "blank",
           type = "lower", method = "square", order = "FPC", # addgrid.col = "transparent", 
           col = c("red", "blue"), bg = "white", cl.cex = 1.25)
  
  # split up four.markets into keep and impute:
  # keep will retain the variables we want but don't want to use for imputation
  # impute will retain the variables we want to use for imputation
  four.markets.keep = data.table(four.markets[, .(Country, Market)])
  four.markets.impute = data.table(four.markets[, !c("Country", "Market")])
  four.markets.impute = data.table(cbind(four.markets.impute, model.matrix(~., data = four.markets.keep[,.(Market)])[,-1]))
  
  # use random forests to impute the missing values
  four.markets.impute = missRanger(four.markets.impute, num.trees = 100, seed = 42, num.threads = 1)
  
  # update four.markets
  four.markets = data.table(cbind(four.markets.keep, four.markets.impute[, !c("MarketLMIC", "MarketHMIC", "MarketHIC")]))
  
  # compute the quintiles of risk
  quintiles = as.numeric(quantile(four.markets$risk, probs = seq(0, 1, 0.2)))
  
  # create classes for risk based on the quintiles
  four.markets[, risk.class := ifelse(risk <= quintiles[2], 1, 
                                 ifelse(risk <= quintiles[3], 2, 
                                        ifelse(risk <= quintiles[4], 3, 
                                               ifelse(risk <= quintiles[5], 4, 5))))]
  four.markets[, risk.class := factor(risk.class, levels = 1:5)]
  
  # give four.markets.missing the same column order as four.markets
  four.markets.missing[, risk.class := NA]
  setcolorder(four.markets.missing, names(four.markets))
  
  # add four.markets.missing to four.markets
  four.markets = rbind(four.markets, four.markets.missing)
  
  # convert four.markets into long format
  four.markets.long = data.table(four.markets[GNIpc <= 40000])
  four.markets.long = data.table(melt(four.markets.long, id.vars = c("Country", "Market", "risk", "risk.class"), 
                                 variable.name = "Indicator", value.name = "Value"))
  
  # make violin plots for each indicator by risk class and Market
  plot.risk = ggplot(na.omit(four.markets.long), aes(x = Market, y = Value, color =risk.class , fill =risk.class)) + # , group = Market)) + 
    geom_point(position = position_jitterdodge(jitter.width = 1/3), alpha = 1/2, size = 3) + 
    # geom_violin(alpha = 1/2) +
    # geom_boxplot(alpha = 1/2) + 
    scale_color_manual(values = c("blue", "orchid3", "lightseagreen", "darkorange2", "red")) + 
    scale_fill_manual(values = c("blue", "orchid3", "lightseagreen", "darkorange2", "red")) + 
    # scale_color_brewer(palette = "RdBu", direction = -1) + 
    # scale_fill_brewer(palette = "RdBu", direction = -1) + 
    # scale_y_continuous(labels = percent) + 
    facet_wrap(~Indicator, scales = "free_y") + 
    ggtitle("Risk Index v. Indicators") + 
    labs(x = "Income Level", y = "Value", color = "Risk Level:", fill = "Risk Level:") + 
    theme_bw(25) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          # axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          axis.title.y = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 0, alpha = 1), nrow = 1, byrow = TRUE))
  
  plot.risk
  
  # update index
  index = data.table(four.markets)
  
  # order index by risk
  index = index[order(risk, decreasing = TRUE)]
  
  # give index an id column to preserve row order
  index[, id := 1:nrow(index)]
  
  # get indicators that correlate with risk regardless of how complete they are
  indicators = na.omit(data.table(dat.full[,.(Country, Market, Year,
                                              BirthCohort = Birth.rate.crude.per.1000.people * (Population.total / 1000),
                                              LifeExpectancy,
                                              Out.of.pocket.Percent.of.total.health.expenditure = Outtooftopocket.health.expenditure.Percent.of.total.expenditure.on.health,
                                              Transparency.of.government.policymaking., 
                                              Health.expenditure.per.capita = Health.expenditure.per.capita.current.US,
                                              Wastefulness.of.government.spending.)]))
  
  # look at how many years there are in indicators
  data.table(Years = sort(unique(indicators$Year)))
  
  # lets pick a span to exponentially avergae our data (this gives more recent data more weight)
  exp.span = 2
  
  # exponentially average indicators that correlate with risk
  indicators = indicators[, .(LifeExpectancy = tail(EMA(LifeExpectancy, n = 1,  ratio = 2 / (exp.span + 1)), 1),
                              BirthCohort = tail(EMA(BirthCohort, n = 1,  ratio = 2 / (exp.span + 1)), 1),
                              Out.of.pocket.Percent.of.total.health.expenditure = tail(EMA(Out.of.pocket.Percent.of.total.health.expenditure, n = 1,  ratio = 2 / (exp.span + 1)), 1),
                              Transparency.of.government.policymaking. = tail(EMA(Transparency.of.government.policymaking., n = 1,  ratio = 2 / (exp.span + 1)), 1),
                              Health.expenditure.per.capita = tail(EMA(Health.expenditure.per.capita, n = 1,  ratio = 2 / (exp.span + 1)), 1),
                              Wastefulness.of.government.spending. = tail(EMA(Wastefulness.of.government.spending., n = 1,  ratio = 2 / (exp.span + 1)), 1)),
                          by = .(Country)]
  
  # join more indicators onto index
  setkey(index, Country)
  setkey(indicators, Country)
  index = indicators[index]
  
  # sort index by id then remove id
  index = index[order(id)]
  index[, id := NULL]
  
  # check correlation of indicators and risk in index
  corrplot(cor(na.omit(index[,!c("Country", "Market", "risk.class")])), 
           diag = FALSE, tl.offset = 0.5, tl.srt = 5, tl.cex = 1, # tl.pos = "n",
           # p.mat = cors.sig$p, insig = "blank",
           type = "lower", method = "square", order = "FPC", # addgrid.col = "transparent", 
           col = c("red", "blue"), bg = "white", cl.cex = 1.25)
  
  # check out index
  na.omit(index[,.(Country, Market, risk, risk.class)])
  index[Country == "UNITEDSTATESOFAMERICA",.(Country, Market, risk, risk.class)]
  
  # update dat to only include indicators that built risk
  dat = dat[,.(Country, Market, Year, risk, PC1, Ethics.and.corruption, B.Trustworthiness.and.confidence,
               HDI, Macroeconomic.environment, BirthCohortMortality)]
  
  # look at all country pairs
  pairs = data.table(expand.grid(Country1 = index$Country, 
                                 Country2 = index$Country))
  
  # give pairs an id number to preserve row order
  pairs[, id := 1:nrow(pairs)]
  
  # get a table of Countries and their Market
  market1 = data.table(index[,.(Country1 = Country, Market1 = as.numeric(factor(Market, levels = c("HIC", "HMIC", "LMIC", "LIC"))))])
  market2 = data.table(index[,.(Country2 = Country, Market2 = as.numeric(factor(Market, levels = c("HIC", "HMIC", "LMIC", "LIC"))))])
  
  # join market1 and market2 onto pairs
  setkey(pairs, Country1)
  setkey(market1, Country1)
  pairs = market1[pairs]
  
  setkey(pairs, Country2)
  setkey(market2, Country2)
  pairs = market2[pairs]
  
  # order pairs by id and then remove it
  pairs = pairs[order(id)]
  pairs[, id := NULL]
  
  # compute market distance between all country pairs
  pairs[, distance := abs(Market1 - Market2)]
  
  # only keep distance > 2
  pairs = pairs[distance > 2]
  
  # assign a penalty column and remove Market information
  pairs[, penalty := 2]
  pairs[, c("Market1", "Market2", "distance") := NULL]
  
  # write out index, dat, and pairs
  write.csv(index, "budget-risk-index.csv", row.names = FALSE)
  write.csv(dat, "budget-risk-data.csv", row.names = FALSE)
  write.csv(pairs, "budget-risk-penalty.csv", row.names = FALSE)
  
  # remove unneeded objects
  rm(dat.copy, cov.plot, exp.span, covs, four.markets, 
     four.markets.missing, keep.pc, pca.dat, pca.load , pca.mod, pca.var, indicators,
     keep.cor, plot.risk, quintiles, four.markets.impute, four.markets.keep, four.markets.long,
     reports, market1, market2, pairs)
  
} else
{
  # import the data
  dat = data.table(read.csv("budget-risk-data.csv", stringsAsFactors = FALSE))
  index = data.table(read.csv("budget-risk-index.csv", stringsAsFactors = FALSE))
}

}

# -----------------------------------------------------------------------------------
# ---- Test Correlation between Income and Vaccine Expenditure ----------------------
# -----------------------------------------------------------------------------------

{

# get GNI and vaccine expenditure data
x = na.omit(data.table(dat.full[,.(Country, Year,
                                   Gov.Expenditure.on.Vaccines, All.Expenditure.on.Vaccines,
                                   GNI)]))

# get the unique years
yrs = unique(x$Year)

# compute avg GNI across all years in the data
x = x[, .(GNI = mean(GNI), 
          Gov.Vac.Exp = mean(Gov.Expenditure.on.Vaccines),
          All.Vac.Exp = mean(All.Expenditure.on.Vaccines)),
      by = .(Country)]

# create 4 groups based on quartiles of GNI
cuts = c(quantile(x$GNI, 0.25),
         quantile(x$GNI, 0.50),
         quantile(x$GNI, 0.75))

# create the 4 groups
g1 = x[GNI < cuts[1]]
g2 = x[GNI >= cuts[1] & GNI < cuts[2]]
g3 = x[GNI >= cuts[2] & GNI < cuts[3]]
g4 = x[GNI >= cuts[3]]

# test is the correlation between GNI and vaccine expenditure is significant
cor.test(g1$Gov.Vac.Exp, g1$GNI)
cor.test(g2$Gov.Vac.Exp, g2$GNI)
cor.test(g3$Gov.Vac.Exp, g3$GNI)
cor.test(g4$Gov.Vac.Exp, g4$GNI)
cor.test(x$Gov.Vac.Exp, x$GNI)

# plot each of the 4 groups
windows()
pairs(g1[,.(Gov.Vac.Exp,
            All.Vac.Exp,
            GNI)], pch = 16)

pairs(g2[,.(Gov.Vac.Exp,
            All.Vac.Exp,
            GNI)], pch = 16)

pairs(g3[,.(Gov.Vac.Exp,
            All.Vac.Exp,
            GNI)], pch = 16)

pairs(g4[,.(Gov.Vac.Exp,
            All.Vac.Exp,
            GNI)], pch = 16)

pairs(x[,.(Gov.Vac.Exp,
           All.Vac.Exp,
           GNI)], pch = 16)

}

# -----------------------------------------------------------------------------------
# ---- Study Vaccine Expenditure Trend ----------------------------------------------
# -----------------------------------------------------------------------------------

{

# get the vaccine expenditure data
vac = na.omit(data.table(dat.full[,.(Country, Year, Gov.Expenditure.on.Vaccines, All.Expenditure.on.Vaccines)]))

# order by Country and Year
vac = vac[order(Country, Year)]

# compute the mean change in vaccine expenditure for each Country
vac = vac[,.(gov.exp.avg.diff = mean(diff(Gov.Expenditure.on.Vaccines)), 
             all.exp.avg.diff = mean(diff(All.Expenditure.on.Vaccines)),
             avg.gov.portion = mean(Gov.Expenditure.on.Vaccines / All.Expenditure.on.Vaccines),
             gov.exp.avg = mean(Gov.Expenditure.on.Vaccines),
             all.exp.avg = mean(All.Expenditure.on.Vaccines),
             years = paste(Year, collapse = ",")),
          by = .(Country)]

# extract the starting and ending year of the trend for each country
vac[, start.year := substr(years, 1, 4)]
vac[, end.year := substr(years, nchar(years) - 3, nchar(years))]
vac[, years := NULL]

# compute if a governemnt is decreasing their expenditure on vaccines
vac[, gov.decreasing := ifelse(gov.exp.avg.diff < 0, 1, 0)]

# lets see how many governemnts are decreasing theri expenditure on vaccines
table(vac$gov.decreasing)

# compute if a governemnt is responsible for at least 90% of their total expenditure on vaccines
vac[, gov.responsible := ifelse(avg.gov.portion >= 0.95, 1, 0)]

# lets see how many governemnts are responsible for the vast majority of their expenditure on vaccines
table(vac$gov.responsible)

# plot a histogram of avg.gov.portion
ggplot(data = vac, aes(x = avg.gov.portion)) + 
  geom_histogram(color = "white") + 
  ggtitle("Average Government Expenditure on Vaccines\nas a Portion of Total Vaccine Expenditure") + 
  labs(x = "Portion", y = "Countries") + 
  scale_x_continuous(labels = percent) + 
  theme_bw(25) + 
  theme(legend.position = "none", 
        legend.key.size = unit(.25, "in"), 
        plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

}

# -----------------------------------------------------------------------------------
# ---- Comparing GINI and Risk Index ------------------------------------------------
# -----------------------------------------------------------------------------------

{

# get the gini data from dat.full
gini = na.omit(data.table(dat.full[,.(Country, Year, GINI)]))

# exponentially average the gini together by country
exp.span = 2
gini = data.table(gini[, .(GINI = tail(EMA(GINI, n = 1,  ratio = 2 / (exp.span + 1)), 1)), 
                       by = .(Country)])

# give index an id column tp preserve row order
index[, id := 1:nrow(index)]

# join gini onto index
setkey(index, Country)
setkey(gini, Country)
index = gini[index]

# order index by id then remove id
index = index[order(id)]
index[, id := NULL]

# make violin plots for each indicator by risk class and Market
plot.risk = ggplot(na.omit(index[,.(Country, Market, risk.class, GINI)]), aes(x = Market, y = GINI, color = risk.class, fill = risk.class)) + 
  geom_point(position = position_jitterdodge(), alpha = 1/2, size = 3) + 
  # geom_violin(alpha = 1/2) +
  # geom_boxplot(alpha = 1/2) + 
  scale_color_manual(values = c("blue", "orchid3", "darkturquoise", "darkorange2", "red")) + 
  scale_fill_manual(values = c("blue", "orchid3", "darkturquoise", "darkorange2", "red")) + 
  # scale_color_brewer(palette = "RdBu", direction = -1) + 
  # scale_fill_brewer(palette = "RdBu", direction = -1) + 
  # scale_y_continuous(labels = percent) + 
  # facet_wrap(~Indicator, scales = "free_y") + 
  ggtitle("Risk Index v. GINI") + 
  labs(x = "Income Level", y = "GINI", color = "Risk Level:", fill = "Risk Level:") + 
  theme_bw(25) +
  theme(legend.position = "top", 
        legend.key.size = unit(.25, "in"),
        plot.title = element_text(hjust = 0.5),
        # axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  guides(color = guide_legend(override.aes = list(size = 10, linetype = 0, alpha = 1), nrow = 1, byrow = TRUE))

plot.risk

# get risk and GINI
gini.test = data.table(na.omit(index[,.(risk, GINI)]))

# test correlation
cor.test(gini.test$GINI, gini.test$risk)

# remove objects we no longer need
rm(cuts, g1, g2, g3, g4, gini, gini.test, ind1, ind2, ind3, ind4,
   vac, x, yrs)

# clean up RAM
gc()

}

# -----------------------------------------------------------------------------------
# ---- Computing Reservation Prices for Vaccines per Country ------------------------
# -----------------------------------------------------------------------------------

{

compute.prices = FALSE

if(compute.prices)
{
  # ---- import paho estimates ----
  
  # import the PAHO data
  paho = data.table(read.csv("PAHO-Estimates.csv", stringsAsFactors = FALSE))
  
  # collapse all Vaccine columns into a Vaccine and Price column
  paho = melt(paho, id.vars = c("Country"), variable.name = "Vaccine", value.name = "Price")
  paho[, Vaccine := as.character(Vaccine)]
  
  # heres our antigens of interest
  antigens = c("DTP", "DaTP", "DTaP", "DwTP", "DTwP", "HepB", "Hib", "IPV", "OPV")
  
  # heres our available vaccines
  vaccines = unique(paho$Vaccine)
  
  # only keep the vaccines with any of the antigens of interest
  keep.vaccines = vaccines[which(sapply(vaccines, function(v) any(sapply(antigens, function(a) grepl(a, v)))))]
  
  # lets pick common labels for duplicate vaccines in keep.vaccines
  keep.vaccines
  update.vaccines = c(rep("DTwP", 4),
                      rep("DTaP", 2),
                      rep("HepB", 7),
                      rep("Hib", 5),
                      rep("IPV", 5),
                      rep("OPV", 6),
                      rep("DTaP-IPV", 2),
                      rep("DTwP-Hib", 2),
                      "DTaP-Hib-IPV",
                      rep("DTwP-HepB-Hib", 8),
                      "DTaP-HepB-Hib-IPV")
  
  # combine keep.vaccines and update.vaccines into 1 table to update paho vaccine names
  update.paho = data.table(Vaccine = keep.vaccines, new = update.vaccines)
  
  # join update.paho onto paho
  setkey(update.paho, Vaccine)
  setkey(paho, Vaccine)
  paho = na.omit(update.paho[paho])
  
  # set Vaccine to be new and remove new
  paho[, Vaccine := new]
  paho[, new := NULL]
  
  # aggregate prices by Country
  paho = paho[, .(Price = mean(Price)), by = .(Country, Vaccine)]
  
  # add a market and year column to paho
  paho[, Year := 2018]
  paho[, Market := "PAHO"]
  
  # aggregate prices by vaccine
  paho.prices = data.table(paho[, .(Price = mean(Price)), by = .(Market, Vaccine, Year)])
  
  # ---- preparing market prices ----
  
  # import the market data
  market.prices = data.table(read.csv("Market-Prices.csv", stringsAsFactors = FALSE))
  link.prices = data.table(read.csv("Linksbridge-Vaccine-Prices.csv", stringsAsFactors = FALSE))
  vaccine.names = data.table(read.csv("vaccine-names.csv", stringsAsFactors = FALSE))
  
  # combine the price data
  prices = rbindlist(list(data.table(market.prices[,.(Market, Vaccine, Year, Price = PriceHigh)]),
                          data.table(link.prices),
                          data.table(paho.prices)))
  
  # update some of the group names
  prices[, Market := ifelse(Market == "UNICEF SD", "UNICEF", 
                            ifelse(Market == "High Income & Super HICs", "HIC", 
                                   ifelse(Market == "China & Upper Middle Income", "China",
                                          ifelse(Market == "UMIC", "HMIC",
                                                 Market))))]
  
  # here's our antigens of interest
  antigens = c("DTP", "DaTP", "DTaP", "DwTP", "DTwP", "HepB", "Hib", "IPV", "OPV")
  
  # heres our available vaccines
  vaccines = unique(prices$Vaccine, 
                    vaccine.names$Vaccine)
  
  # only keep the vaccines with any of the antigens of interest
  keep.vaccines = vaccines[which(sapply(vaccines, function(v) any(sapply(antigens, function(a) grepl(a, v)))))]
  
  # get rid of these vaccines in keep.vaccines
  remove.vaccines = c("HepA-HepB", "HepA-HepB (adult)", "HepB (adult)", "Hib-MenCY")
  keep.vaccines = keep.vaccines[-which(keep.vaccines %in% remove.vaccines)]
  
  # update prices to only contain the vaccines of interest
  prices = prices[Vaccine %in% keep.vaccines]
  
  # there is little vaccine pricing information for LIC and LMIC
  # so lets combine them with UNICEF and PAHO respectively
  prices[, Market := ifelse(Market == "LIC", "UNICEF", 
                            ifelse(Market == "LMIC", "PAHO", Market))]
  
  # order prices by Market, Vaccine, and Year
  # remove missing values as well
  prices = na.omit(prices[order(Market, Vaccine, Year)])
  
  # remove some DTP-Hib-IPV anomolies from HIC
  prices = prices[!(Market == "HIC" & Vaccine == "DTaP-HepB-Hib-IPV" & (Price < 35 | Price > 50))]
  prices = prices[!(Market == "HIC" & Vaccine == "DTP-HepB-Hib-IPV" & (Price < 35 | Price > 50))]
  
  # remove some DTP-Hib-IPV anomolies from HMIC
  prices = prices[!(Market == "HMIC" & Vaccine == "DTP-Hib-IPV" & (Price < 18 | Price > 30))]
  
  # remove some DTP-Hib-IPV anomolies from HMIC
  prices = prices[!(Market == "HMIC" & Vaccine == "DTP-Hib-IPV" & Price < 13)]
  
  # remove some DTP-Hib-IPV anomolies from HIC
  prices = prices[!(Market == "HIC" & Vaccine == "DTP-Hib-IPV" & (Price < 18 | Price > 30))]
  
  # remove some DTP-IPV anomolies from HIC
  prices = prices[!(Market == "HIC" & Vaccine == "DTP-IPV" & (Price <= 13 | Price >= 24))]
  
  # remove some DTP-IPV anomolies from HMIC
  prices = prices[!(Market == "HMIC" & Vaccine == "DTP-IPV" & (Price <= 7 | Price >= 14))]
  
  # remove some IPV anomolies from HIC
  prices = prices[!(Market == "HIC" & Vaccine == "IPV" & (Price < 5.443 | Price >= 12))]
  
  # remove some IPV anomolies from HMIC
  prices = prices[!(Market == "HMIC" & Vaccine == "IPV" & Price < 3)]
  
  # remove some IPV anomolies from PAHO
  prices = prices[!(Market == "PAHO" & Vaccine == "IPV" & (Price <= 1 | Price >= 5))]
  
  # remove some Hib anomolies from HMIC
  prices = prices[!(Market == "HMIC" & Vaccine == "Hib" & (Price < 4 | Price > 10))]
  
  # remove some Hib anomolies from HIC
  prices = prices[!(Market == "HIC" & Vaccine == "Hib" & (Price < 6 | Price > 14))]
  
  # remove some HepB anomolies from PAHO
  prices = prices[!(Market == "PAHO" & Vaccine == "HepB" & Price >= 0.3)]
  
  # remove some HepB anomolies from HMIC
  prices = prices[!(Market == "HMIC" & Vaccine == "HepB" & (Price < 0.4 | Price > 2.25))]
  
  # remove some HepB anomolies from HIC
  prices = prices[!(Market == "HIC" & Vaccine == "HepB" & (Price < 3.9 | Price > 6.525))]
  
  # remove DTwP-Hib vaccines from UNICEF becuase their pricing isn't recent
  prices = prices[!(Market == "UNICEF" & Vaccine == "DTwP-Hib")]
  
  # remove DTP vaccines from PAHO becuase they already differentiate by DTaP and DTwP
  prices = prices[!(Market == "PAHO" & Vaccine == "DTP")]
  
  # remove some DTP anomolies from HMIC
  prices = prices[!(Market == "HMIC" & Vaccine == "DTP")]
  
  # remove some DTP anomolies from HIC
  prices = prices[!(Market == "HIC" & Vaccine == "DTP" & Price %in% c(31, 2.6, 9.1))]
  
  # rename the DTP vaccines in HIC to DTaP vaccines
  prices[Market == "HIC" & Vaccine == "DTP", Vaccine := "DTaP"]
  
  # rename bOPV and tOPV vaccines to be OPV because the pricing is already similar
  prices[Vaccine %in% c("bOPV", "tOPV"), Vaccine := "OPV"]
  
  # update remaining vaccines with DTP in them to DTaP
  prices[, Vaccine := gsub("DTP", "DTaP", Vaccine)]
  
  # aggregate prices by Market, Vaccine, and Year
  prices = prices[,.(Price = mean(Price, na.rm = TRUE)), by = .(Market, Vaccine, Year)]
  
  # lets pick a span to exponentially average risk (this gives more recent years more weight)
  exp.span = 2
  
  # exponentially average the Price together by market and vaccine to create market prices
  prices = data.table(prices[, .(Price = tail(EMA(Price, n = 1,  ratio = 2 / (exp.span + 1)), 1)), 
                             by = .(Market, Vaccine)])
  
  # lets create all possible combinations of Markets and Vaccines
  DT = data.table(expand.grid(Market = unique(prices$Market), Vaccine = unique(prices$Vaccine)))
  
  # join prices onto DT
  setkey(prices, Market, Vaccine)
  setkey(DT, Market, Vaccine)
  DT = prices[DT]
  
  # update antigens
  antigens = c("DTaP", "DTwP", "HepB", "Hib", "IPV", "OPV")
  
  # lets create binary variables indicating which antigens are in each vaccine
  BV = data.table(do.call("cbind", lapply(antigens, function(a) 
    sapply(1:nrow(DT), function(i) 
      as.numeric(grepl(a, DT$Vaccine[i]))))))
  
  setnames(BV, antigens)
  
  # add BV to DT
  DT = cbind(DT, BV)
  
  # assign a ranking to each vaccine within its market
  DT[, Rank := unlist(lapply(unique(DT$Market), function(m) rank(unname(unlist(DT[Market == m, .(Price)])), na.last = "keep")))]
  
  # order DT by Market and Rank to see the cost heirarchy of vaccines
  DT = DT[order(Market, Rank)]
  
  # check out adjacent markets and develop a cost table for vaccines
  DT[Market %in% c("UNICEF", "PAHO")]
  DT[Market %in% c("PAHO", "HMIC")]
  DT[Market %in% c("HMIC", "HIC")]
  
  # set up the cost classification
  cost = data.table(Vaccine = c("OPV", "HepB", "DTwP", "DTwP-HepB", "Hib", "DTwP-Hib", "DTwP-HepB-Hib", "IPV", "DTaP", "DTaP-Hib", "DTaP-IPV", "DTaP-Hib-IPV", "DTaP-HepB-IPV", "DTaP-HepB-Hib-IPV"),
                    Cost = c("lowest", "lowest", "low", "low", "mid", "mid", "mid", "high", "high", "high", "high", "high", "highest", "highest"),
                    Premium = c(rep(0, 7), rep(1, 7)))
  
  # give cost an Order column
  cost[, Order := 1:nrow(cost)]
  
  # join cost onto DT
  setkey(cost, Vaccine)
  setkey(DT, Vaccine)
  DT = cost[DT]
  
  # make Cost into a factor
  DT[, Cost := factor(Cost, levels = c("lowest", "low", "mid", "high", "highest"))]
  
  # make Market into a factor
  DT[, Market := factor(Market, levels = c("UNICEF", "PAHO", "HMIC", "China", "HIC", "USA"))]
  
  # order DT by Market and Order
  DT = DT[order(Market, Order)]
  
  # get USA as its own table
  usa = data.table(DT[Market == "USA", !"Rank"])
  
  # get China as its own table
  china = data.table(DT[Market == "China", !"Rank"])
  
  # set the OPV price in HMIC to be the same as in China
  DT[Market == "HMIC" & Vaccine == "OPV", 
     Price := unname(unlist(DT[Market == "China" & Vaccine == "OPV", .(Price)]))]
  
  # remove china as it doesn't add any new vaccine information
  DT = DT[!(Market == "China")]
  
  # lead the prices by the number of vaccines
  # this allows for lower income markets prices to be compared with upper icome
  DT[, leadPrice := shift(Price, n = length(unique(DT$Vaccine)), type = "lead")]
  DT[, lagPrice := shift(Price, n = length(unique(DT$Vaccine)), type = "lag")]
  
  DT[, leadPrice2 := shift(Price, n = 2 * length(unique(DT$Vaccine)), type = "lead")]
  DT[, lagPrice2 := shift(Price, n = 2 * length(unique(DT$Vaccine)), type = "lag")]
  
  # compute the percent difference in pricing
  DT[, leadPriceChange := (Price / leadPrice) - 1]
  DT[, lagPriceChange := (Price / lagPrice) - 1]
  
  DT[, leadPriceChange2 := (Price / leadPrice2) - 1]
  DT[, lagPriceChange2 := (Price / lagPrice2) - 1]
  
  # compute the median price change between markets
  DT[, leadPriceChange.mid := median(leadPriceChange, na.rm = TRUE),
     by = .(Market)]
  
  DT[, lagPriceChange.mid := median(lagPriceChange, na.rm = TRUE),
     by = .(Market)]
  
  DT[, leadPriceChange2.mid := median(leadPriceChange2, na.rm = TRUE),
     by = .(Market)]
  
  DT[, lagPriceChange2.mid := median(lagPriceChange2, na.rm = TRUE),
     by = .(Market)]
  
  # replace NA's in leadPriceChange with leadPriceChange.mid
  DT[is.na(leadPriceChange), leadPriceChange := leadPriceChange.mid]
  DT[is.na(lagPriceChange), lagPriceChange := lagPriceChange.mid]
  
  DT[is.na(leadPriceChange2), leadPriceChange2 := leadPriceChange2.mid]
  DT[is.na(lagPriceChange2), lagPriceChange2 := lagPriceChange2.mid]
  
  # replace NaN's in lagPriceChange & leadPriceChange with 0's because there is no change
  DT[is.nan(lagPriceChange), lagPriceChange := 0]
  DT[is.nan(leadPriceChange), leadPriceChange := 0]
  
  DT[is.nan(lagPriceChange2), lagPriceChange2 := 0]
  DT[is.nan(leadPriceChange2), leadPriceChange2 := 0]
  
  # remove the usa section from DT
  DT = DT[!(Market == "USA")]
  
  # split up the DT into keep and impute:
  # keep will retain the variables we want but don't want to use for imputation
  # impute will retain the variables we want to use for imputation
  DT.keep = data.table(DT[, .(Market, Cost, Vaccine, DTaP, DTwP, HepB, Hib, IPV, OPV)])
  DT.impute = data.table(DT[, .(Price, Cost, Premium, HI = as.numeric(Market == "HMIC" | Market == "HIC"), leadPrice, leadPriceChange)])
  
  # expand Cost into binary variables
  DT.impute = cbind(DT.impute[,!c("Cost")], model.matrix(~., data = DT.impute[,.(Cost)])[,-1])
  
  # remove USA and China
  # DT.impute = DT.impute[, !c("MarketChina", "MarketUSA")]
  # DT.impute = DT.impute[, !c("MarketPAHO", "MarketHMIC", "MarketHIC")]
  
  # use random forests to impute the missing values
  DT.impute = missRanger(DT.impute, num.trees = 200, min.node.size = 6, seed = 42, num.threads = 1)
  
  # make DT.impute into a data table
  DT.impute = data.table(DT.impute)
  
  # update DT
  DT = cbind(DT.keep, DT.impute, Imputed = is.na(DT$Price))
  
  # lets look at the imputed market prices for every vaccine and make sure they are tiered
  v = 9
  DT[Vaccine == DT$Vaccine[v], .(Market, Vaccine, Price, Imputed)]
  
  # update poorly imputed prices to be tiered
  DT[Market == "PAHO" & Vaccine == "DTwP-HepB", Price := (1 + abs(unname(unlist(DT[Market == "UNICEF" & Vaccine == "DTwP-HepB", .(leadPriceChange)])))) * unname(unlist(DT[Market == "UNICEF" & Vaccine == "DTwP-HepB", .(Price)]))]
  DT[Market == "UNICEF" & Vaccine == "DTwP-HepB-Hib", Price := (1 + leadPriceChange) * unname(unlist(DT[Market == "PAHO" & Vaccine == "DTwP-HepB-Hib", .(Price)]))]
  DT[Market == "HMIC" & Vaccine == "DTaP", Price := mean(c(unname(unlist(DT[Market == "PAHO" & Vaccine == "DTaP", .(Price)])),
                                                           unname(unlist(DT[Market == "HIC" & Vaccine == "DTaP", .(Price)]))))]
  
  DT[Market == "PAHO" & Vaccine == "DTaP-Hib", Price := (1 + abs(unname(unlist(DT[Market == "UNICEF" & Vaccine == "DTaP-Hib", .(leadPriceChange)])))) * unname(unlist(DT[Market == "UNICEF" & Vaccine == "DTaP-Hib", .(Price)]))]
  DT[Market == "UNICEF" & Vaccine == "DTaP-HepB-IPV", Price := (1 + leadPriceChange) * unname(unlist(DT[Market == "PAHO" & Vaccine == "DTaP-HepB-IPV", .(Price)]))]
  DT[Market == "UNICEF" & Vaccine == "DTaP-HepB-Hib-IPV", Price := (1 + leadPriceChange) * unname(unlist(DT[Market == "PAHO" & Vaccine == "DTaP-HepB-Hib-IPV", .(Price)]))]
  
  # update DT to just contain the variables of interest
  DT = DT[, .(Market, Vaccine, Price, Imputed, Cost, DTaP, DTwP, HepB, Hib, IPV, OPV)]
  
  # export DT, china, and usa
  write.csv(DT, "Market-Reservation-Prices-Nick.csv", row.names = FALSE)
  write.csv(china, "China-Reservation-Prices-Nick.csv", row.names = FALSE)
  write.csv(usa, "USA-Reservation-Prices-Nick.csv", row.names = FALSE)
  
  # ---- preparing vaccine expenditure ----
  
  # lets look at which indicators have been most reported on
  reports = data.table(Indicator = names(dat.full)[-(1:3)],
                       Completion = sapply(4:ncol(dat.full), function(j) nrow(na.omit(dat.full[, j, with = FALSE])) / nrow(dat.full)))
  
  # order reports by Completion
  reports = reports[order(Completion, decreasing = TRUE)]
  
  # look at the top reports
  head(reports[Completion > 0.93], 50)
  
  # get indicators that correlate with vaccine expenditure
  ind1 = na.omit(data.table(dat.full[,.(Country, Year, GNI)]))
  ind2 = na.omit(data.table(dat.full[,.(Country, Year, Account.balance = Current.account.balanceUS.dollarsBillions)]))
  ind3 = na.omit(data.table(dat.full[,.(Country, Year, Gov.Budget = Gov.Expenditure.on.Vaccines)]))
  ind4 = na.omit(data.table(dat.full[,.(Country, Year, Population = Population.total)]))
  ind5 = na.omit(data.table(dat.full[,.(Country, Year, Birth.rate = Birth.rate.crude.per.1000.people)]))
  ind6 = na.omit(data.table(dat.full[,.(Country, Year, Death.rate = Mortality.rate.infant.per.1000.live.births)]))
  
  # lets pick a span to exponentially avergae our data (this gives more recent data more weight)
  exp.span = 3
  
  # exponentially average indicators
  ind1 = ind1[, .(GNI = tail(EMA(GNI, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)]
  ind2 = ind2[, .(Account.balance = tail(EMA(Account.balance, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)]
  ind3 = ind3[, .(Gov.Budget = tail(EMA(Gov.Budget, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)]
  ind4 = ind4[, .(Population = tail(EMA(Population, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)]
  ind5 = ind5[, .(Birth.rate = tail(EMA(Birth.rate, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)]
  ind6 = ind6[, .(Death.rate = tail(EMA(Death.rate, n = 1,  ratio = 2 / (exp.span + 1)), 1)), by = .(Country)]
  
  # get the four market system from abp
  four.markets = data.table(abp[Markets == 4, .(Country, 
                                                Market = MarketID)])
  
  # update the Market column in four.markets
  four.markets[, Market := factor(ifelse(Market == 1, "HIC", 
                                         ifelse(Market == 2, "HMIC", 
                                                ifelse(Market == 3, "LMIC", "LIC"))), 
                                  levels = c("LIC", "LMIC", "HMIC", "HIC"))]
  
  # join indicators onto four.markets
  setkey(four.markets, Country)
  setkey(ind1, Country)
  setkey(ind2, Country)
  setkey(ind3, Country)
  setkey(ind4, Country)
  setkey(ind5, Country)
  setkey(ind6, Country)
  
  four.markets = ind1[four.markets]
  four.markets = ind2[four.markets]
  four.markets = ind3[four.markets]
  four.markets = ind4[four.markets]
  four.markets = ind5[four.markets]
  four.markets = ind6[four.markets]
  
  # compute Birth.cohort
  four.markets[, Birth.cohort := Birth.rate * (Population / 1000)]
  
  # compute Cohort.mortality
  four.markets[, Cohort.mortality := Death.rate * (Birth.cohort / 1000)]
  
  # remove Birth.rate and Death.rate
  four.markets[, c("Birth.rate", "Death.rate") := NULL]
  
  # check correlation of indicators and Gov.Budget in four.markets
  corrplot(cor(na.omit(four.markets[,!c("Country", "Market")])), 
           diag = FALSE, tl.offset = 0.5, tl.srt = 5, tl.cex = 1, # tl.pos = "n",
           # p.mat = cors.sig$p, insig = "blank",
           type = "lower", method = "square", order = "FPC", # addgrid.col = "transparent", 
           col = c("red", "blue"), bg = "white", cl.cex = 1.25)
  
  # split up the part of four.markets that is missing all data
  four.markets.missing = data.table(four.markets[is.na(GNI)])
  
  four.markets.missing = rbind(four.markets.missing, 
                               data.table(four.markets[is.na(Population)]))
  
  # remove any duplicates that may have occurred
  four.markets.missing = four.markets.missing[!duplicated(four.markets.missing)]
  
  # remove the missing part from four.markets
  four.markets = data.table(four.markets[!is.na(GNI)])
  four.markets = data.table(four.markets[!is.na(Population)])
  
  # split up four.markets into keep and impute:
  # keep will retain the variables we want but don't want to use for imputation
  # impute will retain the variables we want to use for imputation
  four.markets.keep = data.table(four.markets[, .(Country, Market)])
  four.markets.impute = data.table(four.markets[, !c("Country", "Market")])
  four.markets.impute = data.table(cbind(four.markets.impute, model.matrix(~., data = four.markets.keep[,.(Market)])[,-1]))
  
  # split up four.markets.impute into 2 tables: one for imputing Gov.Budget and the other for computing Birth Rate
  four.markets.impute1 = data.table(four.markets.impute[,.(MarketLMIC, MarketHMIC, MarketHIC, GNI, Account.balance, Gov.Budget)])
  four.markets.impute2 = data.table(four.markets.impute[,.(MarketLMIC, MarketHMIC, MarketHIC, Population, Birth.cohort, Cohort.mortality)])
  
  # use random forests to impute the missing values
  four.markets.impute1 = missRanger(four.markets.impute1, num.trees = 100, seed = 42, num.threads = 1)
  four.markets.impute2 = missRanger(four.markets.impute2, num.trees = 100, seed = 42, num.threads = 1)
  
  # update four.markets
  four.markets = data.table(cbind(four.markets.keep, 
                                  four.markets.impute1[, !c("MarketLMIC", "MarketHMIC", "MarketHIC")],
                                  four.markets.impute2[, !c("MarketLMIC", "MarketHMIC", "MarketHIC")]))
  
  # append four.markets.missing to four.markets
  setcolorder(four.markets.missing, names(four.markets))
  four.markets = rbind(four.markets, four.markets.missing)
  
  # compute Birth.mortality, Gov.VEpb, GNIpc
  four.markets[, Birth.mortality := 100 * (Cohort.mortality / Birth.cohort)]
  four.markets[, Gov.VEpb := Gov.Budget / Birth.cohort]
  four.markets[, GNIpc := GNI / Population]
  
  # get the modeling data from four.markets
  VE = data.table(four.markets[,.(Country, Market, GNIpc, Gov.VEpb, Birth.mortality)])
  
  # update the levels of Market in VE
  VE[Market == "LIC", Market := "UNICEF"]
  VE[Market == "LMIC", Market := "PAHO"]
  VE[, Market := factor(Market, levels = c("UNICEF", "PAHO", "HMIC", "HIC"))]
  
  # ---- anomaly detection ----
  
  # lets remove any countries that are anomolies in their market
  # initialize the h2o instance
  h2o.init()
  
  # remove any objects in the h2o instance
  h2o.removeAll()
  
  # remove the progress bar when model building
  h2o.no_progress()
  
  # identify predictors (x) and response (y)
  y = "Market"
  x = c("GNIpc", "Gov.VEpb", "Birth.mortality")
  
  # make VE into h2o objects
  VE.h2o = as.h2o(na.omit(VE[, c(y, x), with = FALSE]))
  VE.X.h2o = as.h2o(na.omit(VE[, x, with = FALSE]))
  
  # choose the number of folds to train the weights with cross validation
  nfolds = 3
  
  # build the fold assignment
  my.folds = h2o.cross_validation_fold_assignment(h2o.glm(x = x, y = y, 
                                                          training_frame = VE.h2o, 
                                                          nfolds = nfolds,
                                                          keep_cross_validation_fold_assignment = TRUE,
                                                          fold_assignment = "Stratified",
                                                          family = "multinomial",
                                                          seed = 42))
  
  # name the fold assignment
  names(my.folds) = "my.folds"
  
  # set up hyperparameters of interest
  glm.hyper.params = list(lambda = c(1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0),
                          alpha = c(0, 0.5, 1))
  
  # lets use a random grid search and specify a time limit and/or model limit
  glm.search.criteria = list(strategy = "RandomDiscrete", 
                             max_runtime_secs = 5 * 60, 
                             # max_models = 100, 
                             seed = 42)
  
  # lets run a grid search for a good model with intercept = FALSE and standardize = FALSE
  h2o.rm("glm.random.gridA")
  glm.random.gridA = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridA",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(VE.h2o, my.folds),
                              family = "multinomial",
                              fold_column = "my.folds",
                              intercept = FALSE,
                              standardize = FALSE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # lets run a grid search for a good model with intercept = TRUE and standardize = FALSE
  h2o.rm("glm.random.gridB")
  glm.random.gridB = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridB",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(VE.h2o, my.folds),
                              family = "multinomial",
                              fold_column = "my.folds",
                              intercept = TRUE,
                              standardize = FALSE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # lets run a grid search for a good model with intercept = TRUE and standardize = TRUE
  h2o.rm("glm.random.gridC")
  glm.random.gridC = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridC",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(VE.h2o, my.folds),
                              family = "multinomial",
                              fold_column = "my.folds",
                              intercept = TRUE,
                              standardize = TRUE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # lets run a grid search for a good model with intercept = FALSE and standardize = TRUE
  h2o.rm("glm.random.gridD")
  glm.random.gridD = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridD",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(VE.h2o, my.folds),
                              family = "multinomial",
                              fold_column = "my.folds",
                              intercept = FALSE,
                              standardize = TRUE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # rank each model in the random grids
  glm.gridA = h2o.getGrid("glm.random.gridA", sort_by = "accuracy", decreasing = TRUE)
  glm.gridB = h2o.getGrid("glm.random.gridB", sort_by = "accuracy", decreasing = TRUE)
  glm.gridC = h2o.getGrid("glm.random.gridC", sort_by = "accuracy", decreasing = TRUE)
  glm.gridD = h2o.getGrid("glm.random.gridD", sort_by = "accuracy", decreasing = TRUE)
  
  # combine all the grid tables into one grid table that considers the options for intercept and standardize
  glm.grid = rbind(cbind(data.table(glm.gridA@summary_table), intercept = "FALSE", standardize = "FALSE", search = 1),
                   cbind(data.table(glm.gridB@summary_table), intercept = "TRUE", standardize = "FALSE", search = 2),
                   cbind(data.table(glm.gridC@summary_table), intercept = "TRUE", standardize = "TRUE", search = 3),
                   cbind(data.table(glm.gridD@summary_table), intercept = "FALSE", standardize = "TRUE", search = 4))
  
  # combine all the grid models
  glm.grid.models = list(glm.gridA, glm.gridB, glm.gridC, glm.gridD)
  
  # order grid by accuracy
  glm.grid = glm.grid[order(as.numeric(accuracy), decreasing = TRUE)]
  
  # get the grid search that resulted in the best model
  glm.best.grid = glm.grid$search[1]
  
  # get the best model from our grid search
  glm.mod = h2o.getModel(glm.grid.models[[glm.best.grid]]@model_ids[[1]])
  
  # get the summary table of the grid search
  VE.glm.grid = data.table(glm.grid)
  
  # set up the data types for each column in VE.grid for plotting purposes
  VE.glm.grid = VE.glm.grid[, .(alpha = as.factor(alpha),
                                lambda = as.factor(lambda),
                                intercept = as.factor(intercept),
                                standardize = as.factor(standardize),
                                model = removePunctuation(gsub("[A-z]+", "", model_ids)),
                                accuracy = as.numeric(accuracy))]
  
  # plot accuracy v. standardize and lambda to see which structure is most robust
  plot.glm.grid = ggplot(VE.glm.grid, aes(x = lambda, y = accuracy, color = standardize, fill = standardize)) + 
    # geom_boxplot() + 
    geom_jitter(size = 3, alpha = 2/3) + 
    scale_y_continuous(labels = percent) + 
    ggtitle("Cross Validation Error") + 
    labs(x = "Strength of Regularization", y = "Accuracy", color = "Standardize", fill = "Standardize") + 
    theme_bw(base_size = 30) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          # axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1, byrow = TRUE))
  
  plot.glm.grid
  
  # auto encoders explained: 
  # https://www.kaggle.com/devendray/predict-product-backorders-using-h2o-autoencoder
  
  # choose the total number of hidden nodes per layer
  nodes = 8
  
  # choose the number of hidden layers
  layers = 2
  
  # compute the vector of hidden layers
  hidden = rep(nodes, layers)
  
  # pick a fraction of the features for each training row to omit from training in order to improve generalization
  input.dropout = 0
  
  # pick a fraction of the inputs for each hidden layer to omit from training to improve generalization
  hidden.dropout = 0
  
  # initilize the weights and biases
  set.seed(42)
  weights = lapply(1:(length(hidden) + 1), function(j) 
  {
    # compute weights for the input layer to the first hidden layer
    if(j == 1)
    {
      mat = matrix(runif(n = length(x) * hidden[1], min = -1, max = 1), nrow = hidden[1])
      colnames(mat) = x
      
      # compute weights for the last hidden layer to the output layer 
    } else if(j == length(hidden) + 1)
    {
      mat = matrix(runif(n = hidden[j-1] * length(y), min = -1, max = 1), nrow = length(y))
      colnames(mat) = paste0("C", 1:hidden[j-1])
      
      # compute weights between these two hidden layers
    } else
    {
      mat = matrix(runif(n = hidden[j-1] * hidden[j], min = -1, max = 1), nrow = hidden[j])
      colnames(mat) = paste0("C", 1:hidden[j-1])
    }
    
    mat = as.h2o(data.table(mat))
    return(mat)
  })
  
  set.seed(21)
  biases = lapply(1:(length(hidden) + 1), function(j) 
  {
    # compute biases for the last hidden layer to the output layer
    if(j == length(hidden) + 1)
    {
      mat = matrix(runif(n = length(y), min = -1, max = 1), nrow = length(y))
      
      # compute biases between these two layers
    } else
    {
      mat = matrix(runif(n = hidden[j], min = -1, max = 1), nrow = hidden[j])
    }
    
    colnames(mat) = "C1"
    mat = as.h2o(data.table(mat))
    return(mat)
  })
  
  # compute the total number of weights in the system
  num.weights = sum(c(sum(sapply(1:length(weights), function(i) 
  {
    # count the number of weights for the input layer to the first hidden layer
    if(i == 1)
    {
      count = (1 - hidden.dropout) * (nrow(weights[[i]]) * ((1 - input.dropout) * ncol(weights[[i]])))
      
      # count the number of weights for the last hidden layer to the output layer 
    } else if(i == length(weights))
    {
      count = nrow(weights[[i]]) * ncol(weights[[i]])
      
      # count the number of weights between these two hidden layers
    } else
    {
      count = (1 - hidden.dropout) * (nrow(weights[[i]]) * ncol(weights[[i]]))
    }
    
    return(count)
  })), 
  sum(sapply(1:length(biases), function(i) 
    nrow(biases[[i]]) * ncol(biases[[i]])))))
  
  # here's the portion of your training data set (during cross validation) that the neural network could memorize, 
  # if you don't use input and/or hidden dropout ratios
  training.portion = round(num.weights / (nrow(VE.X.h2o) * ((nfolds - 1) / nfolds)), 4)
  noquote(paste0(training.portion * 100, "%"))
  
  # set up hyperparameters of interest
  nnet.hyper.params = list(hidden = list(4, 6, 8,
                                         c(4, 4), c(6, 6), c(8, 8)),
                           epochs = c(250),
                           activation = "Tanh",
                           # activation = "TanhWithDropout",
                           l1 = c(1e-3, 1e-5),
                           l2 = c(0, 1e-3, 1e-5),
                           adaptive_rate = TRUE)
  
  # lets use a random grid search and specify a time limit and/or model limit
  nnet.search.criteria = list(strategy = "RandomDiscrete", 
                              max_runtime_secs = 5 * 60, 
                              # max_models = 100, 
                              seed = 42)
  
  # run a random grid search for a good model
  h2o.rm("nnet.random.grid")
  nnet.random.grid = h2o.grid(algorithm = "deeplearning",
                              grid_id = "nnet.random.grid",
                              autoencoder = TRUE,
                              x = x,
                              training_frame = VE.X.h2o,
                              seed = 21,
                              hyper_params = nnet.hyper.params,
                              search_criteria = nnet.search.criteria)
  
  # rank each model in the random grid
  nnet.grid = h2o.getGrid("nnet.random.grid", sort_by = "rmse", decreasing = FALSE)
  
  # get the best model from our grid search
  nnet.mod = h2o.getModel(nnet.grid@model_ids[[1]])
  
  # get the summary table of the grid search
  DT.nnet.grid = data.table(nnet.grid@summary_table)
  
  # set up the data types for each column in DT.grid for plotting purposes
  DT.nnet.grid = DT.nnet.grid[, .(activation = as.factor(activation),
                                  epochs = as.numeric(epochs),
                                  hidden = as.factor(hidden),
                                  l1 = as.factor(l1),
                                  l2 = as.factor(l2),
                                  model = removePunctuation(gsub("[A-z]+", "", model_ids)),
                                  rmse = as.numeric(rmse))]
  
  # plot rmse v. hidden and activation to see which structure is most robust
  plot.nnet.grid = ggplot(DT.nnet.grid, aes(x = hidden, y = rmse, color = activation, fill = activation)) + 
    # geom_boxplot() + 
    geom_jitter(size = 3, alpha = 2/3) + 
    # scale_y_continuous(labels = dollar) + 
    ggtitle("Cross Validation Error") + 
    labs(x = "Hidden Layer Structure", y = "RMSE", color = "Activation Function", fill = "Activation Function") + 
    theme_bw(base_size = 30) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          # axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1, byrow = TRUE))
  
  plot.nnet.grid
  
  # compute the reconstruction error for each obervation in VE
  VE.an = unlist(as.data.table(h2o.anomaly(nnet.mod, VE.X.h2o)))
  names(VE.an) = VE$Country[-which(is.na(VE$GNIpc))]
  
  # plot the reconstruction eror to find out a cutoff value
  n = length(VE.an)
  head((sort(VE.an, decreasing = TRUE)[-n] / sort(VE.an, decreasing = TRUE)[-1]) - 1, 25)
  cutoff = sort(VE.an, decreasing = TRUE)[4]
  
  plot(sort(VE.an))
  lines(x = rep(cutoff, length(VE.an)), col = "blue")
  
  # determine which observations are outliers, and add it to VE
  VE = rbind(cbind(na.omit(VE), outlier = as.numeric(VE.an >= cutoff)),
             cbind(na.omit(VE, invert = TRUE), outlier = NA))
  
  # make VE into h2o objects
  VE.h2o = as.h2o(VE[outlier == 0, c(y, x), with = FALSE])
  VE.X.h2o = as.h2o(VE[outlier == 0, x, with = FALSE])
  
  # update my.folds to not contain outliers
  my.folds = my.folds[which(VE.an < cutoff),]
  
  # look up the best glm model we built for predicting Market initially
  glm.grid
  
  # rebuild the model without outliers in the data to see if accuracy improves
  h2o.rm("glm.mod2")
  glm.mod2 = h2o.glm(y = y,
                     x = x,
                     training_frame = h2o.cbind(VE.h2o, my.folds),
                     family = "multinomial",
                     fold_column = "my.folds",
                     intercept = TRUE,
                     standardize = FALSE,
                     seed = 21,
                     alpha = 0.5,
                     lambda = 0)
  
  # compare glm.mod and glm.mod2
  glm.mod@model$cross_validation_metrics_summary
  glm.mod2@model$cross_validation_metrics_summary
  
  # compute VE.avg across markets
  VE.avg = data.table(VE[outlier == 0,
                         .(Gov.VEpb = mean(Gov.VEpb),
                           GNIpc = mean(GNIpc),
                           Birth.mortality = mean(Birth.mortality)),
                         by = .(Market)])
  
  # give DT an id column to preserve its original row order
  DT[, id := 1:nrow(DT)]
  
  # join VE.avg onto DT
  setkey(DT, Market)
  setkey(VE.avg, Market)
  DT = VE.avg[DT]
  
  # order DT by id and remove it
  DT = DT[order(id)]
  DT[, id := NULL]
  
  # get a copy of DT
  DT.mod = data.table(na.omit(DT))
  
  # build market variables
  DT.mod[, UNICEF := as.numeric(Market == "UNICEF")]
  DT.mod[, PAHO := as.numeric(Market == "PAHO")]
  DT.mod[, HMIC := as.numeric(Market == "HMIC")]
  DT.mod[, HIC := as.numeric(Market == "HIC")]
  
  # build a transformed response
  DT.mod[, sqrtPrice := sqrt(Price)]
  
  # extract the Vaccine data from DT.mod that will need to be added to VE
  vac.data = data.table(DT.mod[Market == "UNICEF", .(Vaccine, DTaP, DTwP, HepB, Hib, IPV, OPV)])
  
  # loop through VE and add vac.data to each country
  VE.update = rbindlist(lapply(1:nrow(VE), function(i)
  {
    # create a copy of vac.data
    output = data.table(vac.data)
    
    # add VE info to output
    output[, Country := VE$Country[i]]
    output[, Market := VE$Market[i]]
    output[, Gov.VEpb := VE$Gov.VEpb[i]]
    output[, GNIpc := VE$GNIpc[i]]
    output[, Birth.mortality := VE$Birth.mortality[i]]
    output[, outlier := VE$outlier[i]]
    
    return(output)
  }))
  
  # set VE to be VE.update
  VE = VE.update
  rm(VE.update)
  
  # give VE and DT.mod id columns to preserve their row order
  VE[, id := 1:nrow(VE)]
  DT.mod[, id := 1:nrow(DT.mod)]
  
  # join market price onto VE
  setkey(DT.mod, Market, Vaccine)
  setkey(VE, Market, Vaccine)
  VE = DT.mod[,.(Market, Vaccine, Price, Imputed)][VE]
  
  # order VE and DT.mod by id and then remove id
  VE = VE[order(id)]
  VE[, id := NULL]
  
  DT.mod = DT.mod[order(id)]
  DT.mod[, id := NULL]
  
  # ---- learning market prices: GLM ----
  
  # identify predictors (x) and response (y)
  y = "sqrtPrice"
  x = c("Gov.VEpb", "Birth.mortality", "DTaP", "DTwP", "HepB", "Hib", "IPV", "OPV")
  
  # make DT.mod into h2o objects
  DT.mod.h2o = as.h2o(DT.mod[, c(y, x), with = FALSE])
  DT.X.h2o = as.h2o(DT.mod[, x, with = FALSE])
  
  # identify another data set with Market as the response varaible 
  # this is so our cross validation fold assignments will be stratified by Market
  y.folds = "Market"
  
  # make an h2o object for creating folds
  folds.h2o = as.h2o(DT.mod[, c(y.folds, x), with = FALSE])
  
  # choose the number of folds to train the weights with cross validation
  nfolds = 3
  
  # build the fold assignment
  my.folds = h2o.cross_validation_fold_assignment(h2o.glm(x = x, y = y.folds, 
                                                          training_frame = folds.h2o, 
                                                          nfolds = nfolds,
                                                          keep_cross_validation_fold_assignment = TRUE,
                                                          fold_assignment = "Stratified",
                                                          family = "multinomial",
                                                          seed = 42))
  
  # name the fold assignment
  names(my.folds) = "my.folds"
  
  # should we convert our indicators into deep features of an autocoder?
  deep.features = FALSE
  
  if(deep.features)
  {
    # approach.part1: http://www.619.io/blog/2017/07/20/autoencoder-and-deep-features/
    # approach.part2: https://github.com/h2oai/h2o-3/blob/master/h2o-r/tests/testdir_algos/deeplearning/runit_deeplearning_autoencoder_large.R
    # approach.part3: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
    
    # set up hyperparameters of interest
    nnet.hyper.params = list(hidden = list(4, 6, 8,
                                           c(4, 4), c(6, 6), c(8, 8)),
                             epochs = c(250),
                             activation = "Tanh",
                             # activation = "TanhWithDropout",
                             l1 = c(1e-3, 1e-5),
                             l2 = c(0, 1e-3, 1e-5),
                             adaptive_rate = TRUE)
    
    # lets use a random grid search and specify a time limit and/or model limit
    nnet.search.criteria = list(strategy = "RandomDiscrete", 
                                max_runtime_secs = 5 * 60, 
                                # max_models = 100, 
                                seed = 42)
    
    # run a random grid search for a good model
    h2o.rm("nnet.random.grid")
    nnet.random.grid = h2o.grid(algorithm = "deeplearning",
                                grid_id = "nnet.random.grid",
                                x = x,
                                autoencoder = TRUE,
                                training_frame = DT.X.h2o,
                                seed = 21,
                                hyper_params = nnet.hyper.params,
                                search_criteria = nnet.search.criteria)
    
    # rank each model in the random grid
    nnet.grid = h2o.getGrid("nnet.random.grid", sort_by = "rmse", decreasing = FALSE)
    
    # get the best model from our grid search
    autoencoder = h2o.getModel(nnet.grid@model_ids[[1]])
    
    # get the autoencoder parameters
    hidden.ae = autoencoder@parameters$hidden
    epochs.ae = autoencoder@parameters$epochs
    activation.ae = autoencoder@parameters$activation
    l1.ae = autoencoder@parameters$l1
    l2.ae = autoencoder@parameters$l2
    
    # rebuild the autoencoder so we can explicitly refer to it by model_id
    h2o.rm("autoencoder")
    autoencoder = h2o.deeplearning(model_id = "autoencoder",
                                   x = x,
                                   autoencoder = TRUE,
                                   training_frame = DT.X.h2o,
                                   hidden = hidden.ae,
                                   epochs = epochs.ae,
                                   activation = activation.ae,
                                   l1 = l1.ae,
                                   l2 = l2.ae,
                                   seed = 21)
    
    # replace indicators in DT.mod.h2o and DT.X.h2o with the nonlinear features of the last hidden layer in autoencoder
    DT.X.h2o = h2o.deepfeatures(autoencoder, DT.X.h2o, layer = length(hidden.ae))
    DT.mod.h2o = h2o.cbind(DT.mod.h2o[,1], DT.X.h2o)
    x.old = x
    x = names(DT.X.h2o)
  }
  
  # set up hyperparameters of interest
  glm.hyper.params = list(lambda = c(1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0),
                          alpha = c(0, 0.5, 1))
  
  # lets use a random grid search and specify a time limit and/or model limit
  glm.search.criteria = list(strategy = "RandomDiscrete", 
                             max_runtime_secs = 5 * 60, 
                             # max_models = 100, 
                             seed = 42)
  
  # lets run a grid search for a good model with intercept = FALSE and standardize = FALSE
  h2o.rm("glm.random.gridA")
  glm.random.gridA = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridA",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                              fold_column = "my.folds",
                              intercept = FALSE,
                              standardize = FALSE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # lets run a grid search for a good model with intercept = TRUE and standardize = FALSE
  h2o.rm("glm.random.gridB")
  glm.random.gridB = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridB",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                              fold_column = "my.folds",
                              intercept = TRUE,
                              standardize = FALSE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # lets run a grid search for a good model with intercept = TRUE and standardize = TRUE
  h2o.rm("glm.random.gridC")
  glm.random.gridC = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridC",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                              fold_column = "my.folds",
                              intercept = TRUE,
                              standardize = TRUE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # lets run a grid search for a good model with intercept = FALSE and standardize = TRUE
  h2o.rm("glm.random.gridD")
  glm.random.gridD = h2o.grid(algorithm = "glm",
                              grid_id = "glm.random.gridD",
                              y = y,
                              x = x,
                              training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                              fold_column = "my.folds",
                              intercept = FALSE,
                              standardize = TRUE,
                              seed = 21,
                              hyper_params = glm.hyper.params,
                              search_criteria = glm.search.criteria)
  
  # rank each model in the random grids
  glm.gridA = h2o.getGrid("glm.random.gridA", sort_by = "rmse", decreasing = FALSE)
  glm.gridB = h2o.getGrid("glm.random.gridB", sort_by = "rmse", decreasing = FALSE)
  glm.gridC = h2o.getGrid("glm.random.gridC", sort_by = "rmse", decreasing = FALSE)
  glm.gridD = h2o.getGrid("glm.random.gridD", sort_by = "rmse", decreasing = FALSE)
  
  # combine all the grid tables into one grid table that considers the options for intercept and standardize
  glm.grid = rbind(cbind(data.table(glm.gridA@summary_table), intercept = "FALSE", standardize = "FALSE", search = 1),
                   cbind(data.table(glm.gridB@summary_table), intercept = "TRUE", standardize = "FALSE", search = 2),
                   cbind(data.table(glm.gridC@summary_table), intercept = "TRUE", standardize = "TRUE", search = 3),
                   cbind(data.table(glm.gridD@summary_table), intercept = "FALSE", standardize = "TRUE", search = 4))
  
  # combine all the grid models
  glm.grid.models = list(glm.gridA, glm.gridB, glm.gridC, glm.gridD)
  
  # order grid by rmse
  glm.grid = glm.grid[order(as.numeric(rmse))]
  
  # get the grid search that resulted in the best model
  glm.best.grid = glm.grid$search[1]
  
  # get the best model from our grid search
  glm.mod = h2o.getModel(glm.grid.models[[glm.best.grid]]@model_ids[[1]])
  
  # get the summary table of the grid search
  DT.glm.grid = data.table(glm.grid)
  
  # set up the data types for each column in DT.grid for plotting purposes
  DT.glm.grid = DT.glm.grid[, .(alpha = as.factor(alpha),
                                lambda = as.factor(lambda),
                                intercept = as.factor(intercept),
                                standardize = as.factor(standardize),
                                model = removePunctuation(gsub("[A-z]+", "", model_ids)),
                                rmse = as.numeric(rmse))]
  
  # plot rmse v. standardize and lambda to see which structure is most robust
  plot.glm.grid = ggplot(DT.glm.grid, aes(x = lambda, y = rmse, color = standardize, fill = standardize)) + 
    # geom_boxplot() + 
    geom_jitter(size = 3, alpha = 2/3) + 
    # scale_y_continuous(labels = dollar) + 
    ggtitle("Cross Validation Error") + 
    labs(x = "Strength of Regularization", y = "RMSE", color = "Standardize", fill = "Standardize") + 
    theme_bw(base_size = 30) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          # axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1, byrow = TRUE))
  
  plot.glm.grid
  
  # check out summary statistics
  summary(glm.mod)
  
  # make predictions with the training model
  ynew = as.data.frame(predict(glm.mod, newdata = h2o.cbind(DT.X.h2o, my.folds)))$predict
  ynew = ifelse(ynew < 0, 0, ynew)
  ynew = ynew^2
  
  # get the true values
  ytrue = as.vector(DT.mod.h2o[,1])
  ytrue = ytrue^2
  
  # compute prediction metrics
  glm.metrics = h2o.make_metrics(predicted = as.h2o(ynew), actuals = as.h2o(ytrue))
  glm.metrics
  
  # create a copy of DT.mod
  DT.glm = data.table(cbind(DT.mod[,.(Market, Vaccine, Price)], 
                            "Residuals" = DT.mod$Price - ynew,
                            "Fitted" = ynew))
  
  # make Market and Vaccine into factor variables
  DT.glm[, Market := factor(Market, levels = unique(Market))]
  DT.glm[, Vaccine := factor(Vaccine, levels = unique(Vaccine))]
  
  # plot 6 residual plots of glm.mod
  glm.plots = residplots(actual = DT.glm$Price, fit = DT.glm$Fitted, histlabel.y = -1/2, binwidth = 2)
  do.call(grid.arrange,  c(glm.plots, nrow = 2))
  
  # see if the residuals have constant variance
  # color coat by Market to see if any Market is facing worse error than others
  glm.p1 = ggplot(DT.glm, aes(x = Fitted, y = Residuals / Fitted, color = Market)) + 
    geom_point(size = 4) +
    geom_smooth(size = 1, color = "forestgreen", method = "loess", se = FALSE) + 
    geom_hline(yintercept = 0, size = 1, lty = "dashed", color = "black") +
    scale_color_manual(values = colorRampPalette(c("cornflowerblue", "orangered"))(length(unique(DT.glm$Market)))) +
    scale_y_continuous(labels = percent) + 
    scale_x_continuous(labels = dollar) + 
    ggtitle("Residuals v. Fitted") + 
    labs(color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 4))
  
  # color coat by Vaccine to see if any Vaccine is facing worse error than others
  glm.p2 = ggplot(DT.glm, aes(x = Fitted, y = Residuals / Fitted, color = Vaccine)) + 
    geom_point(size = 4) +
    geom_smooth(size = 1, color = "red", method = "loess", se = FALSE) + 
    geom_hline(yintercept = 0, size = 1, lty = "dashed", color = "black") +
    scale_color_manual(values = colorRampPalette(c("purple", "gold", "forestgreen"))(length(unique(DT.glm$Vaccine)))) +
    scale_y_continuous(labels = percent) + 
    scale_x_continuous(labels = dollar) +
    ggtitle("Residuals v. Fitted") + 
    labs(color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 4))
  
  do.call(grid.arrange,  c(list(glm.p1, glm.p2), nrow = 1))
  
  # get a subset of VE with various countries
  countries = c("NICARAGUA", "ELSALVADOR", "DOMINICANREPUBLIC", "BRAZIL", "URUGUAY", "UNITEDSTATESOFAMERICA")
  VE.mod = data.table(VE[Country %in% countries])
  
  # set VE.mod to be an h2o object
  # if we rely on deep features then lets use autoencoder on VE.mod
  if(deep.features)
  {
    # make an h2o object for creating folds
    folds.VE.h2o = as.h2o(VE.mod[, c(y.folds, x.old), with = FALSE])
    
    # build the fold assignment
    VE.folds = h2o.cross_validation_fold_assignment(h2o.glm(x = x.old, y = y.folds, 
                                                            training_frame = folds.VE.h2o, 
                                                            nfolds = nfolds,
                                                            keep_cross_validation_fold_assignment = TRUE,
                                                            fold_assignment = "Stratified",
                                                            family = "multinomial",
                                                            seed = 42))
    
    # name the fold assignment
    names(VE.folds) = "my.folds"
    
    # get original indicators
    VE.mod.h2o = as.h2o(VE.mod[, x.old, with = FALSE])
    
    # convert indicators into the deep features of autoencoder
    VE.mod.h2o = h2o.deepfeatures(autoencoder, VE.mod.h2o, layer = length(hidden.ae))
    
  } else
  {
    # make an h2o object for creating folds
    folds.VE.h2o = as.h2o(VE.mod[, c(y.folds, x), with = FALSE])
    
    # build the fold assignment
    VE.folds = h2o.cross_validation_fold_assignment(h2o.glm(x = x, y = y.folds, 
                                                            training_frame = folds.VE.h2o, 
                                                            nfolds = nfolds,
                                                            keep_cross_validation_fold_assignment = TRUE,
                                                            fold_assignment = "Stratified",
                                                            family = "multinomial",
                                                            seed = 42))
    
    # name the fold assignment
    names(VE.folds) = "my.folds"
    
    # get indicators
    VE.mod.h2o = as.h2o(VE.mod[, x, with = FALSE])
  }
  
  # make predictions with the glm model
  ynew = as.data.frame(predict(glm.mod, newdata = h2o.cbind(VE.mod.h2o, VE.folds)))$predict
  ynew = ifelse(ynew < 0, 0, ynew)
  ynew = ynew^2
  
  # add ynew to VE.mod
  VE.mod[, Fitted.GLM := ynew]
  
  # get a subset of paho with various countries
  countries2 = c("Nicaragua", "El Salvador", "Dominican Republic (the)", "Brazil", "Uruguay")
  paho.mod = data.table(paho[Country %in% countries2, 
                             .(Market, Vaccine, Price, Imputed = NA,
                               DTaP = NA, DTwP = NA, HepB = NA, Hib = NA, IPV = NA, OPV = NA,
                               Country, Gov.VEpb = NA, GNIpc = NA, Birth.mortality = NA, 
                               outlier = NA, Fitted.GLM = Price)])
  
  # build a usa.mod table to add CDC pricing to VE.mod
  usa.mod = data.table(usa[!(Vaccine %in% c("MMR", "V", "MMR-V")),
                           .(Market, Vaccine, Price, Imputed = NA, 
                             DTaP, DTwP, HepB, Hib, IPV, OPV, Country = "CDC",
                             Gov.VEpb = NA, GNIpc = NA, Birth.mortality = NA, outlier = NA,
                             Fitted.GLM = Price)])
  
  # add usa.mod to VE.mod
  VE.mod = data.table(rbind(VE.mod, usa.mod, paho.mod))
  
  # update countries
  countries = c("NICARAGUA", "Nicaragua", 
                "ELSALVADOR", "El Salvador", 
                "DOMINICANREPUBLIC", "Dominican Republic (the)", 
                "BRAZIL", "Brazil", 
                "URUGUAY", "Uruguay", 
                "UNITEDSTATESOFAMERICA", "CDC")
  
  # set up a color for each country
  color.set = c("cornflowerblue", "blue", 
                "lightcoral", "red",
                "plum3", "purple",
                "burlywood4", "saddlebrown",
                "seagreen3", "forestgreen",
                "gray50", "black")
  
  # update countries to be a table that pairs up duplicate countries
  countries = data.table(Country = countries, 
                         Country2 = c("NICARAGUA", "NICARAGUA", 
                                      "ELSALVADOR", "ELSALVADOR", 
                                      "DOMINICANREPUBLIC", "DOMINICANREPUBLIC", 
                                      "BRAZIL", "BRAZIL", 
                                      "URUGUAY", "URUGUAY", 
                                      "UNITEDSTATESOFAMERICA", "UNITEDSTATESOFAMERICA"))
  
  # give VE.mod and countries an id column to preserve row order
  VE.mod[, id := 1:nrow(VE.mod)]
  countries[, id := 1:nrow(countries)]
  
  # join countries onto VE.mod
  setkey(VE.mod, Country)
  setkey(countries, Country)
  VE.mod = countries[VE.mod]
  
  # order VE.mod and countries by id then remove id
  VE.mod = VE.mod[order(id)]
  VE.mod[, id := NULL]
  countries = countries[order(id)]
  countries[, id := NULL]
  
  # make Country, Country2, and Vaccine into factors
  VE.mod[, Country := factor(Country, levels = countries$Country)]
  VE.mod[, Country2 := factor(Country2, levels = unique(countries$Country2))]
  VE.mod[, Vaccine := factor(Vaccine, levels = unique(VE.mod$Vaccine))]
  
  # plot the predicted price points of various countries
  glm.pred = ggplot(VE.mod, aes(x = Vaccine, 
                                y = Fitted.GLM, 
                                color = Country, 
                                group = Country)) + 
    geom_point(size = 3, na.rm = TRUE) +
    geom_line(size = 1, na.rm = TRUE) +
    facet_wrap(~Country2) + 
    scale_color_manual(values = color.set) +
    scale_y_continuous(labels = dollar) + 
    ggtitle("Country Price Predictions") + 
    labs(x = "Vaccine", y = "Price", color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "none", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 2, byrow = FALSE))
  
  glm.pred
  
  # ---- learning market prices: NNET ----
  
  # identify predictors (x) and response (y)
  y = "Price"
  x = c("Gov.VEpb", "DTaP", "DTwP", "HepB", "Hib", "IPV", "OPV")
  
  # make DT.mod into h2o objects
  DT.mod.h2o = as.h2o(DT.mod[, c(y, x), with = FALSE])
  DT.X.h2o = as.h2o(DT.mod[, x, with = FALSE])
  
  # choose the total number of hidden nodes per layer
  nodes = 10
  
  # choose the number of hidden layers
  layers = 1
  
  # compute the vector of hidden layers
  hidden = rep(nodes, layers)
  
  # pick a fraction of the features for each training row to omit from training in order to improve generalization
  input.dropout = 0
  
  # pick a fraction of the inputs for each hidden layer to omit from training to improve generalization
  hidden.dropout = 0
  
  # initilize the weights and biases
  set.seed(42)
  weights = lapply(1:(length(hidden) + 1), function(j) 
  {
    # compute weights for the input layer to the first hidden layer
    if(j == 1)
    {
      mat = matrix(runif(n = length(x) * hidden[1], min = -1, max = 1), nrow = hidden[1])
      colnames(mat) = x
      
      # compute weights for the last hidden layer to the output layer 
    } else if(j == length(hidden) + 1)
    {
      mat = matrix(runif(n = hidden[j-1] * length(y), min = -1, max = 1), nrow = length(y))
      colnames(mat) = paste0("C", 1:hidden[j-1])
      
      # compute weights between these two hidden layers
    } else
    {
      mat = matrix(runif(n = hidden[j-1] * hidden[j], min = -1, max = 1), nrow = hidden[j])
      colnames(mat) = paste0("C", 1:hidden[j-1])
    }
    
    mat = as.h2o(data.table(mat))
    return(mat)
  })
  
  set.seed(21)
  biases = lapply(1:(length(hidden) + 1), function(j) 
  {
    # compute biases for the last hidden layer to the output layer
    if(j == length(hidden) + 1)
    {
      mat = matrix(runif(n = length(y), min = -1, max = 1), nrow = length(y))
      
      # compute biases between these two layers
    } else
    {
      mat = matrix(runif(n = hidden[j], min = -1, max = 1), nrow = hidden[j])
    }
    
    colnames(mat) = "C1"
    mat = as.h2o(data.table(mat))
    return(mat)
  })
  
  # compute the total number of weights in the system
  num.weights = sum(c(sum(sapply(1:length(weights), function(i) 
  {
    # count the number of weights for the input layer to the first hidden layer
    if(i == 1)
    {
      count = (1 - hidden.dropout) * (nrow(weights[[i]]) * ((1 - input.dropout) * ncol(weights[[i]])))
      
      # count the number of weights for the last hidden layer to the output layer 
    } else if(i == length(weights))
    {
      count = nrow(weights[[i]]) * ncol(weights[[i]])
      
      # count the number of weights between these two hidden layers
    } else
    {
      count = (1 - hidden.dropout) * (nrow(weights[[i]]) * ncol(weights[[i]]))
    }
    
    return(count)
  })),
  sum(sapply(1:length(biases), function(i) 
    nrow(biases[[i]]) * ncol(biases[[i]])))))
  
  # here's the portion of your training data set (during cross validation) that the neural network could memorize, 
  # if you don't use input and/or hidden dropout ratios
  training.portion = num.weights / (nrow(DT.mod) * ((nfolds - 1) / nfolds))
  noquote(paste0(training.portion * 100, "%"))
  
  # should we use an autoencoder or a grid search to build our neural network?
  use.autoencoder = FALSE
  
  if(use.autoencoder)
  {
    # set up hyperparameters of interest
    nnet.hyper.params = list(hidden = list(4, 5, 6,
                                           c(4, 4), c(5, 5), c(6, 6)),
                             epochs = c(250),
                             activation = "Tanh",
                             # activation = "TanhWithDropout",
                             l1 = c(1e-3, 1e-5),
                             l2 = c(0, 1e-3, 1e-5),
                             adaptive_rate = TRUE)
    
    # lets use a random grid search and specify a time limit and/or model limit
    nnet.search.criteria = list(strategy = "RandomDiscrete", 
                                max_runtime_secs = 5 * 60, 
                                # max_models = 100, 
                                seed = 42)
    
    # run a random grid search for a good model
    h2o.rm("nnet.random.grid")
    nnet.random.grid = h2o.grid(algorithm = "deeplearning",
                                grid_id = "nnet.random.grid",
                                x = x,
                                autoencoder = TRUE,
                                training_frame = DT.X.h2o,
                                seed = 21,
                                hyper_params = nnet.hyper.params,
                                search_criteria = nnet.search.criteria)
    
    # rank each model in the random grid
    nnet.grid = h2o.getGrid("nnet.random.grid", sort_by = "rmse", decreasing = FALSE)
    
    # get the summary table of the grid search
    DT.nnet.grid = data.table(nnet.grid@summary_table)
    
    # set up the data types for each column in DT.grid for plotting purposes
    DT.nnet.grid = DT.nnet.grid[, .(autoencoder = TRUE,
                                    activation = as.factor(activation),
                                    epochs = as.numeric(epochs),
                                    hidden = as.factor(hidden),
                                    l1 = as.factor(l1),
                                    l2 = as.factor(l2),
                                    model = removePunctuation(gsub("[A-z]+", "", model_ids)),
                                    rmse = as.numeric(rmse))]
    
    # get the best model from our grid search
    autoencoder = h2o.getModel(nnet.grid@model_ids[[1]])
    
    # get the autoencoder parameters
    hidden.ae = autoencoder@parameters$hidden
    epochs.ae = autoencoder@parameters$epochs
    activation.ae = autoencoder@parameters$activation
    l1.ae = autoencoder@parameters$l1
    l2.ae = autoencoder@parameters$l2
    
    # rebuild the autoencoder so we can explicitly refer to it by model_id
    h2o.rm("autoencoder")
    autoencoder = h2o.deeplearning(model_id = "autoencoder",
                                   x = x,
                                   autoencoder = TRUE,
                                   training_frame = DT.X.h2o,
                                   hidden = hidden.ae,
                                   epochs = epochs.ae,
                                   activation = activation.ae,
                                   l1 = l1.ae,
                                   l2 = l2.ae,
                                   seed = 21)
    
    # build our nueral network with autoencoder
    h2o.rm("nnet.mod")
    nnet.mod = h2o.deeplearning(x = x,
                                y = y,
                                training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                                fold_column = "my.folds",
                                export_weights_and_biases = TRUE,
                                variable_importances = TRUE,
                                seed = 21,
                                hidden = hidden.ae,
                                epochs = epochs.ae,
                                activation = activation.ae,
                                l1 = l1.ae,
                                l2 = l2.ae,
                                seed = 21,
                                pretrained_autoencoder = "autoencoder")
    
  } else
  {
    # set up hyperparameters of interest
    nnet.hyper.params = list(hidden = list(4, 6, 8, 10,
                                           c(4, 4), c(6, 6), c(8, 8), c(10, 10)),
                             epochs = c(250),
                             activation = c("Rectifier", "Tanh"),
                             # activation = c("RectifierWithDropout", "TanhWithDropout"),
                             # input_dropout_ratio = c(0, 0.1, 0.2),
                             l1 = c(1e-3, 1e-5),
                             l2 = c(0, 1e-3, 1e-5),
                             adaptive_rate = TRUE)
    
    # lets use a random grid search and specify a time limit and/or model limit
    nnet.search.criteria = list(strategy = "RandomDiscrete", 
                                max_runtime_secs = 5 * 60, 
                                # max_models = 100, 
                                seed = 42)
    
    # run a random grid search for a good model
    h2o.rm("nnet.random.grid")
    nnet.random.grid = h2o.grid(algorithm = "deeplearning",
                                grid_id = "nnet.random.grid",
                                x = x,
                                y = y,
                                training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                                fold_column = "my.folds",
                                export_weights_and_biases = TRUE,
                                variable_importances = TRUE,
                                seed = 21,
                                hyper_params = nnet.hyper.params,
                                search_criteria = nnet.search.criteria)
    
    # rank each model in the random grid
    nnet.grid = h2o.getGrid("nnet.random.grid", sort_by = "rmse", decreasing = FALSE)
    
    # get the best model from our grid search
    nnet.mod = h2o.getModel(nnet.grid@model_ids[[1]])
    
    # get the summary table of the grid search
    DT.nnet.grid = data.table(nnet.grid@summary_table)
    
    # set up the data types for each column in DT.grid for plotting purposes
    DT.nnet.grid = DT.nnet.grid[, .(autoencoder = FALSE,
                                    activation = as.factor(activation),
                                    epochs = as.numeric(epochs),
                                    hidden = as.factor(hidden),
                                    l1 = as.factor(l1),
                                    l2 = as.factor(l2),
                                    model = removePunctuation(gsub("[A-z]+", "", model_ids)),
                                    rmse = as.numeric(rmse))]
    
    # plot rmse v. hidden and activation to see which structure is most robust
    plot.nnet.grid = ggplot(DT.nnet.grid, aes(x = hidden, y = rmse, color = activation, fill = activation)) + 
      # geom_boxplot() + 
      geom_jitter(size = 3, alpha = 2/3) + 
      scale_y_continuous(labels = dollar) + 
      ggtitle("Cross Validation Error") + 
      labs(x = "Hidden Layer Structure", y = "RMSE", color = "Activation Function", fill = "Activation Function") + 
      theme_bw(base_size = 30) +
      theme(legend.position = "top", 
            legend.key.size = unit(.25, "in"),
            plot.title = element_text(hjust = 0.5),
            # axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank()) +
      guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1, byrow = TRUE))
    
    plot.nnet.grid
    
    # get nnet.mod parameters
    hidden.nn = nnet.mod@parameters$hidden
    epochs.nn = nnet.mod@parameters$epochs
    activation.nn = nnet.mod@parameters$activation
    l1.nn = nnet.mod@parameters$l1
    l2.nn = nnet.mod@parameters$l2
    
    # get the model weights and biases
    nnet.weights = do.call("c", lapply(1:(length(nnet.mod@parameters$hidden) + 1), function(i) h2o.weights(nnet.mod, i)))
    nnet.biases = do.call("c", lapply(1:(length(nnet.mod@parameters$hidden) + 1), function(i) h2o.biases(nnet.mod, i)))
    
    # summarize model performance
    summary(nnet.mod)
    
    # get the CV models from nnet.mod
    cv.models = lapply(nnet.mod@model$cross_validation_models, function(i) h2o.getModel(i$name))
    
    # set up graphic parameters for the upcoming plot
    par(mfrow = c(2, ceiling(length(cv.models) / 2)),
        oma = c(0, 0, 2, 0))
    
    # plot the rmse over epochs to judge fittness
    lapply(1:length(cv.models), function(i)
      plot(cv.models[[i]], 
           timestep = "epochs", 
           metric = "rmse"))
    
    # add a main title
    mtext("3-Fold Cross-Validation", outer = TRUE, cex = 1.5)
  }
  
  # should we rebuild the nnet.mod?
  # results vary and sometimes the residuals aren't centered at zero
  rebuild.nnet = TRUE
  
  # consider different options for nnet.mod
  head(DT.nnet.grid)
  head(DT.nnet.grid[!(hidden %in% c("[10, 10]"))])
  
  # pick a model from our grid search
  model = 24
  
  if(rebuild.nnet)
  {
    if(use.autoencoder)
    {
      # get the best model from our grid search
      autoencoder = h2o.getModel(nnet.grid@model_ids[[which(DT.nnet.grid$model == model)]])
      
      # get the autoencoder parameters
      hidden.ae = autoencoder@parameters$hidden
      epochs.ae = autoencoder@parameters$epochs
      activation.ae = autoencoder@parameters$activation
      l1.ae = autoencoder@parameters$l1
      l2.ae = autoencoder@parameters$l2
      
      # rebuild the autoencoder so we can explicitly refer to it by model_id
      h2o.rm("autoencoder")
      autoencoder = h2o.deeplearning(model_id = "autoencoder",
                                     x = x,
                                     autoencoder = TRUE,
                                     training_frame = DT.X.h2o,
                                     hidden = hidden.ae,
                                     epochs = epochs.ae,
                                     activation = activation.ae,
                                     l1 = l1.ae,
                                     l2 = l2.ae,
                                     seed = 21)
      
      # build our nueral network with autoencoder
      h2o.rm("nnet.mod")
      nnet.mod = h2o.deeplearning(x = x,
                                  y = y,
                                  training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                                  fold_column = "my.folds",
                                  export_weights_and_biases = TRUE,
                                  variable_importances = TRUE,
                                  seed = 21,
                                  hidden = hidden.ae,
                                  epochs = epochs.ae,
                                  activation = activation.ae,
                                  l1 = l1.ae,
                                  l2 = l2.ae,
                                  seed = 21,
                                  pretrained_autoencoder = "autoencoder")
      
      
    } else
    {
      # get the best model from our grid search
      nnet.mod = h2o.getModel(nnet.grid@model_ids[[which(DT.nnet.grid$model == model)]])
      
      # get nnet.mod parameters
      hidden.nn = nnet.mod@parameters$hidden
      epochs.nn = nnet.mod@parameters$epochs
      activation.nn = nnet.mod@parameters$activation
      l1.nn = nnet.mod@parameters$l1
      l2.nn = nnet.mod@parameters$l2
      
      # rebuild nnet.mod if you would like (sometimes the residuals aren't centered at zero)
      h2o.rm("nnet.mod")
      nnet.mod = h2o.deeplearning(x = x,
                                  y = y,
                                  training_frame = h2o.cbind(DT.mod.h2o, my.folds),
                                  fold_column = "my.folds",
                                  export_weights_and_biases = TRUE,
                                  variable_importances = TRUE,
                                  seed = 21,
                                  hidden = hidden.nn,
                                  epochs = epochs.nn,
                                  activation = activation.nn,
                                  l1 = l1.nn,
                                  l2 = l2.nn)
    }
  }
  
  # make predictions with the training model
  ynew = as.data.frame(predict(nnet.mod, newdata = h2o.cbind(DT.X.h2o, my.folds)))$predict
  ynew = ifelse(ynew < 0, 0, ynew)
  
  # get the true values
  ytrue = as.vector(DT.mod.h2o[,1])
  
  # compute prediction metrics
  nnet.metrics = h2o.make_metrics(predicted = as.h2o(ynew), actuals = as.h2o(ytrue))
  nnet.metrics
  
  # create a copy of DT.mod
  DT.nnet = data.table(cbind(DT.mod[,.(Market, Vaccine, Price)], 
                             "Residuals" = DT.mod$Price - ynew,
                             "Fitted" = ynew))
  
  # make Market and Vaccine into factor variables
  DT.nnet[, Market := factor(Market, levels = unique(Market))]
  DT.nnet[, Vaccine := factor(Vaccine, levels = unique(Vaccine))]
  
  # plot 6 residual plots of nnet.mod
  nnet.plots = residplots(actual = DT.nnet$Price, fit = DT.nnet$Fitted, histlabel.y = -1/2, binwidth = 2)
  do.call(grid.arrange,  c(nnet.plots, nrow = 2))
  
  # see if the residuals have constant variance
  # color coat by Market to see if any Market is facing worse error than others
  nnet.p1 = ggplot(DT.nnet, aes(x = Fitted, y = Residuals / Fitted, color = Market)) + 
    geom_point(size = 4) +
    geom_smooth(size = 1, color = "forestgreen", method = "loess", se = FALSE) + 
    geom_hline(yintercept = 0, size = 1, lty = "dashed", color = "black") +
    scale_color_manual(values = colorRampPalette(c("cornflowerblue", "orangered"))(length(unique(DT.nnet$Market)))) +
    scale_y_continuous(labels = percent) + 
    scale_x_continuous(labels = dollar) + 
    ggtitle("Residuals v. Fitted") + 
    labs(color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 4))
  
  # color coat by Vaccine to see if any Vaccine is facing worse error than others
  nnet.p2 = ggplot(DT.nnet, aes(x = Fitted, y = Residuals / Fitted, color = Vaccine)) + 
    geom_point(size = 4) +
    geom_smooth(size = 1, color = "red", method = "loess", se = FALSE) + 
    geom_hline(yintercept = 0, size = 1, lty = "dashed", color = "black") +
    scale_color_manual(values = colorRampPalette(c("purple", "gold", "forestgreen"))(length(unique(DT.nnet$Vaccine)))) +
    scale_y_continuous(labels = percent) + 
    scale_x_continuous(labels = dollar) +
    ggtitle("Residuals v. Fitted") + 
    labs(color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "top", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 4))
  
  do.call(grid.arrange,  c(list(nnet.p1, nnet.p2), nrow = 1))
  
  # get a subset of VE with various countries
  countries = c("NICARAGUA", "ELSALVADOR", "DOMINICANREPUBLIC", "BRAZIL", "URUGUAY", "UNITEDSTATESOFAMERICA")
  VE.mod = data.table(VE[Country %in% countries])
  
  # make predictions with the nnet model
  ynew = as.data.frame(predict(nnet.mod, newdata = h2o.cbind(VE.mod.h2o, VE.folds)))$predict
  ynew = ifelse(ynew < 0, 0, ynew)
  
  # add ynew to VE
  VE.mod[, Fitted.NNET := ynew]
  
  # get a subset of paho with various countries
  countries2 = c("Nicaragua", "El Salvador", "Dominican Republic (the)", "Brazil", "Uruguay")
  paho.mod = data.table(paho[Country %in% countries2, 
                             .(Market, Vaccine, Price, Imputed = NA,
                               DTaP = NA, DTwP = NA, HepB = NA, Hib = NA, IPV = NA, OPV = NA,
                               Country, Gov.VEpb = NA, GNIpc = NA, Birth.mortality = NA, 
                               outlier = NA, Fitted.NNET = Price)])
  
  # build a usa.mod table to add CDC pricing to VE.mod
  usa.mod = data.table(usa[,.(Market, Vaccine, Price, Imputed = NA, 
                              DTaP, DTwP, HepB, Hib, IPV, OPV, Country = "CDC",
                              Gov.VEpb = NA, GNIpc = NA, Birth.mortality = NA, outlier = NA,
                              Fitted.NNET = Price)])
  
  # add usa.mod to VE.mod
  VE.mod = data.table(rbind(VE.mod, usa.mod, paho.mod))
  
  # update countries
  countries = c("NICARAGUA", "Nicaragua", 
                "ELSALVADOR", "El Salvador", 
                "DOMINICANREPUBLIC", "Dominican Republic (the)", 
                "BRAZIL", "Brazil", 
                "URUGUAY", "Uruguay", 
                "UNITEDSTATESOFAMERICA", "CDC")
  
  # set up a color for each country
  color.set = c("cornflowerblue", "blue", 
                "lightcoral", "red",
                "plum3", "purple",
                "burlywood4", "saddlebrown",
                "seagreen3", "forestgreen",
                "gray50", "black")
  
  # update countries to be a table that pairs up duplicate countries
  countries = data.table(Country = countries, 
                         Country2 = c("NICARAGUA", "NICARAGUA", 
                                      "ELSALVADOR", "ELSALVADOR", 
                                      "DOMINICANREPUBLIC", "DOMINICANREPUBLIC", 
                                      "BRAZIL", "BRAZIL", 
                                      "URUGUAY", "URUGUAY", 
                                      "UNITEDSTATESOFAMERICA", "UNITEDSTATESOFAMERICA"))
  
  # give VE.mod and countries an id column to preserve row order
  VE.mod[, id := 1:nrow(VE.mod)]
  countries[, id := 1:nrow(countries)]
  
  # join countries onto VE.mod
  setkey(VE.mod, Country)
  setkey(countries, Country)
  VE.mod = countries[VE.mod]
  
  # order VE.mod and countries by id then remove id
  VE.mod = VE.mod[order(id)]
  VE.mod[, id := NULL]
  countries = countries[order(id)]
  countries[, id := NULL]
  
  # make Country, Country2, and Vaccine into factors
  VE.mod[, Country := factor(Country, levels = countries$Country)]
  VE.mod[, Country2 := factor(Country2, levels = unique(countries$Country2))]
  VE.mod[, Vaccine := factor(Vaccine, levels = unique(VE.mod$Vaccine))]
  
  # plot the predicted price points of various countries
  nnet.pred = ggplot(VE.mod, aes(x = Vaccine, 
                                 y = Fitted.NNET, 
                                 color = Country, 
                                 group = Country)) + 
    geom_point(size = 3, na.rm = TRUE) +
    geom_line(size = 1, na.rm = TRUE) +
    facet_wrap(~Country2) + 
    scale_color_manual(values = color.set) +
    scale_y_continuous(labels = dollar) + 
    ggtitle("Country Price Predictions") + 
    labs(x = "Vaccine", y = "Price", color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "none", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 2, byrow = FALSE))
  
  nnet.pred
  
  # ---- final country reservation prices ----
  
  # identify predictors (x) for glm and nnet
  glm.x = c("Gov.VEpb", "Birth.mortality", "DTaP", "DTwP", "HepB", "Hib", "IPV", "OPV")
  nnet.x = c("Gov.VEpb", "DTaP", "DTwP", "HepB", "Hib", "IPV", "OPV")
  
  # set VE to be an h2o object
  VE.h2o = as.h2o(VE[, unique(c(glm.x, nnet.x)), with = FALSE])
  
  # make an h2o object for creating folds
  folds.VE.h2o = as.h2o(VE[, c(y.folds, x), with = FALSE])
  
  # build the fold assignment
  VE.folds = h2o.cross_validation_fold_assignment(h2o.glm(x = x, y = y.folds, 
                                                          training_frame = folds.VE.h2o, 
                                                          nfolds = nfolds,
                                                          keep_cross_validation_fold_assignment = TRUE,
                                                          fold_assignment = "Stratified",
                                                          family = "multinomial",
                                                          seed = 42))
  
  # name the fold assignment
  names(VE.folds) = "my.folds"
  
  # make predictions with the glm model
  ynew = as.data.frame(predict(glm.mod, newdata = h2o.cbind(VE.h2o[,which(names(VE.h2o) %in% glm.x)], VE.folds)))$predict
  ynew = ifelse(ynew < 0, 0, ynew)
  ynew = ynew^2
  
  # add ynew to VE
  VE[, Fitted.GLM := ynew]
  
  # make predictions with the nnet model
  ynew = as.data.frame(predict(nnet.mod, newdata = h2o.cbind(VE.h2o[,which(names(VE.h2o) %in% nnet.x)], VE.folds)))$predict
  ynew = ifelse(ynew < 0, 0, ynew)
  
  # add ynew to VE
  VE[, Fitted.NNET := ynew]
  
  # build a paho.mod table to add PAHO pricing to VE
  paho.mod = data.table(paho[,.(Market = paste0(Market, "-eigen"), Vaccine, Price, Imputed = NA,
                                DTaP = NA, DTwP = NA, HepB = NA, Hib = NA, IPV = NA, OPV = NA,
                                Country, Gov.VEpb = NA, GNIpc = NA, Birth.mortality = NA, 
                                outlier = NA, Fitted.GLM = Price, Fitted.NNET = Price)])
  
  # build a usa.mod table to add CDC pricing to VE
  usa.mod = data.table(usa[,.(Market, Vaccine, Price, Imputed = NA, 
                              DTaP, DTwP, HepB, Hib, IPV, OPV, Country = "CDC",
                              Gov.VEpb = NA, GNIpc = NA, Birth.mortality = NA, outlier = NA,
                              Fitted.GLM = Price, Fitted.NNET = Price)])
  
  # add paho.mod and usa.mod to VE
  VE = data.table(rbind(VE, paho.mod, usa.mod))
  
  # compute Fitted.AVG
  VE[, Fitted.AVG := (Fitted.NNET + Fitted.GLM) / 2]
  
  # compute the average country price by market to see if country fitted values avergae out hte the known market prices
  VE.check = data.table(VE[outlier == 0, 
                           .(Fitted.NNET = mean(Fitted.NNET, na.rm = TRUE),
                             Fitted.GLM = mean(Fitted.GLM, na.rm = TRUE),
                             Fitted.AVG = mean(Fitted.AVG, na.rm = TRUE),
                             Price = mean(Price, na.rm = TRUE)),
                           by = .(Market, Vaccine)])
  
  # compute absolute percent error
  VE.check[, NNET.APE := 100 * (abs(Price - Fitted.NNET) / Price)]
  VE.check[, GLM.APE := 100 * (abs(Price - Fitted.GLM) / Price)]
  VE.check[, AVG.APE := 100 * (abs(Price - Fitted.AVG) / Price)]
  
  # summarize absolute percent error
  summary(VE.check[,.(NNET.APE, GLM.APE, AVG.APE)])
  
  # plot 6 residual plots of VE.check for each model
  VE.check.nnet.plots = residplots(actual = VE.check$Price, fit = VE.check$Fitted.NNET, histlabel.y = -1/2, binwidth = 2)
  do.call(grid.arrange,  c(VE.check.nnet.plots, nrow = 2))
  
  VE.check.glm.plots = residplots(actual = VE.check$Price, fit = VE.check$Fitted.GLM, histlabel.y = -1/2, binwidth = 2)
  do.call(grid.arrange,  c(VE.check.glm.plots, nrow = 2))
  
  VE.check.avg.plots = residplots(actual = VE.check$Price, fit = VE.check$Fitted.AVG, histlabel.y = -1/2, binwidth = 2)
  do.call(grid.arrange,  c(VE.check.avg.plots, nrow = 2))
  
  # compute performance matrics by market for each model
  VE.metrics = lapply(unique(VE.check$Market), function(m)
    list("NNET" = h2o.make_metrics(predicted = as.h2o(VE.check[Market == m,.(Fitted.NNET)]), actuals = as.h2o(VE.check[Market == m,.(Price)])),
         "GLM" = h2o.make_metrics(predicted = as.h2o(VE.check[Market == m,.(Fitted.GLM)]), actuals = as.h2o(VE.check[Market == m,.(Price)])),
         "AVG" = h2o.make_metrics(predicted = as.h2o(VE.check[Market == m,.(Fitted.AVG)]), actuals = as.h2o(VE.check[Market == m,.(Price)]))))
  
  names(VE.metrics) = unique(VE.check$Market)
  
  # include overall performance metrics
  VE.metrics$Overall = list("NNET" = h2o.make_metrics(predicted = as.h2o(VE.check[,.(Fitted.NNET)]), actuals = as.h2o(VE.check[,.(Price)])),
                            "GLM" = h2o.make_metrics(predicted = as.h2o(VE.check[,.(Fitted.GLM)]), actuals = as.h2o(VE.check[,.(Price)])),
                            "AVG" = h2o.make_metrics(predicted = as.h2o(VE.check[,.(Fitted.AVG)]), actuals = as.h2o(VE.check[,.(Price)])))
  
  # check out model performance
  VE.metrics
  
  # pick a set of countries to plot prices for
  countries = c("NICARAGUA", "Nicaragua", 
                "ELSALVADOR", "El Salvador", 
                "DOMINICANREPUBLIC", "Dominican Republic (the)", 
                "BRAZIL", "Brazil", 
                "URUGUAY", "Uruguay", 
                "UNITEDSTATESOFAMERICA", "CDC")
  VE.mod = data.table(VE[Country %in% countries])
  
  # set up a color for each country
  color.set = c("cornflowerblue", "blue", 
                "lightcoral", "red",
                "plum3", "purple",
                "burlywood4", "saddlebrown",
                "seagreen3", "forestgreen",
                "gray50", "black")
  
  # update countries to be a table that pairs up duplicate countries
  countries = data.table(Country = countries, 
                         Country2 = c("NICARAGUA", "NICARAGUA", 
                                      "ELSALVADOR", "ELSALVADOR", 
                                      "DOMINICANREPUBLIC", "DOMINICANREPUBLIC", 
                                      "BRAZIL", "BRAZIL", 
                                      "URUGUAY", "URUGUAY", 
                                      "UNITEDSTATESOFAMERICA", "UNITEDSTATESOFAMERICA"))
  
  # give VE.mod and countries an id column to preserve row order
  VE.mod[, id := 1:nrow(VE.mod)]
  countries[, id := 1:nrow(countries)]
  
  # join countries onto VE.mod
  setkey(VE.mod, Country)
  setkey(countries, Country)
  VE.mod = countries[VE.mod]
  
  # order VE.mod and countries by id then remove id
  VE.mod = VE.mod[order(id)]
  VE.mod[, id := NULL]
  countries = countries[order(id)]
  countries[, id := NULL]
  
  # make Country, Country2, and Vaccine into factors
  VE.mod[, Country := factor(Country, levels = countries$Country)]
  VE.mod[, Country2 := factor(Country2, levels = unique(countries$Country2))]
  VE.mod[, Vaccine := factor(Vaccine, levels = unique(VE.mod$Vaccine))]
  
  # plot the GLM price points of various countries
  glm.pred = ggplot(VE.mod, aes(x = Vaccine, 
                                 y = Fitted.GLM, 
                                 color = Country, 
                                 group = Country)) + 
    geom_point(size = 3, na.rm = TRUE) +
    geom_line(size = 1, na.rm = TRUE) +
    facet_wrap(~Country2) + 
    scale_color_manual(values = color.set) +
    scale_y_continuous(labels = dollar) + 
    ggtitle("Country Price Predictions") + 
    labs(x = "Vaccine", y = "Price", color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "none", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 2, byrow = FALSE))
  
  glm.pred
  
  # plot the NNET price points of various countries
  nnet.pred = ggplot(VE.mod, aes(x = Vaccine, 
                                 y = Fitted.NNET, 
                                 color = Country, 
                                 group = Country)) + 
    geom_point(size = 3, na.rm = TRUE) +
    geom_line(size = 1, na.rm = TRUE) +
    facet_wrap(~Country2) + 
    scale_color_manual(values = color.set) +
    scale_y_continuous(labels = dollar) + 
    ggtitle("Country Price Predictions") + 
    labs(x = "Vaccine", y = "Price", color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "none", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 2, byrow = FALSE))
  
  nnet.pred
  
  # plot the AVG price points of various countries
  avg.pred = ggplot(VE.mod, aes(x = Vaccine, 
                                 y = Fitted.AVG, 
                                 color = Country, 
                                 group = Country)) + 
    geom_point(size = 3, na.rm = TRUE) +
    geom_line(size = 1, na.rm = TRUE) +
    facet_wrap(~Country2) + 
    scale_color_manual(values = color.set) +
    scale_y_continuous(labels = dollar) + 
    ggtitle("Country Price Predictions") + 
    labs(x = "Vaccine", y = "Price", color = "") + 
    theme_bw(base_size = 20) +
    theme(legend.position = "none", 
          legend.key.size = unit(.25, "in"),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 2, byrow = FALSE))
  
  avg.pred
  
  # lets assign prices to outlier countries that are the same as a similar country
  # the most similar country will be computed based on squared distance
  
  # get the indicators that decribe countries with good prices
  good.countries = data.table(VE[outlier == 0, .(Country, Market, Gov.VEpb, GNIpc, Birth.mortality)])
  good.countries = good.countries[!duplicated(good.countries)]
  
  # get the indicators that decribe countries with bad prices
  bad.countries = data.table(VE[(outlier == 1 | is.na(outlier)) & Country != "CDC", .(Country, Market, Gov.VEpb, GNIpc, Birth.mortality)])
  bad.countries = bad.countries[!duplicated(bad.countries)]
  bad.countries = bad.countries[Market != "PAHO-eigen"]
  
  # lets find a match for country k in good.countries
  # w represents the width percentage to consider a country to have the same numeric value
  k = bad.countries$Country[8]
  w = 0.05
  
  rbind(bad.countries[Country == k],
        good.countries[GNIpc >= (1 - w) * unname(unlist(bad.countries[Country == k, .(GNIpc)])) &
                       GNIpc <= (1 + w) * unname(unlist(bad.countries[Country == k, .(GNIpc)]))])
  
  # create a table of country matches
  country.matches = data.table(bad = c("ANGOLA", "CENTRALAFRICANREPUBLIC", "MONACO", "SANMARINO", "COOKISLANDS", "DEMOCRATICPEOPLESREPUBLICOFKOREA", "DJIBOUTI", "NIUE"), 
                               good = c("EGYPT", "NIGER", "SWITZERLAND", "AUSTRALIA", "BARBADOS", "ZIMBABWE", "LAOPEOPLESDEMOCRATICREPUBLIC", "SAMOA"))
  
  # update each bad country with the prices from their good country match
  for(k in 1:nrow(country.matches))
  {
    VE[Country == country.matches$bad[k], Fitted.GLM := unname(unlist(VE[Country == country.matches$good[k], .(Fitted.GLM)]))]
    VE[Country == country.matches$bad[k], Fitted.NNET := unname(unlist(VE[Country == country.matches$good[k], .(Fitted.NNET)]))]
    VE[Country == country.matches$bad[k], Fitted.AVG := unname(unlist(VE[Country == country.matches$good[k], .(Fitted.AVG)]))]
    VE[Country == country.matches$bad[k], outlier := unname(unlist(VE[Country == country.matches$good[k], .(outlier)]))]
  }
  
  # write out the results
  write.csv(VE, "Country-Reservation-Prices.csv", row.names = FALSE)
  write.csv(DT, "Market-Reservation-Prices.csv", row.names = FALSE)
  
  # plot the neural network
  plot(neuralnet(formula = Price ~ Gov.VEpb + DTaP + DTwP + HepB + Hib + IPV + OPV,
                 data = DT.mod, hidden = c(10), threshold = 1e50, stepmax = 1), 
       intercept = TRUE,
       information = FALSE,
       show.weights = FALSE,
       fontsize = 20,
       radius = 1/4,
       col.entry = "blue", 
       col.hidden = "red",
       col.out = "forestgreen")
  
  # clean up work space
  gc()
  
  # shutdown the h2o instance
  h2o.removeAll()
  h2o.shutdown(prompt = FALSE)
  
} else
{
  # import the price data
  VE = data.table(read.csv("Country-Reservation-Prices.csv", stringsAsFactors = FALSE))
  DT = data.table(read.csv("Market-Reservation-Prices.csv", stringsAsFactors = FALSE))
}

}

# -----------------------------------------------------------------------------------
# ---- Clustering Countries into Markets --------------------------------------------
# -----------------------------------------------------------------------------------

{

# get income and birth data from abp
income = na.omit(data.table(abp[Markets == 4, .(GNIpc = GNI, BirthCohort = AnnBirths)]))
income = na.omit(data.table(index[, .(GNIpc, Market)]))
income = data.table(model.matrix(~., data = income)[,-1])

# normalize the data set for kmeans clustering
income = data.table(scale(income))

# build a kmeans model for 2, 4, 8, 9, 12, 13 market systems
segmentations = c(2, 4, 8, 9, 12, 13)
km.mods = lapply(segmentations, function(k)
{
  # build the model
  set.seed(42)
  output = kmeans(x = income, 
                  centers = k, 
                  nstart = 50, 
                  iter.max = 50)
  
  # free memory
  gc()
  
  return(output)
})

# extract the cluster solutions from each model
income.clusters = foreach(i = 1:length(segmentations), .combine = "cbind") %do%
{
  return(as.numeric(km.mods[[i]]$cluster))
}

# give each column better names
colnames(income.clusters) = paste0("Markets", segmentations)

# add each of the cluster solutions to our data set
income.clusters = data.table(cbind(income.clusters, income))

# convert the cluster solution columns into long format with 2 columns: Model and Cluster
income.clusters = melt(income.clusters, 
                       measure.vars = paste0("Markets", segmentations), 
                       variable.name = "Model", 
                       value.name = "Cluster")

# compute the distance matrix used by kmeans
km.dmat = dist(income)

# lets create cluster plots for each model
km.plots = lapply(unique(income.clusters$Model), function(i)
{
  # extract model i from pca.clusters
  DT = data.table(income.clusters[Model == i])
  
  # remove Model from DT
  DT[, Model := NULL]
  
  # build cluster plots
  output = plot.clusters(dat = DT, 
                         cluster.column.name = "Cluster",
                         distance.matrix = km.dmat,
                         DC.title = paste("Discriminant Coordinate Cluster Plot\nK-Means ", i), 
                         pairs.title = paste("Cluster Pairs Plot\nK-Means ", i), 
                         silhouette.title = paste("Silhouette Width\nK-Means ", i))
  
  return(output)
})

# look at the silhouette plots of each model
windows()
km.plots[[1]]$plot.sil.width

windows()
km.plots[[2]]$plot.sil.width

windows()
km.plots[[3]]$plot.sil.width

windows()
km.plots[[4]]$plot.sil.width

windows()
km.plots[[5]]$plot.sil.width

windows()
km.plots[[6]]$plot.sil.width

# update income to have the original data
income = na.omit(data.table(index[,.(Country, Market, GNIpc)]))

# add the cluster solutions to income
income[, Markets2 := unname(unlist(income.clusters[Model == "Markets2", .(Cluster)]))]
income[, Markets4 := unname(unlist(income.clusters[Model == "Markets4", .(Cluster)]))]
income[, Markets8 := unname(unlist(income.clusters[Model == "Markets8", .(Cluster)]))]
income[, Markets9 := unname(unlist(income.clusters[Model == "Markets9", .(Cluster)]))]
income[, Markets12 := unname(unlist(income.clusters[Model == "Markets12", .(Cluster)]))]
income[, Markets13 := unname(unlist(income.clusters[Model == "Markets13", .(Cluster)]))]

# compute the median GNIpc of each cluster solution, to relabel clusters by decreasing GNIpc
M2 = data.table(income[, .(GNIpcMID = median(GNIpc), Size = .N), by = .(Markets2)])
M2 = M2[order(GNIpcMID, decreasing = TRUE)]
M2[, New2 := 1:nrow(M2)]

M4 = data.table(income[, .(GNIpcMID = median(GNIpc), Size = .N), by = .(Markets4)])
M4 = M4[order(GNIpcMID, decreasing = TRUE)]
M4[, New4 := 1:nrow(M4)]

M8 = data.table(income[, .(GNIpcMID = median(GNIpc), Size = .N), by = .(Markets8)])
M8 = M8[order(GNIpcMID, decreasing = TRUE)]
M8[, New8 := 1:nrow(M8)]

M9 = data.table(income[, .(GNIpcMID = median(GNIpc), Size = .N), by = .(Markets9)])
M9 = M9[order(GNIpcMID, decreasing = TRUE)]
M9[, New9 := 1:nrow(M9)]

M12 = data.table(income[, .(GNIpcMID = median(GNIpc), Size = .N), by = .(Markets12)])
M12 = M12[order(GNIpcMID, decreasing = TRUE)]
M12[, New12 := 1:nrow(M12)]

M13 = data.table(income[, .(GNIpcMID = median(GNIpc), Size = .N), by = .(Markets13)])
M13 = M13[order(GNIpcMID, decreasing = TRUE)]
M13[, New13 := 1:nrow(M13)]

# update M12 to be M13 and then combine the first two Markets
M12 = data.table(M13[,.(Markets13, New12 = c(1, 1:12))])

# update M8 to be M9 and then combine the first two Markets
M8 = data.table(M9[,.(Markets9, New8 = c(1, 1:8))])

# update M2 and M4
M2 = M2[,.(Markets2, New2)]
M4 = M4[,.(Markets4, New4)]

# give income an id to preserve original row order
income[, id := 1:nrow(income)]

# join M2, M4, M8, and M12 onto income
setkey(M2, Markets2)
setkey(income, Markets2)
income = M2[income]

setkey(M4, Markets4)
setkey(income, Markets4)
income = M4[income]

setkey(M8, Markets9)
setkey(income, Markets9)
income = M8[income]

setkey(M12, Markets13)
setkey(income, Markets13)
income = M12[income]

# order income by id and remove it
income = income[order(id)]
income[, id := NULL]

# update income to have the final market assignments
income[, Markets2 := New2]
income[, Markets4 := New4]
income[, Markets8 := New8]
income[, Markets12 := New12]
income = income[,.(Country, Market, GNIpc, Markets2, Markets4, Markets8, Markets12)]

# write out the results
write.csv(income, "Country-Income-Clustering.csv", row.names = FALSE)

}

# -----------------------------------------------------------------------------------
# ---- Computing Fair Prices for Vaccines per Country -------------------------------
# -----------------------------------------------------------------------------------

#####################################################################################
#### double check this approach later ###############################################
#####################################################################################

{

# do we need to compute prices?
compute.prices = FALSE

if(compute.prices)
{
  # get the vaccine, birth, and income data
  fair = na.omit(data.table(dat.full[,.(Country, Year, Gov.Expenditure.on.Vaccines, All.Expenditure.on.Vaccines, Birth.rate.crude.per.1000.people, GNI, Population.total, Mortality.rate.infant.per.1000.live.births)]))
  
  # compute BirthCohort
  fair[, BirthCohort := Birth.rate.crude.per.1000.people * (Population.total / 1000)]
  
  # compute CohortMortality ~ portion of the birth cohort that is lost
  fair[, CohortMortality := Mortality.rate.infant.per.1000.live.births / (BirthCohort / 1000)]
  
  # order by Country and Year
  fair = fair[order(Country, Year)]
  
  # only keep the necessary information in fair and fair.avg
  fair = data.table(fair[,.(Country, Year, All.Expenditure.on.Vaccines, Gov.Expenditure.on.Vaccines, BirthCohort)])
  
  # rename the columns of fair
  setnames(fair, c("Country", "Year", "Total.Budget", "Gov.Budget", "BirthCohort"))
  
  # compute avgerage total vaccine expenditure by country
  fair.avg = data.table(fair[, .(Total.Budget = mean(Total.Budget, na.rm = TRUE),
                                 Gov.Budget = mean(Gov.Budget, na.rm = TRUE)), 
                             by = .(Country)])
  
  # get the dosage and mapping data
  dose = data.table(read.csv("dosage-current.csv", stringsAsFactors = FALSE))
  mapping = data.table(read.csv("bundle-antigen-mapping-52B.csv", stringsAsFactors = FALSE))
  
  # remove all market information from dose
  dose = dose[Markets == 2, .(Antigen, DosePerChild)]
  dose = dose[!duplicated(dose)]
  
  # create DosePortion
  dose[, DosePortion := DosePerChild / sum(DosePerChild)]
  
  # join dose onto mapping
  setkey(dose, Antigen)
  setkey(mapping, Antigen)
  dose = dose[mapping]
  
  # order dose by Bundle and Antigen
  dose = dose[order(Bundle, Antigen)]
  
  # reorder the columns of dose
  setcolorder(dose, c("Bundle", "Antigen", "DosePerChild", "DosePortion"))
  
  # create a copy of abp
  fair.abp = data.table(abp[Markets == 4, .(Country, Gov.Expenditure.on.Vaccines.Portion.of.Total, AnnBirths, GNI, MarketID)])
  
  # give fair.abp an id column to preserve row order
  fair.abp[, id := 1:nrow(fair.abp)]
  
  # join vac onto fair.abp
  setkey(fair.avg, Country)
  setkey(fair.abp, Country)
  fair.abp = fair.avg[fair.abp]
  
  # order fair.abp by id then remove id
  fair.abp = fair.abp[order(id)]
  fair.abp[, id := NULL]
  
  # make MarketID into a factor variable
  fair.abp[, MarketID := factor(MarketID, levels = 1:4)]
  
  # get Countrys with missing data
  fair.abp.NA = data.table(na.omit(fair.abp, invert = TRUE))
  fair.abp.missing = sort(unique(fair.abp.NA$Country))
  
  # compute the portion of missing data
  length(fair.abp.missing) / length(unique(fair.abp$Country))
  
  # split up the abp into keep and impute:
  # keep will retain the variables we want but don't want to use for imputation
  # impute will retain the variables we want to use for imputation
  abp.keep = data.table(fair.abp[, .(Country, Gov.Budget, Gov.Expenditure.on.Vaccines.Portion.of.Total)])
  abp.impute = data.table(fair.abp[, !c("Country", "Gov.Budget", "Gov.Expenditure.on.Vaccines.Portion.of.Total"), with = FALSE])
  
  # use random forests to impute the missing values
  abp.impute = missRanger(abp.impute, num.trees = 100, seed = 42, num.threads = 1)
  
  # make abp.impute into a data table
  abp.impute = data.table(abp.impute)
  
  # combine keep and impute together again
  fair.abp = data.table(cbind(abp.keep, abp.impute))
  
  # replace all NAs in Gov.Budget with: "Gov.Expenditure.on.Vaccines.Portion.of.Total" * "Total.Budget"
  fair.abp[is.na(Gov.Budget), Gov.Budget := Gov.Expenditure.on.Vaccines.Portion.of.Total * Total.Budget]
  
  # ensure Total.Budget is never less than Gov.Budget
  fair.abp[, Total.Budget := pmax(Total.Budget, Gov.Budget)]
  
  # only keep the necessary columns in fair.abp
  fair.abp = fair.abp[, .(Country, Total.Budget, Gov.Budget, BirthCohort = AnnBirths)]
  
  # add the dose information to abp.fair
  fair.abp = lapply(1:nrow(fair.abp), function(i)
  {
    # create a copy of dose
    DT = data.table(dose)
    
    # give country i info to DT
    DT[, Country := fair.abp$Country[i]]
    DT[, Total.Budget := fair.abp$Total.Budget[i]]
    DT[, Gov.Budget := fair.abp$Gov.Budget[i]]
    DT[, BirthCohort := fair.abp$BirthCohort[i]]
    
    return(DT)
  })
  
  # combine the list of tables into one table
  fair.abp = rbindlist(fair.abp)
  
  # compute the budget each country would allocate for each antigen based on DosePortion
  fair.abp[, Antigen.Total.Budget := DosePortion * Total.Budget]
  fair.abp[, Antigen.Gov.Budget := DosePortion * Gov.Budget]
  
  # compute the PricePerDose that a country would pay to acheive 90% national coverage (a GVAP goal)
  fair.abp[, Total.PricePerDose := Antigen.Total.Budget / (BirthCohort * DosePerChild * 0.90)]
  fair.abp[, Gov.PricePerDose := Antigen.Gov.Budget / (BirthCohort * DosePerChild * 0.90)]
  
  # create a copy of fair.abp to compute the price per bundle
  fair.abp.prices = data.table(fair.abp[, .(Total.PricePerBundle = sum(Total.PricePerDose),
                                            Gov.PricePerBundle = sum(Gov.PricePerDose),
                                            Total.Budget = mean(Total.Budget),
                                            Gov.Budget = mean(Gov.Budget),
                                            BirthCohort = mean(BirthCohort)),
                                        by = .(Country, Bundle)])
  
  # compute the cost of purchasing any bundle for 90% of the BirthCohort
  fair.abp.prices[, Gov.Cost := Gov.PricePerBundle * BirthCohort * 0.90]
  fair.abp.prices[, Total.Cost := Total.PricePerBundle * BirthCohort * 0.90]
  
  # compute the portion of the budget that would be taken up by buying any bundle for 90% of the BirthCohort
  fair.abp.prices[, Gov.Portion := Gov.Cost / Gov.Budget]
  fair.abp.prices[, Total.Portion := Total.Cost / Total.Budget]
  
  # look at summary statistics for the portions
  summary(fair.abp.prices$Gov.Portion)
  summary(fair.abp.prices$Total.Portion)
  
  # add the dose information to abp.fair
  fair = lapply(1:nrow(fair), function(i)
  {
    # create a copy of dose
    DT = data.table(dose)
    
    # give country i info to DT
    DT[, Country := fair$Country[i]]
    DT[, Year := fair$Year[i]]
    DT[, Total.Budget := fair$Total.Budget[i]]
    DT[, Gov.Budget := fair$Gov.Budget[i]]
    DT[, BirthCohort := fair$BirthCohort[i]]
    
    return(DT)
  })
  
  # combine the list of tables into one table
  fair = rbindlist(fair)
  
  # compute the budget each country would allocate for each antigen based on DosePortion
  fair[, Antigen.Total.Budget := DosePortion * Total.Budget]
  fair[, Antigen.Gov.Budget := DosePortion * Gov.Budget]
  
  # compute the PricePerDose that a country would pay to acheive 90% coverage (a GVAP goal)
  fair[, Total.PricePerDose := Antigen.Total.Budget / (BirthCohort * DosePerChild * 0.90)]
  fair[, Gov.PricePerDose := Antigen.Gov.Budget / (BirthCohort * DosePerChild * 0.90)]
  
  # create a copy of fair to compute the price per bundle
  fair.prices = data.table(fair[, .(Total.PricePerBundle = sum(Total.PricePerDose),
                                    Gov.PricePerBundle = sum(Gov.PricePerDose),
                                    Total.Budget = mean(Total.Budget),
                                    Gov.Budget = mean(Gov.Budget),
                                    BirthCohort = mean(BirthCohort)),
                                by = .(Country, Year, Bundle)])
  
  # compute the cost of purchasing any bundle for 90% of the BirthCohort
  fair.prices[, Gov.Cost := Gov.PricePerBundle * BirthCohort * 0.90]
  fair.prices[, Total.Cost := Total.PricePerBundle * BirthCohort * 0.90]
  
  # compute the portion of the budget that would be taken up by buying any bundle for 90% of the BirthCohort
  fair.prices[, Gov.Portion := Gov.Cost / Gov.Budget]
  fair.prices[, Total.Portion := Total.Cost / Total.Budget]
  
  # lets replace all NaN in fair.prices with 0
  fair.prices[is.nan(Gov.Portion), Gov.Portion := 0]
  
  # look at summary statistics for the portions
  summary(fair.prices$Gov.Portion)
  summary(fair.prices$Total.Portion)
  
  # write out fair.prices and fair.abp.prices
  write.csv(fair.abp.prices, "Fair-ABP-Prices.csv", row.names = FALSE)
  write.csv(fair.prices, "Fair-Prices.csv", row.names = FALSE)
  
} else
{
  # import the data
  fair.prices = data.table(read.csv("Fair-Prices.csv", stringsAsFactors = FALSE))
  fair.abp.prices = data.table(read.csv("Fair-ABP-Prices.csv", stringsAsFactors = FALSE))
}

}

# give fair.abp.prices an id column to preserve original row order
fair.abp.prices[, id := 1:nrow(fair.abp.prices)]

# extract the 4 market system from abp
m4 = data.table(abp[Markets == 4, .(Country, MarketID)])

# join m4 onto fair.abp.prices
setkey(m4, Country)
setkey(fair.abp.prices, Country)
fair.abp.prices = m4[fair.abp.prices]

# create a Market column for fair.abp.prices like in market.prices
fair.abp.prices[, Market := ifelse(MarketID == 4, "LIC", 
                                   ifelse(MarketID == 3, "LMIC", 
                                          ifelse(MarketID == 2, "HMIC", "HIC")))]

# create an average market price across the years in market.prices
market.prices.avg = data.table(market.prices[, .(PriceLow = mean(PriceLow), 
                                                 PriceHigh = mean(PriceHigh)),
                                             by = .(Vaccine, Market)])

# compute an average market price across the countries in fair.abp.prices
fair.abp.prices.avg = data.table(fair.abp.prices[, .(PriceTotal = mean(Total.PricePerBundle),
                                                     PriceGov = mean(Gov.PricePerBundle)),
                                                 by = .(Vaccine, Market)])

# create an average market price across the years in market.prices
market.prices.max = data.table(market.prices[, .(PriceLow = max(PriceLow), 
                                                 PriceHigh = max(PriceHigh)),
                                             by = .(Vaccine, Market)])

#####################################################################################
# everything below is example code for clustering and unsupervised machine learning #
#####################################################################################

# -----------------------------------------------------------------------------------
# ---- Find Clusters ----------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# lets set up an experiment to try different clusters
doe = data.table(expand.grid(clusters = 2:20))

# choose the number of workers/threads and tasks for parallel processing
# specifying a value > 1 for workers means that multiple models in doe will be built in parallel
workers = max(1, floor((2/3) * detectCores()))
tasks = nrow(doe)

# set up a cluster if workers > 1, otherwise don't set up a cluster
if(workers > 1)
{
  # setup parallel processing
  cl = makeCluster(workers, type = "SOCK", outfile = "")
  registerDoSNOW(cl)
  
  # define %dopar%
  `%fun%` = `%dopar%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("kmeans clustering\n")
  cat(paste(workers, "workers started at", Sys.time(), "\n"))
  sink()
  
} else
{
  # define %do%
  `%fun%` = `%do%`
  
  # write out start time to log file
  sink(myfile, append = TRUE)
  cat("\n------------------------------------------------\n")
  cat("kmeans clustering\n")
  cat(paste("task 1 started at", Sys.time(), "\n"))
  sink()
}

# build each model in doe
km.mods = foreach(i = 1:tasks) %fun%
{
  # build the model
  output = kmeans(x = pca.dat, 
                  centers = doe$clusters[i], 
                  nstart = 500, 
                  iter.max = 500)
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
  sink()
  
  return(output)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
sink()

# end the cluster if it was set up
if(workers > 1)
{
  stopCluster(cl)
}

# extract the cluster solutions from each model
pca.clusters = foreach(i = 1:tasks, .combine = "cbind") %do%
{
  return(as.numeric(km.mods[[i]]$cluster))
}

# give each column better names
colnames(pca.clusters) = paste0("model", 1:ncol(pca.clusters))

# add each of the cluster solutions to our data set
pca.clusters = cbind(pca.clusters, pca.dat)

# convert the cluster solution columns into long format with 2 columns: Model and Cluster
pca.clusters = melt(pca.clusters, 
                    measure.vars = names(pca.clusters)[1:nrow(doe)], 
                    variable.name = "Model", 
                    value.name = "Cluster")

# keep just the number in the column Model
pca.clusters[, Model := as.numeric(gsub("model", "", Model))]

# compute the distance matrix used by kmeans
km.dmat = dist(pca.dat)

# lets create cluster plots for each model
km.plots = lapply(1:tasks, function(i) ynew = as.data.frame(predict(nnet.mod2, newdata = h2o.cbind(DT.X.h2o, my.folds.X)))$predict
{
  # extract model i from pca.clusters
  DT = data.table(pca.clusters[Model == i])
  
  # remove Model from DT
  DT[, Model := NULL]
  
  # build cluster plots
  output = plot.clusters(dat = DT, 
                         cluster.column.name = "Cluster",
                         distance.matrix = km.dmat,
                         DC.title = paste("Discriminant Coordinate Cluster Plot\nK-Means Model", i), 
                         pairs.title = paste("Cluster Pairs Plot\nK-Means Model", i), 
                         silhouette.title = paste("Silhouette Width\nK-Means Model", i))
  
  return(output)
})

# compare the silhouette plots of each model
windows()
km.plots[[1]]$plot.sil.width

windows()
km.plots[[2]]$plot.sil.width

windows()
km.plots[[3]]$plot.sil.width

windows()
km.plots[[4]]$plot.sil.width

windows()
km.plots[[5]]$plot.sil.width

windows()
km.plots[[6]]$plot.sil.width

windows()
km.plots[[7]]$plot.sil.width

windows()
km.plots[[8]]$plot.sil.width

windows()
km.plots[[9]]$plot.sil.width

windows()
km.plots[[10]]$plot.sil.width

windows()
km.plots[[11]]$plot.sil.width

windows()
km.plots[[12]]$plot.sil.width

windows()
km.plots[[13]]$plot.sil.width

windows()
km.plots[[14]]$plot.sil.width

windows()
km.plots[[15]]$plot.sil.width

windows()
km.plots[[16]]$plot.sil.width

windows()
km.plots[[17]]$plot.sil.width

windows()
km.plots[[18]]$plot.sil.width

windows()
km.plots[[19]]$plot.sil.width

# compare the 2D cluster plots of each model
windows()
km.plots[[2]]$plot.2D

windows()
km.plots[[8]]$plot.2D

windows()
km.plots[[11]]$plot.2D

# compare the 3D cluster plots of each model
windows()
grid.draw(km.plots[[2]]$plot.3D)

windows()
grid.draw(km.plots[[8]]$plot.3D)

windows()
grid.draw(km.plots[[11]]$plot.3D)

# compare the cluster pairs plots of each model
windows()
km.plots[[2]]$plot.pairs

windows()
km.plots[[8]]$plot.pairs

windows()
km.plots[[11]]$plot.pairs

# lets go with model 11
# extract PCA data
DT = data.table(pca.clusters[Model == 11])

# remove the model column from DT
DT[, Model := NULL]

# make cluster into a factor variable
DT[, Cluster := factor(Cluster, levels = sort(unique(Cluster)))]

# only keep the Cluster variable in DT and give DT the original variables in dat
DT = data.table(cbind(DT[,.(Cluster)], dat))

# plot TSS across Cluster
tss = cbind(DT[, .(Cluster)], TSS = apply(DT[,!"Cluster"], 1, max))

TSS.plot = ggplot(tss, aes(x = Cluster, y = TSS / 1e9, color = Cluster, fill = Cluster)) +
  geom_jitter(alpha = 1/3) +
  stat_summary(fun.y = mean, color = "black", geom = "point", shape = 18, size = 5) +
  scale_y_continuous(label = dollar_format(suffix = "B")) +
  ggtitle("Total Social Surplus") + 
  labs(x = "Cluster", y = "Value") + 
  theme_bw(25) + 
  theme(legend.position = "none", 
        legend.key.size = unit(.25, "in"), 
        plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1, alpha = 1), nrow = 1))

TSS.plot

# update factors to be binary indicating they took on a value or not
binary = as.matrix(DT[,!"Cluster"] / DT[,!"Cluster"])
binary[is.nan(binary)] = 0

# update DT to have just binary factors
DT = data.table(Cluster = DT$Cluster, binary)

}

# -----------------------------------------------------------------------------------
# ---- Predict Clusters -------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# the classes are imbalanced so lets define the case.weights parameter where a class with more observations is weighted less
case.weights = table(DT$Cluster)
case.weights = max(case.weights) / case.weights

# make case.weights into a table so we can join it onto mod.dat
case.weights = data.table(Cluster = names(case.weights), 
                          value = as.numeric(case.weights))

# create a copy of DT
mod.dat = data.table(DT)

# give mod.dat an ID column so we can maintain the original order of rows
mod.dat[, ID := 1:nrow(mod.dat)]

# set Cluster as the key column for joining purposes
setkey(mod.dat, Cluster)
setkey(case.weights, Cluster)

# join case.weights onto mod.dat
case.weights = data.table(case.weights[mod.dat])

# order case.weights and mod.dat by ID
case.weights = case.weights[order(ID)]
mod.dat = mod.dat[order(ID)]

# extract the value column of case.weights
case.weights = case.weights$value

# remove the ID column in mod.dat
mod.dat[, ID := NULL]

# choose how many tasks and threads to use
tasks = 2
num.threads = max(1, floor((2/3) * detectCores()))

# choose how many trees to use
num.trees = 500

# set up seeds for reproducability
set.seed(42)
seeds = sample(1:1000, tasks)

# write out start time to log file
sink(myfile, append = TRUE)
cat("\n------------------------------------------------\n")
cat("random forest - variable importance\n")
cat(paste("task 1 started at", Sys.time()), "\n")
sink()

# build random forest models
var.imp = foreach(i = 1:tasks) %do%
{
  # build the random forest
  mod = ranger(Cluster ~ ., 
               data = mod.dat,
               num.trees = num.trees,
               case.weights = case.weights,
               num.threads = num.threads,
               seed = seeds[i],
               importance = "impurity",
               verbose = FALSE)
  
  # extract variable importance
  imp = ranger:::importance(mod)
  
  # make imp into a table
  imp = data.table(variable = names(imp), 
                   value = as.numeric(imp))
  
  # order by variable name
  imp = imp[mixedorder(variable)]
  
  # add the task number to imp
  imp[, task := i]
  
  # free memory
  gc()
  
  # export progress information
  sink(myfile, append = TRUE)
  cat(paste("task", i, "of", tasks, "finished at", Sys.time()), "\n")
  sink()
  
  return(imp)
}

# write out end time to log file
sink(myfile, append = TRUE)
cat(paste(tasks, "tasks finished at", Sys.time()), "\n")
sink()

# combine the list of data tables into one table
var.imp = rbindlist(var.imp)

# put importance on a 0-1 scale for easy comparison
var.imp[, value := rescale(value)]

# average importance of variables
var.imp = var.imp[, .(value = mean(value)), by = .(variable)]

# order by importance
var.imp = var.imp[order(value, decreasing = TRUE)]

# make variable a factor for plotting purposes
var.imp[, variable := factor(variable, levels = unique(variable))]

# plot a barplot of variable importance
var.imp.plot = ggplot(var.imp, aes(x = variable, y = value, fill = value, color = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Relative Gini Importance\nRandom Forests") +
  labs(x = "Variable", y = "Scaled Importance") +
  scale_y_continuous(labels = percent) +
  scale_fill_gradient(low = "yellow", high = "red") +
  scale_color_gradient(low = "yellow", high = "red") +
  theme_dark(30) +
  theme(legend.position = "none", 
        plot.title = element_text(hjust = 0.5), 
        axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(), 
        panel.grid.major.x = element_blank())

windows()
var.imp.plot

# lets onyl keep the variables with at least 25% importance
DT = DT[, c("Cluster", as.character(var.imp[value >= 0.25, variable])), with = FALSE]

# lets rebuild our model
mod = ranger(Cluster ~ ., 
             data = DT,
             num.trees = 500,
             case.weights = case.weights,
             num.threads = num.threads,
             seed = 42,
             verbose = FALSE)

# lets check out the confusion martrix
mod$confusion.matrix

}




