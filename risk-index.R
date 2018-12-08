# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# set the path of where the input files are
mywd = "C:/ ... /All-Data"

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



