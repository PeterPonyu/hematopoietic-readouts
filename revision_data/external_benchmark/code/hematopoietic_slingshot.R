args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
folder <- args[[1L]]
root <- args[[2L]]
ends <- strsplit(args[[3L]], ',', fixed=TRUE)[[1L]]
lib <- args[[4L]]
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(slingshot))
set.seed(0)
x <- as.matrix(read.csv(gzfile(file.path(folder,'pca.csv.gz')), row.names=1, check.names=FALSE))
obs <- read.csv(file.path(folder,'clusters.csv'), row.names=1, check.names=FALSE,
                colClasses=c('character','character'))
stopifnot(identical(rownames(x),rownames(obs)))
fit <- slingshot(x, clusterLabels=factor(obs$leiden), start.clus=root,
                 end.clus=ends, maxit=10, approx_points=150)
pt <- slingPseudotime(fit, na=FALSE)
weights <- slingCurveWeights(fit, as.probs=FALSE)
stopifnot(identical(dim(pt),dim(weights)), identical(rownames(pt),rownames(x)))
write.csv(pt,gzfile(file.path(folder,'pseudotime.csv.gz')))
write.csv(weights,gzfile(file.path(folder,'curve_weights.csv.gz')))
lineages <- slingLineages(fit)
writeLines(vapply(lineages,paste,collapse=',',FUN.VALUE=character(1)),file.path(folder,'lineage_paths.txt'))
writeLines(as.character(packageVersion('slingshot')),file.path(folder,'package_version.txt'))
writeLines(capture.output(sessionInfo()),file.path(folder,'sessionInfo.txt'))
saveRDS(fit,file.path(folder,'fit.rds'))
