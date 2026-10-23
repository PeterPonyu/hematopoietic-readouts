args <- commandArgs(TRUE)
stopifnot(length(args)==5)
.libPaths(c(args[[1]],.libPaths()))
suppressPackageStartupMessages(library(slingshot))
folder <- args[[2]]; source <- args[[3]]; root <- args[[4]]
ends <- strsplit(args[[5]],',',fixed=TRUE)[[1]]
dir.create(folder,recursive=TRUE,showWarnings=FALSE)
Sys.setenv(R4_STOPPING_CSV=file.path(folder,'stopping.csv'))
method <- getMethod('getCurves','PseudotimeOrdering')
native <- eval(body(method)[[2]][[3]],envir=environment(method))
stopifnot(is.function(native))
writeLines(deparse(native),file.path(folder,'method_source_before.R'))
original <- body(native)
logger <- quote(on.exit(write.csv(data.frame(iterations=it,maxit=maxit,
                   converged=hasConverged,threshold=thresh,
                   previous_distance=dist.old,final_distance=dist.new),
                   Sys.getenv('R4_STOPPING_CSV'),row.names=FALSE),add=TRUE))
body(native) <- as.call(c(as.name('{'),list(logger),as.list(original)[-1]))
writeLines(deparse(native),file.path(folder,'method_source_instrumented.R'))
setMethod('getCurves',signature(data='PseudotimeOrdering'),definition=native)
x <- as.matrix(read.csv(gzfile(file.path(source,'pca.csv.gz')),row.names=1,check.names=FALSE))
obs <- read.csv(file.path(source,'clusters.csv'),row.names=1,check.names=FALSE,
                colClasses=c('character','character'))
stopifnot(identical(rownames(x),rownames(obs)))
set.seed(0)
fit <- slingshot(x,clusterLabels=factor(obs$leiden),start.clus=root,end.clus=ends,
                 maxit=20,approx_points=150)
saveRDS(fit,file.path(folder,'fit.rds'))
pt <- slingPseudotime(fit,na=FALSE);weights <- slingCurveWeights(fit,as.probs=FALSE)
writeBin(as.double(pt),file.path(folder,'pseudotime.f64'),size=8,endian='little')
writeBin(as.double(weights),file.path(folder,'weights.f64'),size=8,endian='little')
writeLines(paste(dim(pt),collapse=','),file.path(folder,'shape.txt'))
writeLines(rownames(pt),file.path(folder,'cell_ids.txt'))
writeLines(capture.output(sessionInfo()),file.path(folder,'sessionInfo.txt'))
