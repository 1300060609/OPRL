#利用本程序可以实现一致性聚类分析，并基于分型结果进行生存分析得到相应风险因子（HR）及P值；


###############
#包的安装
###############

#已安装的R包
pkgs <- rownames(installed.packages())
#本程序所需包
packages_1=c("survival","ConsensusClusterPlus")
#将要安装的R包
packages_2=setdiff(packages_1,pkgs)
#安装所需R包
if(length(packages_2)>0){
  options(repos=structure(c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")))
  install.packages(packages_2)
  source("http://bioconductor.org/biocLite.R")
  biocLite(packages_2)
}

args <- commandArgs(TRUE)
library("survival")
library("ConsensusClusterPlus")

recuced <- read.table(args[1], sep = ',', header = T)
rownames(recuced) <- recuced$X
recuced <- as.matrix(recuced[, -1])
suppressWarnings(storage.mode(recuced) <- "numeric")
recuced <- na.omit(recuced)

# samplename <- colnames(recuced)
# n <- union(grep('T', samplename), grep('tumor', samplename))
# samplename <- gsub('tumor', '', gsub('T', '', samplename[n]))
# recuced <- recuced[, n]
# colnames(recuced) <- samplename


results = ConsensusClusterPlus(recuced, maxK=as.numeric(args[3]),
                               # reps=ceiling(ncol(recuced)/as.numeric(args[3])),  #根据样本总量和最大分组数确定每个分组下最大的样本数
                               reps=1000,
                               pItem=0.8,
                               pFeature=1,
                               #title="/var/www/galaxyoutput",
                               clusterAlg=args[4],   #'hc',
                               distance=args[5],   #"pearson",
                               innerLinkage="ward.D2", finalLinkage="ward.D2",
                               seed=NULL, #1262118388.71279,
                               title="result",  #生成图片和结果文件的目录
                               plot="pdf")
consensusClass <- as.matrix(results[[2]][["consensusClass"]])
colnames(consensusClass)="k=2"
consensusClass=cbind(sample=row.names(consensusClass), consensusClass)
row.names(consensusClass)=NULL
for(i in 3:as.numeric(15)){
  consensusClass_i <- as.matrix(results[[i]][["consensusClass"]])
  colnames(consensusClass_i)=paste0("k=",i)
  consensusClass_i=cbind(sample=row.names(consensusClass_i), consensusClass_i)
  row.names(consensusClass_i)=NULL
  consensusClass <- merge(consensusClass,consensusClass_i,by="sample")
}
write.table(consensusClass,'result/ConsensusResult.csv',sep=",",row.names=FALSE)

#设置停顿，用户自己输入最合适的K值
# cat("Please check your consensuscluster results [./result] and select the most appropriate cluster K: ")
clusterN <-as.integer(args[6])#as.numeric(readLines(file("stdin"),1))
consensusClass <- as.matrix(results[[clusterN]][["consensusClass"]])
colnames(consensusClass)='ClusterClass'
consensusClass=cbind(SampleNumber=row.names(consensusClass), consensusClass)
row.names(consensusClass)=NULL
runcolor <- rainbow(clusterN)


clinic_info <- read.table(args[2], header = T, sep = ',', na.strings = NA)   #'lihc_stomac_survival.csv'
clinic_cluster <- merge(clinic_info, consensusClass, by = 'SampleNumber', all.y = T)
rownames(clinic_cluster) <- clinic_cluster$SampleNumber
clinic_cluster <- clinic_cluster[, -1]
clinic_cluster$ClusterClass[which(clinic_cluster$Live_status == 2)] <- 1

### 构建生存分析，并进行cox回归评估
su_dfs <- Surv(as.numeric(as.matrix(clinic_cluster$Disease_Free_Survival)), as.numeric(as.matrix(clinic_cluster$Live_status)))
fit_dfs <- survfit(su_dfs ~ clinic_cluster$ClusterClass)
cox_hr <- coxph(su_dfs ~ clinic_cluster$ClusterClass)

### 提取分析结果中的重要参数
p <- survdiff(su_dfs ~ clinic_cluster$ClusterClass)
pv <- ifelse(is.na(p),next,(round(1 - pchisq(p$chisq, length(p$n) - 1),3)))[[1]]
cox_hr_sum <- summary(cox_hr)
HR <- signif(cox_hr_sum$coef[2], digits=2)
p.HR <- signif(cox_hr_sum$wald["pvalue"], digits=2)


out_result <- matrix(NA, nrow = 1, ncol = 3)
colnames(out_result) <- c('Logrank p', 'HR(95% CI for HR)', 'p(HR)')
out_result[1, ] <- c(pv, HR, p.HR)
write.table(out_result, 'survival_value.csv', sep = ',', quote = T, row.names = F)

#plot survival curve
pdf('dfs_survival1.pdf', width = 10, height = 8)
plot(fit_dfs, conf.int=F, mark.time = TRUE, xlab="Disease Free Survival (Days)", col=runcolor, ylab = "Survival Probability",lwd=3,cex.lab=1.4,cex.axis=1.4)


legend(0,0.4,
       legend=paste("Logrank p = ",pv[[1]],sep=""),bty="n",cex=1.0)
# legend(0,0.3,
#        legend=paste("HR(95% CI for HR) = ",HR,sep=""),bty="n",cex=1.0)
# legend(0,0.2,
#        legend=paste("p(HR) = ",p.HR,sep=""),bty="n",cex=1.0)
legend(65,0.2,
       legend=paste0('ClusterClass ', 1:clusterN),bty="n",cex=0.8,lwd=2,col=runcolor)
#
# legend(40,0.1,
#        legend=c(paste("high expressed=",low_n),paste("low expressed=",high_n)),bty="n",cex=1.0,lwd=3,col=c("red","blue"))
dev.off()
