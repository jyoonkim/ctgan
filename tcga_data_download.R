library(TCGAbiolinks)
library(SummarizedExperiment)
library(data.table)
library(readxl)

if(!file.exists("./data/survival_data")){dir.create("./data/survival_data")}
if(!file.exists("./data/raw_ge_data")){dir.create("./data/raw_ge_data")}
if(!file.exists("./data/clinical_info_data")){dir.create("./data/clinical_info_data")}

current_dir = getwd()

#### 1. Download gene expression ####

gene_expression_download <- function(){
  save_dir = paste(current_dir, "/data/raw_ge_data/", sep="")
  
  print("Downloading gene expression data..")
  
  projects = getGDCprojects()$project_id
  projects = projects[substr(projects, 0, 4) == "TCGA"] # All types of TCGA cancer
  
  #projects = list(projects[7]) ### this is only COAD
  
  for(pj in projects){
    print("cancer type")
    print(pj)
    data <- NULL
    eset <- NULL
    
    query <- GDCquery(project=pj,
                      data.category="Transcriptome Profiling",
                      data.type="Gene Expression Quantification",
                      workflow.type="STAR - Counts")
    
    GDCdownload(query)
    data <- GDCprepare(query)
    
    eset <- assay(data, "fpkm_unstrand")
    eset <- as.data.frame(t(eset))
    
    f_name <- paste(pj, ".txt", sep="")
    write.table(eset,paste(save_dir,f_name, sep=""),sep="\t")
    #write.table(eset,f_name,sep="\t")
    
  }
}


#### 2. ensemble id, gene symbol matching #####
ensemble_id_gene_symbol_match <- function(){
  
  save_dir = paste(current_dir, "/data/", sep="")
  
  data <- NULL
  eset <- NULL
  
  pj = projects[7]
  query <- GDCquery(project=pj,
                    data.category="Transcriptome Profiling",
                    data.type="Gene Expression Quantification",
                    workflow.type="STAR - Counts")
  
  #GDCdownload(query)
  data <- GDCprepare(query)
  
  eset <- as.data.frame(rowData(data))
  gene_symbol_df <-eset['gene_name']
  gene_symbol_df$ensemble_id <- row.names(gene_symbol_df)
  gene_symbol_df <- gene_symbol_df[c(2:1)]
  
  write.table(gene_symbol_df, paste(save_dir,'ensemble2genesymbol.csv', sep=""), row.names=FALSE,sep=",")
  #write.table(gene_symbol_df, 'ensemble2genesymbol.csv', row.names=FALSE,sep=",")
  
}


#### 3. Download survival data (event, days) ####
survival_data_download <- function(){                  
  
  save_dir = paste(current_dir, "/data/survival_data/", sep="")
  
  print("Downloading survival data..")
  projects = getGDCprojects()$project_id
  projects = projects[substr(projects, 0, 4) == "TCGA"] # All types of TCGA cancer
  
  #projects = list(projects[7])
  
  for(pj in projects){
    query <- GDCquery(project = pj, 
                      data.category = "Clinical",
                      data.type = "Clinical Supplement", 
                      data.format = "BCR Biotab")
    GDCdownload(query)
    clinical.BCRtab.all <- GDCprepare(query)
    pid = which(substr(names(clinical.BCRtab.all), 0, 10) == "clinical_p")
    clinical = as.data.frame(clinical.BCRtab.all[[pid]])
    clinical = clinical[-c(1,2),]
    clinical_names = clinical[,"bcr_patient_barcode"]
    
    if(sum(c("vital_status",  "last_contact_days_to",  "death_days_to") %in% names(clinical)) == 3){
      clinical_survival_pre = clinical[,c("vital_status",  "last_contact_days_to",  "death_days_to")]
      clinical_survival_pre[(clinical_survival_pre == "[Not Applicable]") | (clinical_survival_pre == "[Not Available]") | (clinical_survival_pre == "[Discrepancy]")] = 0
      clinical_survival_pre[(clinical_survival_pre == "Alive"), 1] = 0
      clinical_survival_pre[(clinical_survival_pre == "Dead"), 1] = 1
      clinical_survival = transform(clinical_survival_pre, vital_status = as.numeric(vital_status), last_contact_days_to = as.numeric(last_contact_days_to), death_days_to = as.numeric(death_days_to))
    }else if(sum(c("days_to_last_followup",  "days_to_death") %in% names(clinical)) == 2){
      clinical_survival_pre = clinical[,c("days_to_last_followup",  "days_to_death")]
      clinical_survival_pre[(clinical_survival_pre == "[Not Applicable]") | (clinical_survival_pre == "[Not Available]") | (clinical_survival_pre == "[Discrepancy]")] = 0
      clinical_survival_pre = as.data.frame(cbind(vital_status=as.numeric(!!as.numeric(clinical_survival_pre[,"days_to_death"])), clinical_survival_pre))
      clinical_survival = transform(clinical_survival_pre, vital_status = as.numeric(vital_status), days_to_last_followup = as.numeric(days_to_last_followup), days_to_death = as.numeric(days_to_death))
    }else{next}
    event_days = cbind(days = rowSums(clinical_survival[,2:3]), event = clinical_survival[,1])
    event_days[event_days<0] = 0
    rownames(event_days) = clinical_names
    
    file_name = strsplit(pj, split="-")[[1]][2]
    print("cancer type")
    print(file_name)
    full_file_name = paste("event_days_", file_name, ".csv", sep="")

    write.csv(event_days, paste(save_dir, full_file_name, sep=""), sep=",")
    #write.csv(event_days, full_file_name, sep=",")
  }
}


##### 4. Download completion date dataframe #####
completion_data_dataframe <- function(){
  
  save_dir = paste(current_dir, "/data/clinical_info_data/", sep="")
  
  projects = getGDCprojects()$project_id
  projects = projects[substr(projects, 0, 4) == "TCGA"]
  
  #projects = list(projects[7])
  
  for(pj in projects){
    query <- GDCquery(project = pj, 
                      data.category = "Clinical",
                      data.type = "Clinical Supplement", 
                      data.format = "BCR Biotab")
    
    clinical.BCRtab.all <- GDCprepare(query)
    pid = which(substr(names(clinical.BCRtab.all), 0, 10) == "clinical_p")
    clinical = as.data.frame(clinical.BCRtab.all[[pid]])
    clinical = clinical[-c(1,2),]
    clinical_names = clinical[,"bcr_patient_barcode"]
    
    complete_form_year <- clinical[, c("form_completion_date")]
    complete_form_year_df <- as.data.frame(cbind(complete_year=as.numeric(gsub("-","",complete_form_year))))
    
    rownames(complete_form_year_df) = clinical_names
    
    file_name = strsplit(pj, split="-")[[1]][2]
    full_file_name = paste("complete_form_year_", file_name, ".csv", sep="")
    
    write.csv(complete_form_year_df, paste(save_dir, full_file_name, sep=""), sep=",")
    #write.csv(complete_form_year_df, full_file_name, sep=",")
  }

}


gene_expression_download()
ensemble_id_gene_symbol_match()
survival_data_download()
completion_data_dataframe()
