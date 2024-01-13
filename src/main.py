from data_preprocessing import data_preprocess
from significant_genes_select import cox_significant_gene, sorted_significant_gene
from train import model_train
from generate_fake_data import make_fake_data
from generate_reconstructed_data import make_reconstructed_data
from survival_analysis import c_index_dataframe,boxplot_all_cancer_improvement_c_index,boxplot_one_cancer_improvement_c_index
from tsne import tsne_plot
from violin import violin_plot
from kaplan_meier import log_rank_p_value, kaplan_meier_plot_complete_form_year,boxplot_one_cancer_improvement_log_rank_p_value, boxplot_all_cancer_improvement_log_rank_p_value
from clustering import clustering_evaluation

### Sample test Code for COAD ####

print("data preprocessing")
data_preprocess()

print("select significant gene")
cox_significant_gene(cancer_type = ['COAD'])
sorted_significant_gene(cancer_type = ['COAD'])

print("model training")
model_train(cancer_type= 'COAD', significant_gene_num=300, epoch_num=500)

print("generate fake data and reconstructed data")
make_fake_data(cancer_type=['COAD'],significant_gene_num=300, train_epoch=500)
make_reconstructed_data(cancer_type=['COAD'],significant_gene_num=300, train_epoch=500)

print("survival analysis - C index")
c_index_dataframe(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch =500, augmentation = 'no')
c_index_dataframe(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch =500, augmentation = 'yes', days_source = 'ctgan')
c_index_dataframe(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch =500, augmentation = 'yes', days_source = 'BRCA')

# print("boxplot for all eleven cancer c index improvement")
# boxplot_all_cancer_improvement_c_index(significant_gene_num = 300, train_epoch = 500)  ### when you have already generated all eleven cancers using ctGAN
print("boxplot for one selected cancer c index improvement")
boxplot_one_cancer_improvement_c_index(cancer='COAD', significant_gene_num=300, train_epoch=500)

print("data distribution evaluation : tnse plot and violin plot")
tsne_plot(cancer_type = ['COAD'], significant_gene_num  = 300, train_epoch = 500)
violin_plot(cancer_type = ['COAD'], significant_gene_num  = 300, train_epoch = 500)

print("kaplan meier long rank p value")
log_rank_p_value(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch = 500, augmentation = 'no')
log_rank_p_value(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch = 500, augmentation = 'yes')

print("kaplan meier plot for selected cancer log rank p value")
kaplan_meier_plot_complete_form_year(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch = 500, augmentation = 'no')
kaplan_meier_plot_complete_form_year(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch = 500, augmentation = 'yes')

# print("boxplot for all eleven cancer log rank p value")
# boxplot_all_cancer_improvement_log_rank_p_value(cancer = 'COAD', significant_gene_num = 300, train_epoch = 500) ### when you have already generated all eleven cancers using ctGAN
print("boxplot for one selected cancer log rank p value")
boxplot_one_cancer_improvement_log_rank_p_value(significant_gene_num = 300, train_epoch = 500)

print("clustering evaluation")
clustering_evaluation(cancer_type = ['COAD'], significant_gene_num  = 300, train_epoch = 500,augmentation= 'no')
clustering_evaluation(cancer_type = ['COAD'], significant_gene_num  = 300, train_epoch = 500,augmentation= 'yes')





