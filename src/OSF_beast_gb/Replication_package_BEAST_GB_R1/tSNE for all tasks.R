library(ggplot2)
library(cluster) # for gower's distance
library(Rtsne)
library(fastDummies)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload")

load('data for tSNE.RData')

# 1. Data Preprocessing
color_values <- all_tasks_psych$dataset 
all_tasks_psych$dataset <- NULL
numeric_cols <- sapply(all_tasks_psych, is.numeric)
all_tasks_psych[numeric_cols] <- scale(all_tasks_psych[numeric_cols])
all_tasks_psych_dummy <- dummy_cols(all_tasks_psych)
factor_cols <- sapply(all_tasks_psych, is.factor)
all_tasks_psych_dummy <- all_tasks_psych_dummy[, !names(all_tasks_psych_dummy) %in% names(all_tasks_psych[factor_cols])]

# 2. Distance Computation
dist_matrix_psych <- daisy(all_tasks_psych_dummy, metric = "gower")

# 3. t-SNE computation
set.seed(42)
tsne_result_psych <- Rtsne(dist_matrix_psych, is_distance = TRUE, perplexity = 40,  check_duplicates = FALSE)

# 4. plot
plot_data <- data.frame(
  x = tsne_result_psych$Y[, 1],
  y = tsne_result_psych$Y[, 2]
)
plot_data$dataset = ifelse(color_values=="CPC18", "CPC18",
                           ifelse(color_values=="Choices13k", "Choices13k", "HAB22"))

ggplot(plot_data, aes(x = x, y = y, color = dataset, alpha=0.75, shape=dataset)) + 
  geom_point(size = 1.9) + 
  scale_color_manual(name = "Dataset", values = c("#BBBBBB", "#FF0000", "#00ccFF")) +
  scale_shape_manual(name = "Dataset", values = c(1,17,18)) +
  scale_alpha_identity() +
  labs(color = "Dataset", x="", y="", shape="") + 
  theme_minimal() + 
  theme(
    legend.text = element_text(size = 14),        
    legend.title = element_text(size = 14, face = "bold")  
  )

