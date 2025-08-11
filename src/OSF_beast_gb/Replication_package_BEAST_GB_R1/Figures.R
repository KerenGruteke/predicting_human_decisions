library(ggplot2)
library(dplyr)
library(ggnewscale)
library(tidyr)
setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload")

######

# Figure 2a - CPC18 BEAST-GB error with different feature sets
# Data
data <- data.frame(
  Model = c("BEAST-GB\n(full)", "Naïve\nfeatures\nremoved", "Psychological\nfeatures\nremoved", "Foresight\n(BEAST)\nremoved"),
  MSE = c(0.005563083, 0.005518492, 0.006601885, 0.01171374), 
  Completeness = 100*c(0.9263483, 0.9270584, 0.9098049, 0.8283964),
  Color <- c('red', 'grey', 'grey', 'grey')
)
data$Model <- factor(data$Model, levels = data$Model)

# Add a gap after the first bar
gap_size <- 0.2
data$X <- seq_along(data$Model)
data$X[2:nrow(data)] <- data$X[2:nrow(data)] + gap_size
# Plot
p <- ggplot(data, aes(x = X, y = MSE, fill=Color)) +
  geom_bar(stat = 'identity', width=0.8) +
  scale_fill_identity() +
  theme_minimal() +
  ylab('Test set MSE') +
  xlab('Model') +
  theme(
    axis.text.x = element_text(size = 14, color="black"),
    axis.text.y = element_text(size = 13, color="black"),
    axis.title.x = element_text(size = 14, color="black"),
    axis.title.y = element_text(size = 14, color="black"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  scale_x_continuous(breaks = data$X, labels = data$Model)+
  scale_y_continuous(breaks = seq(0, 0.014, 0.0025), limits = c(0, max(data$MSE) * 1.2))

# Add MSE and completeness as text above bars, split into three lines
p + geom_text(aes(label = paste0('MSE=', sprintf("%.4f", MSE), '\n', sprintf("%.1f", Completeness), '% comp.'), y = MSE), vjust = -0.35, size = 4.)



# Figure 2b - CPC18 BEAST-GB SHAP values
load('CPC18/SHAP CPC18.RData')
top20 = mean_shap[1:20,]
lookup <- read.csv("feature labels.csv", stringsAsFactors = FALSE)
top20_annot <- top20 %>%
  left_join(lookup, by = "variable")
cat_colors <- c("Objective" = "#0072B2",    
                "Naïve"     = "#009E73",    
                "Psychological" = "#E69F00", 
                "Foresight" = "#F0E442")     
desired_order <- top20_annot %>% 
  arrange(mean_value) %>% 
  pull(intuitive_name)
top20_annot <- top20_annot %>%
  mutate(intuitive_name = factor(intuitive_name, levels = desired_order))

# special treatment of diffMins (stripes)
special_label <- "Δ Min payoffs"
regular_data <- top20_annot %>% filter(intuitive_name != special_label)
special_data <- top20_annot %>% filter(intuitive_name == special_label)
ypos <- as.numeric(special_data$intuitive_name)[1]  
bar_height <- 0.9                                  
xmin_special <- 0
xmax_special <- special_data$mean_value[1]
ymin_special <- ypos - bar_height/2
ymax_special <- ypos + bar_height/2
# grid of tiles over the special bar’s rectangle.
nx <- 100   # number of tiles along x
ny <- 20    # number of tiles along y
x_seq <- seq(xmin_special, xmax_special, length.out = nx)
y_seq <- seq(ymin_special, ymax_special, length.out = ny)
tile_data <- expand.grid(x = x_seq, y = y_seq)
# stripe parameters.
stripe_count <- 6
stripe_width_d <- (xmax_special - xmin_special) / stripe_count
freq <- pi / stripe_width_d
color1 <- "#009E73"  # green
color2 <- "#E69F00"  # orange
tile_data$fill_color <- ifelse(sin((tile_data$x + tile_data$y) / sqrt(2) * freq) > 0, color1, color2)

ggplot() +
  geom_bar(data = regular_data,
           aes(x = mean_value, y = intuitive_name, fill = category),
           stat = "identity", width = bar_height) +
  scale_fill_manual(name = "Feature Category",
                    values = cat_colors,
                    breaks = c("Objective", "Naïve", "Psychological", "Foresight"),
                    guide = guide_legend(nrow = 2, byrow = TRUE)) +
  # Reset fill scale so the tiles can use their own fill values.
  new_scale_fill() +
  # Add the special bar using the tile grid.
  geom_tile(data = tile_data,
            aes(x = x, y = y, fill = fill_color),
            width = (xmax_special - xmin_special)/nx,
            height = (ymax_special - ymin_special)/ny) +
  # Use identity scale so the tile fill colors are used as provided.
  scale_fill_identity() +
  scale_y_discrete(limits = desired_order) +
  labs(x = "Mean |SHAP| Value", y = "Feature (Descriptive Label)") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14),
    legend.text = element_text(size=14),
    legend.justification = c(1.6,0)
  )



# Figure S1 - CPC18 BEAST-GB vs other models in predicting CPC18 decisions under risk
# Data
data <- data.frame(
  Model = c('BEAST', 'Stochastic CPT', 'Deterministic CPT', 'Decision by Sampling', 'Priority heuristic', 'Ensemble of foresights'),
  MSE = c(0.006321442, 0.01745834, 0.02426323, 0.02916444, 0.03058142, 0.006811105)
)
MSE_naive = 0.05095104
MSE_irreducible = 0.001126812
data$Completeness <- with(data, 100*(MSE_naive - MSE) / (MSE_naive - MSE_irreducible))
data$Color <- c('red', 'grey', 'grey', 'grey', 'grey', 'rosybrown')
data$Model <- c("BEAST", 'Stochastic\nCPT', 'Deterministic\nCPT', 'Decision by\nSampling', 'Priority\nheuristic', 'All (Ensemble\nof foresights)')
data$Model <- factor(data$Model, levels = data$Model)

gap_size <- 0.2
data$X <- seq_along(data$Model)
data$X[2:nrow(data)] <- data$X[2:nrow(data)] + gap_size
data$X[nrow(data)] <- data$X[nrow(data)] + gap_size

# Plot
p <- ggplot(data, aes(x = X, y = MSE, fill=Color)) +
  geom_bar(stat = 'identity', width=0.8) +
  scale_fill_identity() +
  theme_minimal() +
  ylab('Test set MSE (only decisions under risk w/o feedback)') +
  xlab('Model used as foresight feature') +
  theme(
    axis.text.x = element_text(size = 13, color = "black"),
    axis.text.y = element_text(size = 13, color = "black"),
    axis.title.x = element_text(size = 14, color = "black"),
    axis.title.y = element_text(size = 14, color = "black"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  ylim(0, max(data$MSE) * 1.3) +
  scale_x_continuous(breaks = data$X, labels = data$Model) # Control breaks and labels for the x-axis

# Add MSE and completeness as text above bars, split into three lines
p + geom_text(aes(label = paste0('MSE=', sprintf("%.4f", MSE), '\n', sprintf("%.1f", Completeness), '% comp.'), y = MSE), vjust = -0.35, size = 4)


# Figure 3 - 13k performance as function of training data
load('Choices13k/C13k R1 data for figures.RData')
names(plot_data)[1:2] = c('perc_train','mse')
plot_data$model = "BEAST-GB"
neural_pt = read.csv('Choices13k/neural pt train curve from figure.csv')
neural_pt$model = "Neural PT"
ctxt_dep = read.csv('Choices13k/context_dependent train curve from figure.csv')
ctxt_dep$model = "Context dependant"
train_df = bind_rows(plot_data, neural_pt, ctxt_dep)
ggplot(train_df, aes(x = perc_train, y = mse, color = model)) +
  geom_line(linewidth = 1.5) +
  theme_minimal() +
  ylab('Test Set MSE') +
  xlab('Proportion of Training Data') +
  scale_color_brewer(palette = "Dark2") +
  theme(
    axis.text = element_text(size = 14),     # Font size for axis text (ticks)
    axis.title = element_text(size = 14),    # Font size for axis labels
    legend.title = element_text(size = 14),  # Font size for legend title
    legend.text = element_text(size = 14)    # Font size for legend text
  ) +
  labs(color = "Model") + 
  scale_y_continuous(breaks = seq(0.005, 0.035, by = 0.005)) 


# Figure S2a: 13k training performance with different feature sets
load('Choices13k/C13k R1 data for figures.RData')
train_df = reshape(plot_data, direction="long",varying = 2:7, 
                   times = c("BEAST-GB (Full model)","No Objective features","No BEAST feature","No Psychological features","No Naive features","No Psychological or BEAST features"),
                   v.names = "mse", timevar = "model")
cat_colors <- c("BEAST-GB (Full model)" = "#000000",    
                "No Psychological or BEAST features" = "#F0E442",     # green
                "No BEAST feature" = "#CC79A7", # orange
                "No Psychological features" = "#009E73",
                "No Naive features" = "#E69F00",
                "No Objective features" = "#0072B2"
                )     
alphas <- c("BEAST-GB (Full model)" = 0.75,    
                "No Psychological or BEAST features" = 1,     # green
                "No BEAST feature" = 1, 
                "No Psychological features" = 0.87,
                "No Naive features" = 0.87,
                "No Objective features" = 1
            )
train_df$model <- factor(train_df$model, 
                         levels = c("No Psychological or BEAST features", "No Psychological features", 
                                    "No BEAST feature", "No Objective features", 
                                    "No Naive features", "BEAST-GB (Full model)"))
ggplot(train_df, aes(x = props_train, y = mse, color = model, alpha = model)) +
  geom_line(linewidth = 1.9) +
  theme_minimal() +
  ylab('Test Set MSE') +
  xlab('Proportion of Training Data') +
  scale_color_manual(values = cat_colors) +
  scale_alpha_manual(values = alphas) +
  guides(alpha = "none") +
  theme(
    axis.text = element_text(size = 16, color = 'black'),  
    axis.title = element_text(size = 16, color = 'black'), 
    legend.title = element_text(size = 16, color = 'black'),
    legend.text = element_text(size = 15, color = 'black')  
  ) +
  labs(color = "Model")+  
  scale_y_continuous(breaks = seq(0.005, 0.02, by = 0.0025), limits = c(0.0082, 0.0162))



# Figure S2b - 13k BEAST-GB SHAP values
load('Choices13k/C13k R1 data for figures.RData')
top20 = mean_shap[1:20,]
lookup <- read.csv("feature labels.csv", stringsAsFactors = FALSE)
top20_annot <- top20 %>%
  left_join(lookup, by = "variable")
cat_colors <- c("Objective" = "#0072B2",     # dark blue
                "Naïve"     = "#009E73",     # green
                "Psychological" = "#E69F00", # orange
                "Foresight" = "#F0E442")     # yellow
desired_order <- top20_annot %>% 
  arrange(mean_value) %>% 
  pull(intuitive_name)
top20_annot <- top20_annot %>%
  mutate(intuitive_name = factor(intuitive_name, levels = desired_order))

# special treatment of diffMins (stripes)
special_label <- "Δ Min payoffs"
regular_data <- top20_annot %>% filter(intuitive_name != special_label)
special_data <- top20_annot %>% filter(intuitive_name == special_label)
ypos <- as.numeric(special_data$intuitive_name)[1]  
bar_height <- 0.9                                  
xmin_special <- 0
xmax_special <- special_data$mean_value[1]
ymin_special <- ypos - bar_height/2
ymax_special <- ypos + bar_height/2
nx <- 100   # number of tiles along x
ny <- 37    # number of tiles along y
x_seq <- seq(xmin_special, xmax_special, length.out = nx)
y_seq <- seq(ymin_special, ymax_special, length.out = ny)
tile_data <- expand.grid(x = x_seq, y = y_seq)
stripe_count <- 6
stripe_width_d <- (xmax_special - xmin_special) / stripe_count
freq <- pi / stripe_width_d

color1 <- "#009E73"  # green
color2 <- "#E69F00"  # orange
tile_data$fill_color <- ifelse(sin((tile_data$x + tile_data$y) / sqrt(2) * freq) > 0, color1, color2)

ggplot() +
  # Plot regular bars (using the category fill scale)
  geom_bar(data = regular_data,
           aes(x = mean_value, y = intuitive_name, fill = category),
           stat = "identity", width = bar_height) +
  scale_fill_manual(name = "Feature Category",
                    values = cat_colors,
                    breaks = c("Objective", "Naïve", "Psychological", "Foresight"),
                    guide = guide_legend(nrow = 2, byrow = TRUE)) +
  # Reset fill scale so the tiles can use their own fill values.
  new_scale_fill() +
  # Add the special bar using the tile grid.
  geom_tile(data = tile_data,
            aes(x = x, y = y, fill = fill_color),
            width = (xmax_special - xmin_special)/nx,
            height = (ymax_special - ymin_special)/ny) +
  # Use identity scale so the tile fill colors are used as provided.
  scale_fill_identity() +
  # Draw a border around the special bar.
  geom_rect(data = special_data,
            aes(xmin = xmin_special, xmax = xmax_special, ymin = ymin_special, ymax = ymax_special),
            fill = NA) +
  # Ensure the y-axis preserves our ordering.
  scale_y_discrete(limits = desired_order) +
  labs(x = "Mean |SHAP| Value", y = "Feature (Descriptive Label)") +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 16),
    legend.justification = c(0.2,0),
    legend.text = element_text(size = 16)
  )



# Figure 4: HAB22 main analysis
load('HAB22/HAB22 R1 data for figures.RData')
BEAST_GB = mean(mean(mse_xgb))
sem_beast_gb = sd(mse_xgb, na.rm = TRUE) / sqrt(sum(!is.na(mse_xgb)))
data <- data.frame(
  Model = names(mean_mse),
  Value = as.numeric(mean_mse),
  SEM = as.numeric(sem_mse)
)
data <- data[!is.nan(data$Value),]
data$Model[data$Model == "BEASTpred"] <- "BEAST (not re-trained)"
data <- rbind(data, data.frame(Model = "BEAST-GB", Value = BEAST_GB, SEM = sem_beast_gb))
data$Color <- ifelse(data$Model == "BEAST-GB", "red", ifelse(data$Model == "BEAST (not re-trained)", "rosybrown", "grey"))
data = data[data$Model != "Baseline",]
p <- ggplot(data, aes(x = reorder(Model, Value), y = Value, fill = Color)) +
  geom_bar(stat = 'identity') +
  geom_errorbar(aes(ymin = Value - SEM, ymax = Value + SEM), width = 0, linewidth=0.85) +
  scale_fill_identity() +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 12.5, color='black'),
    axis.title = element_text(size = 15, color = 'black'),
    axis.text.y = element_text(size =15, color = 'black'),
    panel.grid.major.x = element_blank()
  ) +
  xlab('Model') +
  ylab('Test set MSE') +
  scale_y_continuous(minor_breaks = seq(0,0.14,0.02), breaks = seq(0,0.12,0.04))
p


# Figure S4b - HAB22 BEAST-GB SHAP values
load('HAB22/HAB22 R1 data for figures.RData')
mean_shap = unique(shap_long_long[,c(2,6)])
mean_shap = mean_shap[order(-mean_shap$mean_value),]
top20 = mean_shap[1:20,]
lookup <- read.csv("feature labels.csv", stringsAsFactors = FALSE)
top20_annot <- top20 %>%
  left_join(lookup, by = "variable")
cat_colors <- c("Objective" = "#0072B2",     # dark blue
                "Naïve"     = "#009E73",     # green
                "Psychological" = "#E69F00", # orange
                "Foresight" = "#F0E442")     # yellow
desired_order <- top20_annot %>% 
  arrange(mean_value) %>% 
  pull(intuitive_name)
top20_annot <- top20_annot %>%
  mutate(intuitive_name = factor(intuitive_name, levels = desired_order))

# special treatment of diffMins (stripes)
special_label <- "Δ Min payoffs"
regular_data <- top20_annot %>% filter(intuitive_name != special_label)
special_data <- top20_annot %>% filter(intuitive_name == special_label)
ypos <- as.numeric(special_data$intuitive_name)[1]  
bar_height <- 0.9                                  
xmin_special <- 0
xmax_special <- special_data$mean_value[1]
ymin_special <- ypos - bar_height/2
ymax_special <- ypos + bar_height/2
nx <- 100   # number of tiles along x
ny <- 49    # number of tiles along y
x_seq <- seq(xmin_special, xmax_special, length.out = nx)
y_seq <- seq(ymin_special, ymax_special, length.out = ny)
tile_data <- expand.grid(x = x_seq, y = y_seq)
stripe_count <- 6
stripe_width_d <- (xmax_special - xmin_special) / stripe_count
freq <- pi / stripe_width_d

color1 <- "#009E73"  # green
color2 <- "#E69F00"  # orange
tile_data$fill_color <- ifelse(sin((tile_data$x + tile_data$y) / sqrt(2) * freq) > 0, color1, color2)

ggplot() +
  # Plot regular bars (using the category fill scale)
  geom_bar(data = regular_data,
           aes(x = mean_value, y = intuitive_name, fill = category),
           stat = "identity", width = bar_height) +
  scale_fill_manual(name = "Feature Category",
                    values = cat_colors,
                    breaks = c("Objective", "Naïve", "Psychological", "Foresight"),
                    guide = guide_legend(nrow = 2, byrow = TRUE)) +
  # Reset fill scale so the tiles can use their own fill values.
  new_scale_fill() +
  # Add the special bar using the tile grid.
  geom_tile(data = tile_data,
            aes(x = x, y = y, fill = fill_color),
            width = (xmax_special - xmin_special)/nx,
            height = (ymax_special - ymin_special)/ny) +
  # Use identity scale so the tile fill colors are used as provided.
  scale_fill_identity() +
  # Draw a border around the special bar.
  geom_rect(data = special_data,
            aes(xmin = xmin_special, xmax = xmax_special, ymin = ymin_special, ymax = ymax_special),
            fill = NA) +
  # Ensure the y-axis preserves our ordering.
  scale_y_discrete(limits = desired_order) +
  labs(x = "Mean |SHAP| Value", y = "Feature (Descriptive Label)") +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 12.5, color='black'),
    legend.justification = c(1.2,0),
    legend.text = element_text(size = 13),
    legend.title = element_text(size=14),
    axis.title = element_text(size = 14)
  )


# Figure S4a - HAB22 with different features sets
# Data
load('HAB22/HAB22 R1 data for figures.RData')
data <- data.frame(value = c(mse_xgb, mse_xgb_no_naive, mse_xgb_no_Psych, mse_xgb_no_BEAST),
                   model = rep(c("BEAST-GB\n(full)", "Naïve\nfeatures\nremoved", "Psychological\nfeatures\nremoved", "Foresight\n(BEAST)\nremoved"), each = 50),
                   completeness = rep(c(completness_xgb, completness_xgb_no_naive, completness_xgb_no_Psych, completness_xgb_no_BEAST), each = 50),
                   color = rep(c('red', 'grey', 'grey', 'grey'), each = 50))
data_summary <- data %>%
  group_by(model) %>%
  summarize(mean = mean(value),
            lower = mean - 1. * sd(value)/sqrt(n()),
            upper = mean + 1. * sd(value)/sqrt(n()),
            completeness = first(completeness),
            color = first(color))
data_summary$Model <- factor(data_summary$model, levels = c("BEAST-GB\n(full)", "Naïve\nfeatures\nremoved", "Psychological\nfeatures\nremoved", "Foresight\n(BEAST)\nremoved"))
data_summary <- data_summary %>% arrange(Model)

# Add a gap after the first bar
gap_size <- 0.2
data_summary$X <- seq_along(data_summary$Model)
data_summary$X[2:nrow(data_summary)] <- data_summary$X[2:nrow(data_summary)] + gap_size

# Plot
p <- ggplot(data_summary[1:4,], aes(x = X, y = mean, fill=color)) +
  geom_bar(stat = 'identity', width=0.8) +
  scale_fill_identity() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2)+
  theme_minimal() +
  ylab('Test set MSE') +
  xlab('Model') +
  theme(
    axis.text.x = element_text(size = 14, color='black'),
    axis.text.y = element_text(size = 13),
    axis.title = element_text(size = 14),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  scale_x_continuous(breaks = data_summary$X, labels = data_summary$Model)+
  scale_y_continuous(breaks = seq(0, 0.035, 0.01), limits = c(0, max(data_summary$mean) * 1.2))

# Add MSE and completeness as text above bars, split into three lines
p + geom_text(aes(label = paste0('MSE=', sprintf("%.4f", mean), '\n', sprintf("%.1f", 100*completeness), '% comp.'), y = mean), vjust = -0.65, size = 4.5)



# Figure SI.1: HAB22 analysis in-sample aggregate prediction
load('HAB22/HAB22 R1 data for figures.RData')
BEAST_GB = mean(mean(mse_xgb_in_sample))
sem_beast_gb = sd(mse_xgb_in_sample, na.rm = TRUE) / sqrt(sum(!is.na(mse_xgb_in_sample)))
data <- data.frame(
  Model = names(mean_mse_in_sample),
  Value = as.numeric(mean_mse_in_sample),
  SEM = as.numeric(sem_mse_in_sample)
)
data <- data[!is.nan(data$Value),]
data$Model[data$Model == "BEASTpred"] <- "BEAST (not re-trained)"
data <- rbind(data, data.frame(Model = "BEAST-GB", Value = BEAST_GB, SEM = sem_beast_gb))
data$Color <- ifelse(data$Model == "BEAST-GB", "red", ifelse(data$Model == "BEAST (not re-trained)", "rosybrown", "grey"))
data = data[data$Model != "Baseline",]
p <- ggplot(data, aes(x = reorder(Model, Value), y = Value, fill = Color)) +
  geom_bar(stat = 'identity') +
  geom_errorbar(aes(ymin = Value - SEM, ymax = Value + SEM), width = 0, linewidth=1) +
  scale_fill_identity() +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 12.5, color='black'),
    axis.title = element_text(size = 14),
    axis.text.y = element_text(size =14),
    panel.grid.major.x = element_blank()
  ) +
  xlab('Model') +
  ylab('Test set MSE')+
  scale_y_continuous(minor_breaks = seq(0,0.12,0.01), breaks = seq(0,0.12,0.02))
p



# Figure 5: HAB22 domain generalization main analysis
load('Context generalization/Context R1 - data for figure.RData')
BEAST_GB = mean(pred_all$sq_diff)
sem_beast_gb = sd(pred_all$sq_diff, na.rm = TRUE) / sqrt(sum(!is.na(pred_all$sq_diff)))
data <- data.frame(
  Model = names(mean_squared_differences),
  Value = as.numeric(mean_squared_differences),
  SEM = as.numeric(sem_squared_differences)
)
data <- data[!is.nan(data$Value),]
data <- rbind(data, data.frame(Model = "BEAST-GB", Value = BEAST_GB, SEM = sem_beast_gb))
data$Model[data$Model == "choice"] = "Other experiments"
data$Color <- ifelse(data$Model == "BEAST-GB", "red", ifelse(data$Model == "Other experiments", "rosybrown", "grey"))
data = data[data$Model != "Baseline",]
data = data[order(data$Value),]

# Add a gap after the first bar
gap_size <- 0.25
data$X <- seq_along(data$Model)
data$X[2:nrow(data)] <- data$X[2:nrow(data)] + gap_size
data$X[3:nrow(data)] <- data$X[3:nrow(data)] + gap_size
bold_labels <- ifelse(data$Model == "Other experiments", "bold", ifelse(data$Model == "BEAST-GB", "bold","plain"))

p <- ggplot(data, aes(x = X, y = Value, fill=Color)) +
  geom_bar(stat = 'identity', width=0.8) +
  geom_errorbar(aes(ymin = Value - SEM, ymax = Value + SEM), width = 0, linewidth=0.9) +
  scale_fill_identity() +
  theme_minimal() +
  ylab('Context generalization MSE') +
  xlab('Model') +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 12.5, color='gray24', face = bold_labels),
    axis.text.y = element_text(size = 15),
    axis.title = element_text(size = 15),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  scale_x_continuous(breaks = data$X, labels = data$Model) # Control breaks and labels for the x-axis

p 



# Figure SI.2
load("HAB22/Individual analyses/Individuals R1 all mses.RData")

# Remove columns with any NAs and drop "MSE_Baseline"
subject_mse_clean <- subject_mse %>%
  select(-MSE_Baseline) %>% 
  select(subj_id, where(~ all(!is.na(.))))

# Pivot from wide to long format (excluding subj_id)
mse_long <- subject_mse_clean %>%
  pivot_longer(cols = -subj_id, names_to = "model", values_to = "mse")

# Remove "MSE_" prefix from model names
mse_long <- mse_long %>%
  mutate(model = gsub("^MSE_", "", model))

# Compute mean and standard error for each model
mse_summary <- mse_long %>%
  group_by(model) %>%
  summarise(mean_mse = mean(mse),
            se = sd(mse) / sqrt(n())) %>%
  ungroup()

# Create display names and custom fill colors
mse_summary <- mse_summary %>%
  mutate(model_display = case_when(
    model == "Bayes_reg" ~ "Mixed-effects Regression",
    model == "RFs" ~ "Individual Random Forests",
    model == "BEAST_GB_pred" ~ "BEAST-GB (population model)",
    model == "BEASTpred" ~ "BEAST (population model)",
    TRUE ~ model
  ),
  fillcolor = case_when(
    model == "Bayes_reg" ~ "red",
    model == "RFs" ~ "red",
    model == "BEASTpred" ~ "rosybrown",
    model == "BEAST_GB_pred" ~ "rosybrown",
    TRUE ~ "grey"
  ))

mse_summary$model_display <- factor(mse_summary$model_display,
                                    levels = mse_summary$model_display[order(mse_summary$mean_mse)])

ggplot(mse_summary, aes(x = model_display, y = mean_mse, fill = fillcolor)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = mean_mse - se, ymax = mean_mse + se), width = 0, linewidth=0.9) +
  scale_fill_identity() +
  labs(x = "Model", y = "Mean Subject MSE") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.48, size = 12.5, color='black'),
    axis.text.y = element_text(size =14),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    panel.grid.major.x = element_blank()
    )

